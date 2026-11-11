"""
End-to-end training pipeline for the AI-based Intrusion Detection System.

Reproduces every artifact used by the dashboard and the report:

    python src/pipeline.py --data "C:/path/to/MachineLearningCVE" --out .

Stages (run all by default, or pick with --stages):
    sample      chunked read -> clean -> per-class capped working sample
    binary      constant/correlation feature reduction + binary Random Forest
    multiclass  multiclass Random Forest on the same selected features
    compare     Logistic Regression / Random Forest / XGBoost / Isolation Forest
    crossday    train on Mon-Thu, test on Friday (generalisation check)
    figures     PNG figures for the report

The script is memory-friendly: files are streamed in chunks and only a random
sample of at most --per-class rows per label is kept in memory.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TEST_SIZE = 0.15
CORR_THRESHOLD = 0.995
CHUNK = 200_000
LABEL = "Label"


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #
def normalize_label(s: pd.Series) -> pd.Series:
    """Repair the en-dash encoding artefacts in the web-attack labels."""
    s = s.astype(str).str.strip()
    for bad in ("ï¿½", "â€“", "�", "–", "\x96"):
        s = s.str.replace(bad, "-", regex=False)
    return (s.str.replace(r"^Web Attack\s*-\s*", "Web Attack - ", regex=True)
             .str.replace(r"\s+", " ", regex=True))


def clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]          # 'Fwd Header Length' appears twice
    df[LABEL] = normalize_label(df[LABEL])
    feats = [c for c in df.columns if c != LABEL]
    df[feats] = (df[feats].apply(pd.to_numeric, errors="coerce")
                 .replace([np.inf, -np.inf], np.nan).astype("float32"))
    return df.drop_duplicates()


def iter_files(data_dir: Path, include=None, exclude=None):
    for f in sorted(data_dir.glob("*.csv")):
        name = f.name.lower()
        if include and not any(k in name for k in include):
            continue
        if exclude and any(k in name for k in exclude):
            continue
        yield f


def capped_sample(files, per_class: int | None, seed: int = SEED) -> pd.DataFrame:
    """Stream files and keep a uniform random sample of <= per_class rows per label.

    Each row gets a random key; for every label only the rows with the smallest
    keys are kept, which gives a uniform sample without holding everything.
    """
    rng = np.random.default_rng(seed)
    kept: dict[str, pd.DataFrame] = {}
    raw_counts: dict[str, int] = {}
    for f in files:
        t = time.time()
        for chunk in pd.read_csv(f, chunksize=CHUNK, encoding="latin-1", low_memory=False):
            chunk = clean_chunk(chunk)
            chunk["_key"] = rng.random(len(chunk))
            for lab, g in chunk.groupby(LABEL, sort=False):
                raw_counts[lab] = raw_counts.get(lab, 0) + len(g)
                cur = pd.concat([kept[lab], g]) if lab in kept else g
                if per_class is not None and len(cur) > per_class:
                    cur = cur.nsmallest(per_class, "_key")
                kept[lab] = cur
        print(f"  read {f.name} ({time.time() - t:.0f}s)", flush=True)
    out = pd.concat(kept.values(), ignore_index=True).drop(columns="_key").drop_duplicates()
    out.attrs["raw_counts"] = raw_counts
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature reduction
# --------------------------------------------------------------------------- #
def select_features(X_train: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    constant = [c for c in X_train.columns if X_train[c].nunique(dropna=True) <= 1]
    X = X_train.drop(columns=constant)
    corr = X.corr().abs()                               # pairwise-complete correlation
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    correlated = [c for c in upper.columns if (upper[c] > CORR_THRESHOLD).any()]
    selected = [c for c in X.columns if c not in correlated]
    return selected, constant, correlated


def binary_metrics(y_true, y_pred, score) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, score),
    }


def make_rf(n: int) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=n, class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=SEED)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
class Pipeline:
    def __init__(self, data_dir: Path, out: Path, per_class: int):
        self.data_dir, self.out, self.per_class = data_dir, out, per_class
        self.res = out / "results"
        self.res.mkdir(parents=True, exist_ok=True)
        self.sample_path = self.res / "working_sample.parquet"

    # -- helpers ------------------------------------------------------------ #
    def load_sample(self) -> pd.DataFrame:
        if not self.sample_path.exists():
            self.stage_sample()
        return pd.read_parquet(self.sample_path)

    def split(self, df):
        X = df.drop(columns=LABEL)
        y = df[LABEL]
        return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

    def save_json(self, name, obj):
        (self.res / name).write_text(json.dumps(obj, indent=2, default=float))

    # -- stages ------------------------------------------------------------- #
    def stage_sample(self):
        print(f"[sample] streaming CSVs, cap {self.per_class:,} rows/class")
        df = capped_sample(iter_files(self.data_dir), self.per_class)
        raw = df.attrs["raw_counts"]
        df.to_parquet(self.sample_path, index=False)
        dist = pd.DataFrame({"deduplicated_rows": pd.Series(raw),
                             "working_sample": df[LABEL].value_counts()}).fillna(0).astype(int)
        dist.sort_values("deduplicated_rows", ascending=False).to_csv(self.res / "class_distribution.csv")
        print(f"[sample] {len(df):,} rows, {df[LABEL].nunique()} labels")

    def stage_binary(self):
        df = self.load_sample()
        X_tr, X_te, y_tr, y_te = self.split(df)
        selected, constant, correlated = select_features(X_tr)
        print(f"[binary] constant={len(constant)} correlated={len(correlated)} selected={len(selected)}")
        imp = SimpleImputer(strategy="median").fit(X_tr[selected])
        yb_tr, yb_te = (y_tr != "BENIGN").astype(int), (y_te != "BENIGN").astype(int)
        rf = make_rf(200).fit(imp.transform(X_tr[selected]), yb_tr)
        Xt = imp.transform(X_te[selected])
        pred, prob = rf.predict(Xt), rf.predict_proba(Xt)[:, 1]
        m = binary_metrics(yb_te, pred, prob)
        print("[binary]", {k: round(v, 4) for k, v in m.items()})

        joblib.dump(rf, self.out / "random_forest_binary.joblib", compress=3)
        joblib.dump(imp, self.out / "median_imputer.joblib")
        joblib.dump(selected, self.out / "selected_features.joblib")
        cm = confusion_matrix(yb_te, pred)
        pd.DataFrame(cm, index=["actual_benign", "actual_attack"],
                     columns=["pred_benign", "pred_attack"]).to_csv(self.out / "confusion_matrix_binary.csv")
        (pd.Series(rf.feature_importances_, index=selected, name="importance")
         .sort_values(ascending=False).to_csv(self.out / "feature_importance_binary.csv"))
        self.save_json("feature_selection.json", {"constant": constant, "correlated": correlated,
                                                  "selected": selected, "threshold": CORR_THRESHOLD})
        self.save_json("binary_metrics.json", m)

    def stage_multiclass(self):
        df = self.load_sample()
        X_tr, X_te, y_tr, y_te = self.split(df)
        selected = joblib.load(self.out / "selected_features.joblib")
        imp = SimpleImputer(strategy="median").fit(X_tr[selected])
        rf = make_rf(250).fit(imp.transform(X_tr[selected]), y_tr)
        pred = rf.predict(imp.transform(X_te[selected]))
        acc, mf1 = accuracy_score(y_te, pred), f1_score(y_te, pred, average="macro")
        print(f"[multiclass] accuracy={acc:.4f} macro_f1={mf1:.4f}")
        joblib.dump(rf, self.out / "random_forest_multiclass.joblib", compress=3)
        joblib.dump(imp, self.out / "multiclass_imputer.joblib")
        joblib.dump(selected, self.out / "multiclass_features.joblib")
        labels = list(rf.classes_)
        pd.DataFrame(confusion_matrix(y_te, pred, labels=labels), index=labels, columns=labels) \
            .to_csv(self.out / "confusion_matrix_multiclass.csv")
        rep = classification_report(y_te, pred, output_dict=True, zero_division=0)
        pd.DataFrame(rep).T.to_csv(self.res / "multiclass_report.csv")
        self.save_json("multiclass_metrics.json", {"accuracy": acc, "macro_f1": mf1})

    def stage_compare(self):
        from xgboost import XGBClassifier
        df = self.load_sample()
        X_tr, X_te, y_tr, y_te = self.split(df)
        selected = joblib.load(self.out / "selected_features.joblib")
        imp = SimpleImputer(strategy="median").fit(X_tr[selected])
        Xtr, Xte = imp.transform(X_tr[selected]), imp.transform(X_te[selected])
        ytr, yte = (y_tr != "BENIGN").astype(int).values, (y_te != "BENIGN").astype(int).values
        rows = []

        models = {
            "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=SEED)),
            "Random Forest": make_rf(200),
            "XGBoost": XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                     subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                                     eval_metric="logloss", n_jobs=-1, random_state=SEED),
        }
        for name, model in models.items():
            t = time.time()
            model.fit(Xtr, ytr)
            m = binary_metrics(yte, model.predict(Xte), model.predict_proba(Xte)[:, 1])
            rows.append({"Model": name, **m})
            print(f"[compare] {name}: F1={m['F1']:.4f} ({time.time() - t:.0f}s)")

        # Unsupervised: learn "normal" from benign training traffic only.
        iso = IsolationForest(n_estimators=200, contamination="auto", random_state=SEED, n_jobs=-1)
        iso.fit(Xtr[ytr == 0])
        anomaly = -iso.score_samples(Xte)
        m = binary_metrics(yte, (iso.predict(Xte) == -1).astype(int), anomaly)
        rows.append({"Model": "Isolation Forest", **m})
        print(f"[compare] Isolation Forest: F1={m['F1']:.4f}")
        pd.DataFrame(rows).to_csv(self.out / "model_comparison.csv", index=False)

    def stage_crossday(self):
        selected = joblib.load(self.out / "selected_features.joblib")
        print("[crossday] training sample from Monday-Thursday")
        train = capped_sample(iter_files(self.data_dir, exclude=["friday"]), self.per_class)
        print("[crossday] full Friday test set")
        test = capped_sample(iter_files(self.data_dir, include=["friday"]), None)
        imp = SimpleImputer(strategy="median").fit(train[selected])
        rf = make_rf(200).fit(imp.transform(train[selected]), (train[LABEL] != "BENIGN").astype(int))
        Xte = imp.transform(test[selected])
        yte = (test[LABEL] != "BENIGN").astype(int)
        pred, prob = rf.predict(Xte), rf.predict_proba(Xte)[:, 1]
        m = binary_metrics(yte, pred, prob)
        print("[crossday]", {k: round(v, 4) for k, v in m.items()})
        pd.DataFrame([m]).rename(columns=str.lower).to_csv(self.out / "cross_day_binary_metrics.csv", index=False)
        per_label = (pd.DataFrame({"label": test[LABEL].values, "detected": pred})
                     .groupby("label")["detected"].agg(["count", "mean"])
                     .rename(columns={"count": "flows", "mean": "flagged_as_attack_rate"}))
        per_label.to_csv(self.res / "cross_day_per_label.csv")
        print(per_label)

    def stage_figures(self):
        from make_figures import make_all
        make_all(self.out, self.out / "report" / "figures")


STAGES = ["sample", "binary", "multiclass", "compare", "crossday", "figures"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path, help="folder with the 8 CIC-IDS2017 CSV files")
    ap.add_argument("--out", default=Path(__file__).resolve().parents[1], type=Path)
    ap.add_argument("--per-class", default=20_000, type=int)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    a = ap.parse_args()
    p = Pipeline(a.data, a.out, a.per_class)
    for s in a.stages:
        t = time.time()
        getattr(p, f"stage_{s}")()
        print(f"== {s} done in {time.time() - t:.0f}s\n", flush=True)


if __name__ == "__main__":
    main()
