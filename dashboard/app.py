"""Streamlit dashboard for the AI-based Intrusion Detection System.

Run from the project root:
    streamlit run dashboard/app.py
"""
from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore", category=UserWarning)   # sklearn version pickle notices

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from pipeline import normalize_label  # noqa: E402
FIG = ROOT / "report" / "figures"
SAMPLE = ROOT / "data" / "sample_flows.csv"

st.set_page_config(page_title="AI Intrusion Detection System", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------- #
# Artifacts
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading binary model…")
def load_binary():
    return (joblib.load(ROOT / "random_forest_binary.joblib"),
            joblib.load(ROOT / "median_imputer.joblib"),
            joblib.load(ROOT / "selected_features.joblib"))


@st.cache_resource(show_spinner="Loading multiclass model…")
def load_multiclass():
    return (joblib.load(ROOT / "random_forest_multiclass.joblib"),
            joblib.load(ROOT / "multiclass_imputer.joblib"),
            joblib.load(ROOT / "multiclass_features.joblib"))


@st.cache_data
def read_csv(data: bytes | str) -> pd.DataFrame:
    src = data if isinstance(data, str) else io.BytesIO(data)
    df = pd.read_csv(src, low_memory=False, encoding="latin-1")
    df.columns = df.columns.str.strip()
    df = df.loc[:, ~df.columns.duplicated()]
    if "Label" in df.columns:
        df["Label"] = normalize_label(df["Label"])
    return df


def prepare(df: pd.DataFrame, features: list[str], imputer) -> np.ndarray:
    X = df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return imputer.transform(X)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
st.sidebar.title("🛡️ AI-IDS")
mode = st.sidebar.radio("Detection mode", ["Binary: Normal vs Attack", "Multiclass: Attack type"])
binary = mode.startswith("Binary")
if binary:
    threshold = st.sidebar.slider("Alert threshold (attack probability)", 0.05, 0.95, 0.50, 0.05,
                                  help="Lower = more sensitive (more alerts, more false positives).")
else:
    threshold = st.sidebar.slider("Minimum confidence to raise an alert", 0.0, 0.95, 0.0, 0.05,
                                  help="Attack predictions below this confidence are marked 'REVIEW'.")
max_rows = st.sidebar.number_input("Rows shown in tables", 50, 5000, 200, 50)
st.sidebar.caption("Models: Random Forest trained on CIC-IDS2017 (57 flow features).")

st.title("AI-Based Intrusion Detection System")
st.caption("CIC-IDS2017 • Machine-learning network-flow analysis")

tab_detect, tab_perf, tab_about = st.tabs(["🔍 Detection", "📊 Model performance", "ℹ️ About"])

# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #
def render_detection():
    c_up, c_sample = st.columns([3, 1])
    file = c_up.file_uploader("Upload a CIC-IDS2017-compatible CSV", type=["csv"])
    use_sample = False
    if SAMPLE.exists():
        c_sample.write("")
        c_sample.write("")
        use_sample = c_sample.toggle("Use bundled sample", key="sample_on",
                                     help="Ignored while a file is uploaded.")
        c_sample.download_button("Download sample CSV", SAMPLE.read_bytes(), SAMPLE.name, "text/csv")

    if file is None and not use_sample:
        st.info("Upload a CSV of network flows (CICFlowMeter format) or switch on the bundled sample.")
        st.markdown("**Pipeline:** CSV → clean (∞/NaN → median) → 57 selected features → "
                    "Random Forest → prediction, probability and alert")
        return

    df = read_csv(file.getvalue() if file is not None else str(SAMPLE))
    model, imputer, features = load_binary() if binary else load_multiclass()

    missing = [c for c in features if c not in df.columns]
    if missing:
        st.error(f"The file is missing {len(missing)} of {len(features)} required features, "
                 f"e.g. {missing[:5]}")
        with st.expander("Required feature list"):
            st.code("\n".join(features))
        return

    X = prepare(df, features, imputer)
    proba = model.predict_proba(X)
    res = pd.DataFrame(index=df.index)
    if "Destination Port" in df.columns:
        res["Destination Port"] = df["Destination Port"]

    if binary:
        p_attack = proba[:, list(model.classes_).index(1)]
        is_attack = p_attack >= threshold
        res["Prediction"] = np.where(is_attack, "ATTACK", "BENIGN")
        res["Attack_Probability"] = p_attack.round(4)
        res["Alert"] = np.where(is_attack, "ALERT", "SAFE")
    else:
        idx = proba.argmax(1)
        conf = proba.max(1)
        pred = model.classes_[idx]
        is_attack = pred != "BENIGN"
        res["Prediction"] = pred
        res["Confidence"] = conf.round(4)
        res["Attack_Probability"] = (1 - proba[:, list(model.classes_).index("BENIGN")]).round(4)
        res["Alert"] = np.select([~is_attack, conf >= threshold], ["SAFE", "ALERT"], "REVIEW")

    has_truth = "Label" in df.columns
    if has_truth:
        res["Actual"] = df["Label"]

    n = len(res)
    k = int(is_attack.sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total flows", f"{n:,}")
    c2.metric("Normal", f"{n - k:,}")
    c3.metric("Attacks", f"{k:,}")
    c4.metric("Attack rate", f"{k / max(n, 1) * 100:.2f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Traffic classification")
        st.bar_chart(res["Prediction"].value_counts(), horizontal=not binary)
    with right:
        st.subheader("Attack probability distribution")
        edges = np.linspace(0, 1, 11)
        labels = [f"{a:.1f}–{b:.1f}" for a, b in zip(edges[:-1], edges[1:])]
        hist = pd.cut(res["Attack_Probability"], bins=edges, labels=labels, include_lowest=True)
        st.bar_chart(hist.value_counts().reindex(labels, fill_value=0))

    if "Destination Port" in res.columns and k:
        st.subheader("Most targeted destination ports (alerts)")
        st.bar_chart(res.loc[is_attack, "Destination Port"].astype(str).value_counts().head(10))

    # Optional ground-truth evaluation
    if has_truth:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        st.subheader("Evaluation against the file's `Label` column")
        if binary:
            yt = (res["Actual"] != "BENIGN").astype(int)
            yp = is_attack.astype(int)
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Accuracy", f"{accuracy_score(yt, yp):.4f}")
            e2.metric("Precision", f"{precision_score(yt, yp, zero_division=0):.4f}")
            e3.metric("Recall", f"{recall_score(yt, yp, zero_division=0):.4f}")
            e4.metric("F1", f"{f1_score(yt, yp, zero_division=0):.4f}")
            cm = pd.crosstab(yt.map({0: "Benign", 1: "Attack"}).rename("Actual"),
                             pd.Series(np.where(yp, "Attack", "Benign"), index=res.index, name="Predicted"))
        else:
            e1, e2 = st.columns(2)
            e1.metric("Accuracy", f"{accuracy_score(res['Actual'], res['Prediction']):.4f}")
            e2.metric("Macro F1", f"{f1_score(res['Actual'], res['Prediction'], average='macro', zero_division=0):.4f}")
            cm = pd.crosstab(res["Actual"], res["Prediction"])
        st.dataframe(cm, width="stretch")

    st.subheader("Detection results")
    only_alerts = st.checkbox("Show only alerts", value=False)
    view = res[res["Alert"] != "SAFE"] if only_alerts else res
    st.dataframe(view.head(int(max_rows)), width="stretch")

    if k:
        st.warning(f"⚠️ {k:,} flow(s) were classified as potentially malicious.")
    else:
        st.success("✅ No attack was detected in the uploaded flows.")

    out = pd.concat([df, res.drop(columns=[c for c in ("Destination Port", "Actual") if c in res])], axis=1)
    st.download_button("Download predictions CSV", out.to_csv(index=False).encode(),
                       "ids_predictions.csv", "text/csv")


with tab_detect:
    render_detection()

# --------------------------------------------------------------------------- #
# Model performance
# --------------------------------------------------------------------------- #
with tab_perf:
    st.subheader("Binary model comparison (randomised 85/15 split)")
    comp = pd.read_csv(ROOT / "model_comparison.csv").set_index("Model")
    st.table(comp.style.format("{:.4f}").highlight_max(axis=0, props="font-weight:bold"))

    cd = pd.read_csv(ROOT / "cross_day_binary_metrics.csv").iloc[0]
    st.subheader("Cross-day generalisation (train Mon–Thu, test Friday)")
    cols = st.columns(5)
    for col, m in zip(cols, ["accuracy", "precision", "recall", "f1", "roc_auc"]):
        rnd = comp.loc["Random Forest", m.upper() if m == "roc_auc" else m.capitalize()]
        col.metric(m.replace("_", "-").upper() if m == "roc_auc" else m.capitalize(),
                   f"{cd[m]:.3f}", f"{cd[m] - rnd:+.3f}")
    st.caption("Deltas are relative to the Random Forest on the randomised split. "
               "Randomised splits overestimate real-world performance: unseen attack families "
               "(DDoS, PortScan, Bot on Friday) are much harder to detect.")

    figs = [("model_comparison.png", "Model comparison"),
            ("cross_day_comparison.png", "Randomised vs cross-day"),
            ("confusion_matrix_binary.png", "Binary confusion matrix"),
            ("feature_importance.png", "Feature importance"),
            ("confusion_matrix_multiclass.png", "Multiclass confusion matrix"),
            ("per_class_f1.png", "Per-class F1"),
            ("class_distribution.png", "Class distribution")]
    figs = [(f, t) for f, t in figs if (FIG / f).exists()]
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, (f, t) in zip(cols, figs[i:i + 2]):
            col.markdown(f"**{t}**")
            col.image(str(FIG / f), width="stretch")

# --------------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown("""
**Dataset.** CIC-IDS2017 (Canadian Institute for Cybersecurity) – ~2.8 M labelled flows over five
working days with 14 attack types (DoS/DDoS, PortScan, brute force, web attacks, botnet, infiltration,
Heartbleed).

**Preprocessing.** Column names stripped, label encoding repaired, ∞ → missing, median imputation
learned on training data only, 10 constant and 11 highly correlated (|r| > 0.995) features removed → 57 features.

**Models.** Random Forest (binary: 200 trees, multiclass: 250 trees, balanced class weights).
Logistic Regression, XGBoost and Isolation Forest were evaluated for comparison.

**Limitations.** File-based (not live packet capture); trained on one lab network; rare classes
(Infiltration, SQL injection, XSS, Heartbleed) have very few examples; performance drops sharply on
attack families not seen during training.
""")
    with st.expander(f"Required input features ({len(load_binary()[2])})"):
        st.code("\n".join(load_binary()[2]))
