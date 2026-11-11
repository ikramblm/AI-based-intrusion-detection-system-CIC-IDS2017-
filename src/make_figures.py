"""Create report figures from the result CSVs.

    python src/make_figures.py            # uses the project root
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BLUE, RED, GREY = "#2563eb", "#dc2626", "#6b7280"
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 9})


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", path.name)


def fig_model_comparison(root, out):
    df = pd.read_csv(root / "model_comparison.csv").set_index("Model")
    fig, ax = plt.subplots(figsize=(8, 3.6))
    df[["Precision", "Recall", "F1", "ROC_AUC"]].plot.bar(
        ax=ax, width=0.8, color=["#93c5fd", "#3b82f6", "#1d4ed8", "#1e3a8a"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_title("Binary detection: model comparison (randomised split)")
    ax.legend(ncol=4, loc="lower right", frameon=False)
    plt.setp(ax.get_xticklabels(), rotation=0)
    _save(fig, out / "model_comparison.png")


def fig_cm_binary(root, out):
    cm = pd.read_csv(root / "confusion_matrix_binary.csv", index_col=0)
    cm.index, cm.columns = ["Benign", "Attack"], ["Benign", "Attack"]
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Random Forest – binary")
    _save(fig, out / "confusion_matrix_binary.png")


def fig_cm_multiclass(root, out):
    cm = pd.read_csv(root / "confusion_matrix_multiclass.csv", index_col=0)
    norm = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
    short = [c.replace("Web Attack - ", "Web ") for c in cm.columns]
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(norm, annot=cm.values, fmt="d", cmap="Blues", vmin=0, vmax=1, ax=ax,
                xticklabels=short, yticklabels=short, cbar_kws={"label": "Row-normalised recall"},
                annot_kws={"size": 7})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Random Forest – multiclass (counts, colour = recall)")
    _save(fig, out / "confusion_matrix_multiclass.png")


def fig_importance(root, out, top=20):
    imp = pd.read_csv(root / "feature_importance_binary.csv", index_col=0)["importance"].head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp.index, imp.values, color=BLUE)
    ax.set_xlabel("Mean decrease in impurity")
    ax.set_title(f"Top {top} features – binary Random Forest")
    _save(fig, out / "feature_importance.png")


def fig_crossday(root, out):
    rnd = pd.read_csv(root / "model_comparison.csv").set_index("Model").loc["Random Forest"]
    cd = pd.read_csv(root / "cross_day_binary_metrics.csv").iloc[0]
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    vals = pd.DataFrame({"Randomised split": [rnd[m] for m in metrics],
                         "Cross-day (Friday held out)": [cd[m.lower()] for m in metrics]},
                        index=[m.replace("_", "-") for m in metrics])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    vals.plot.bar(ax=ax, color=[BLUE, RED], width=0.75)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", fontsize=7, padding=1)
    ax.set_ylim(0, 1.12)
    ax.set_title("Random Forest: randomised benchmark vs cross-day generalisation")
    ax.legend(frameon=False, loc="upper right", ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=0)
    _save(fig, out / "cross_day_comparison.png")


def fig_per_class_f1(root, out):
    cm = pd.read_csv(root / "confusion_matrix_multiclass.csv", index_col=0).values
    labels = pd.read_csv(root / "confusion_matrix_multiclass.csv", index_col=0).index
    tp = np.diag(cm)
    prec = tp / np.maximum(cm.sum(axis=0), 1)
    rec = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = np.where(prec + rec > 0, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0)
    s = pd.Series(f1, index=labels).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(s.index, s.values, color=[RED if v < 0.8 else BLUE for v in s.values])
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-score")
    ax.set_title("Per-class F1 – multiclass Random Forest")
    _save(fig, out / "per_class_f1.png")


def fig_class_distribution(root, out):
    p = root / "results" / "class_distribution.csv"
    if not p.exists():
        return
    d = pd.read_csv(p, index_col=0).sort_values("deduplicated_rows")
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(d.index, d["deduplicated_rows"], color=GREY, label="Full dataset (deduplicated)")
    ax.barh(d.index, d["working_sample"], color=BLUE, label="Working sample")
    ax.set_xscale("log")
    ax.set_xlabel("Flows (log scale)")
    ax.set_title("Class distribution")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, out / "class_distribution.png")


def make_all(root: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_model_comparison, fig_cm_binary, fig_cm_multiclass, fig_importance,
              fig_crossday, fig_per_class_f1, fig_class_distribution):
        f(root, out)


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
    make_all(root, root / "report" / "figures")
