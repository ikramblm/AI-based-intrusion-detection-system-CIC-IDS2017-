"""Build a small labelled demo CSV (raw CIC-IDS2017 format) for the dashboard.

    python src/make_sample.py --data "C:/path/to/MachineLearningCVE" [--per-class 150] [--benign 1500]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline import CHUNK, LABEL, SEED, normalize_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--per-class", type=int, default=150)
    ap.add_argument("--benign", type=int, default=1500)
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "data" / "sample_flows.csv")
    a = ap.parse_args()

    rng = np.random.default_rng(SEED)
    kept: dict[str, pd.DataFrame] = {}
    for f in sorted(a.data.glob("*.csv")):
        for chunk in pd.read_csv(f, chunksize=CHUNK, encoding="latin-1", low_memory=False):
            chunk.columns = chunk.columns.str.strip()
            chunk[LABEL] = normalize_label(chunk[LABEL])
            chunk["_key"] = rng.random(len(chunk))
            for lab, g in chunk.groupby(LABEL, sort=False):
                cap = a.benign if lab == "BENIGN" else a.per_class
                cur = pd.concat([kept[lab], g]) if lab in kept else g
                kept[lab] = cur.nsmallest(cap, "_key")
        print("read", f.name)
    out = pd.concat(kept.values()).drop(columns="_key").sample(frac=1, random_state=SEED)
    out.columns = [c.split(".")[0] if c.startswith("Fwd Header Length.") else c for c in out.columns]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out} ({len(out):,} rows)\n{out[LABEL].value_counts()}")


if __name__ == "__main__":
    main()
