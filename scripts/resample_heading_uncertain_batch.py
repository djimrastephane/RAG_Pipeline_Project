"""
Create next assisted-labeling batch from an existing label CSV.

Expected input columns from build_assisted_heading_labelset.py:
- doc_id, page, line_text, auto_label, auto_confidence, final_label

Optional:
- model_score (if a classifier has been trained; values in [0,1])
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resample uncertain heading-labeling batch.")
    p.add_argument("--in-csv", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--batch-size", type=int, default=300)
    p.add_argument("--max-per-doc", type=int, default=60)
    p.add_argument("--prefer-unlabeled", action="store_true", default=True)
    return p.parse_args()


def _to_float(v: object, default: float = 0.5) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return default


def _is_labeled(v: object) -> bool:
    s = str(v).strip()
    return s in {"0", "1"}


def main() -> None:
    args = parse_args()
    in_csv = Path(args.in_csv).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    if "final_label" not in df.columns:
        df["final_label"] = ""
    if "model_score" not in df.columns:
        df["model_score"] = np.nan
    if "auto_confidence" not in df.columns:
        df["auto_confidence"] = 0.5

    df["is_labeled"] = df["final_label"].map(_is_labeled)
    if args.prefer_unlabeled:
        cand = df[~df["is_labeled"]].copy()
    else:
        cand = df.copy()

    if cand.empty:
        print("No candidate rows available.")
        out_csv.write_text("", encoding="utf-8")
        return

    # Uncertainty from model if available, else from auto confidence.
    has_model = cand["model_score"].notna().any()
    if has_model:
        score = cand["model_score"].map(lambda x: _to_float(x, 0.5))
    else:
        # convert confidence to pseudo-prob for uncertainty ranking
        score = cand["auto_confidence"].map(lambda x: _to_float(x, 0.5))
        # map confidence of predicted class back to heading-prob proxy
        if "auto_label" in cand.columns:
            auto = cand["auto_label"].fillna(0).astype(int)
            score = np.where(auto == 1, score, 1.0 - score)
            score = pd.Series(score, index=cand.index)

    cand["uncertainty"] = 1.0 - (2.0 * (score - 0.5).abs())
    cand["uncertainty"] = cand["uncertainty"].clip(lower=0.0, upper=1.0)

    # Keep some diversity: cap per doc.
    pieces = []
    per_doc = int(max(1, args.max_per_doc))
    for _, g in cand.groupby("doc_id", sort=True):
        gg = g.sort_values(["uncertainty", "page"], ascending=[False, True]).head(per_doc)
        pieces.append(gg)
    pool = pd.concat(pieces, ignore_index=True) if pieces else cand

    out = (
        pool.sort_values(["uncertainty", "doc_id", "page"], ascending=[False, True, True])
        .head(int(args.batch_size))
        .copy()
    )
    out.to_csv(out_csv, index=False)

    print("Saved:", out_csv)
    print("Rows:", len(out))
    print("Docs:", out["doc_id"].nunique() if len(out) else 0)
    print("Mean uncertainty:", float(out["uncertainty"].mean()) if len(out) else 0.0)


if __name__ == "__main__":
    main()
