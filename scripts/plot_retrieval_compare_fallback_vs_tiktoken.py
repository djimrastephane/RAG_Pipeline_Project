from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "results" / "retrieval_compare_fallback_vs_tiktoken_2020_2025.csv"
DEFAULT_PNG = REPO_ROOT / "results" / "retrieval_compare_fallback_vs_tiktoken_2020_2025.png"


def _short_label(doc_id: str) -> str:
    parts = str(doc_id).split("-")
    if len(parts) >= 3:
        return f"{parts[-2]}-{parts[-1]}"
    return str(doc_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot fallback vs tiktoken retrieval deltas.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out", default=str(DEFAULT_PNG))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    labels = [_short_label(v) for v in df["doc_id"].tolist()]
    x = list(range(len(df)))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)

    left = axes[0]
    left.bar(
        [i - width / 2 for i in x],
        df["fallback_k1_page_hit_rate"],
        width=width,
        color="#93a8c3",
        label="fallback",
    )
    left.bar(
        [i + width / 2 for i in x],
        df["tiktoken_k1_page_hit_rate"],
        width=width,
        color="#2bb3c8",
        label="tiktoken",
    )
    left.set_title("Page Hit@1")
    left.set_ylim(0, 1.0)
    left.set_xticks(x, labels, rotation=20)
    left.grid(axis="y", alpha=0.22)
    left.legend(frameon=False)
    for i, delta in enumerate(df["delta_k1_page_hit_rate"].tolist()):
        top = max(df.loc[i, "fallback_k1_page_hit_rate"], df.loc[i, "tiktoken_k1_page_hit_rate"])
        left.text(i, float(top) + 0.025, f"{delta:+.02f}", ha="center", va="bottom", fontsize=9)

    right = axes[1]
    right.bar(
        [i - width / 2 for i in x],
        df["fallback_k10_page_mrr"],
        width=width,
        color="#b0bccf",
        label="fallback",
    )
    right.bar(
        [i + width / 2 for i in x],
        df["tiktoken_k10_page_mrr"],
        width=width,
        color="#6ce3a5",
        label="tiktoken",
    )
    right.set_title("Page MRR@10")
    right.set_ylim(0, 1.0)
    right.set_xticks(x, labels, rotation=20)
    right.grid(axis="y", alpha=0.22)
    for i, delta in enumerate(df["delta_k10_page_mrr"].tolist()):
        top = max(df.loc[i, "fallback_k10_page_mrr"], df.loc[i, "tiktoken_k10_page_mrr"])
        right.text(i, float(top) + 0.025, f"{delta:+.02f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Retrieval Impact of Reprocessing with tiktoken (Grampian 2020-2025)", fontsize=14, fontweight="bold")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
