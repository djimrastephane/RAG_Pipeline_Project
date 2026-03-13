from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "results" / "tiktoken_vs_fallback_2020_2025.csv"
DEFAULT_PNG = REPO_ROOT / "results" / "tiktoken_vs_fallback_2020_2025.png"


def _short_label(doc_id: str) -> str:
    parts = str(doc_id).split("-")
    if len(parts) >= 3:
        return f"{parts[-2]}-{parts[-1]}"
    return str(doc_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tiktoken vs fallback comparison.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--out", default=str(DEFAULT_PNG))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError("Comparison CSV is empty.")

    labels = [_short_label(v) for v in df["doc_id"].tolist()]
    x = range(len(df))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), constrained_layout=True)

    token_ax = axes[0]
    width = 0.38
    token_ax.bar(
        [i - width / 2 for i in x],
        df["page_tokens_tiktoken"],
        width=width,
        label="tiktoken",
        color="#2bb3c8",
    )
    token_ax.bar(
        [i + width / 2 for i in x],
        df["page_tokens_fallback"],
        width=width,
        label="fallback",
        color="#6ce3a5",
    )
    token_ax.set_title("Total Page Token Counts")
    token_ax.set_ylabel("Tokens")
    token_ax.set_xticks(list(x), labels, rotation=20)
    token_ax.grid(axis="y", alpha=0.22)
    token_ax.legend(frameon=False)

    chunk_ax = axes[1]
    chunk_ax.bar(
        [i - width / 2 for i in x],
        df["text_chunks_tiktoken"],
        width=width,
        label="tiktoken",
        color="#2799cc",
    )
    chunk_ax.bar(
        [i + width / 2 for i in x],
        df["text_chunks_fallback"],
        width=width,
        label="fallback",
        color="#93d977",
    )
    chunk_ax.set_title("Total Text Chunk Counts")
    chunk_ax.set_ylabel("Chunks")
    chunk_ax.set_xticks(list(x), labels, rotation=20)
    chunk_ax.grid(axis="y", alpha=0.22)

    for idx, row in df.reset_index(drop=True).iterrows():
        token_ax.text(
            idx,
            max(float(row["page_tokens_tiktoken"]), float(row["page_tokens_fallback"])) * 1.02,
            f"{float(row['page_tokens_delta_pct']):.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#425466",
        )
        chunk_ax.text(
            idx,
            max(float(row["text_chunks_tiktoken"]), float(row["text_chunks_fallback"])) * 1.03,
            f"{float(row['text_chunks_delta_pct']):.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#425466",
        )

    fig.suptitle("tiktoken vs Fallback: Grampian 2020-2025", fontsize=14, fontweight="bold")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
