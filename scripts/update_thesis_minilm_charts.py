#!/usr/bin/env python3
"""Regenerate MiniLM thesis figures with consistent styling."""

from __future__ import annotations

from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import pandas as pd


CAP = 256
EFFECTIVE_X_MAX = 750
EFFECTIVE_Y_MAX = 300

# Keep one palette across Grampian/Shetland figures.
LINE_COLOR = "#1f77b4"
BAR_COLOR = "#1f77b4"
BAR_SECONDARY = "#c7d6ea"


def effective_tokens(chunk_size: int, cap: int = CAP) -> int:
    return min(chunk_size, cap)


def plot_effective_context(output_path: Path, label: str) -> None:
    tested_chunk_sizes = [224, 256, 280]
    tested_effective = [effective_tokens(x) for x in tested_chunk_sizes]

    x_values = list(range(0, EFFECTIVE_X_MAX + 1))
    y_values = [effective_tokens(x) for x in x_values]
    y_theoretical = x_values

    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    fig.patch.set_facecolor("#dcdcdc")
    # Background guidance zones.
    ax.axvspan(0, 224, color="#2e8b57", alpha=0.18, zorder=0)
    ax.axvspan(224, 256, color="#f1c15f", alpha=0.24, zorder=0)
    ax.axvspan(256, EFFECTIVE_X_MAX, color="#c0392b", alpha=0.18, zorder=0)

    # Reference (no truncation) diagonal and effective curve.
    ax.plot(x_values, y_theoretical, linewidth=1.3, linestyle="--", color="gray", alpha=0.9, label="Theoretical (no cap)")
    ax.plot(x_values, y_values, linewidth=2.2, color=LINE_COLOR, label="Effective: min(chunk_size, 256)")

    # Waste zone: explicit truncation loss after cap.
    x_tail = [x for x in x_values if x >= CAP]
    y_tail_eff = [effective_tokens(x) for x in x_tail]
    ax.fill_between(x_tail, y_tail_eff, x_tail, color="#c0392b", alpha=0.10)

    # Cap and chosen operating point.
    ax.axvline(CAP, linestyle="--", linewidth=1.6, color="#2c3e50", label="MiniLM cap = 256 tokens")
    ax.axvline(224, linestyle="--", linewidth=1.6, color="#27ae60", label="Chosen chunk size (224) - fully embedded")

    # Single marker at knee.
    ax.scatter([256], [256], s=56, color=LINE_COLOR, zorder=3)

    # Single callout at the knee point.
    ax.annotate(
        "Chunks > 256 tokens are silently truncated.\nA 280-token chunk loses ~24 tokens per embedding.",
        xy=(280, 256),
        xytext=(370, 210),
        textcoords="data",
        fontsize=11.0,
        bbox={"facecolor": "white", "edgecolor": "0.6", "boxstyle": "round,pad=0.35"},
        arrowprops={"arrowstyle": "->", "linewidth": 1.0, "color": "black"},
    )

    # Zone labels.
    ax.text(112, 155, "Operating range\n(0 - 224 tokens)", fontsize=11.0, color="#1b7f4c", fontweight="bold", ha="center", va="center")
    ax.text(240, 82, "Buffer\n(+32)", fontsize=11.0, color="#b9770e", fontweight="bold", ha="center", va="center")
    ax.text(500, 82, "Truncation zone  (256+)", fontsize=11.0, color="#c0392b", fontweight="bold", ha="center", va="center")

    ax.set_title(f"224-Token Chunks Stay Within MiniLM Cap - Full Semantic Context Preserved ({label})", fontsize=14.0, fontweight="bold", pad=12)
    ax.set_xlabel("Chunk size (tokens)", fontsize=11)
    ax.set_ylabel("Effective tokens seen by MiniLM", fontsize=11)
    ax.set_xlim(0, EFFECTIVE_X_MAX)
    ax.set_ylim(0, 320)
    ax.tick_params(labelsize=10)
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.9, borderpad=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved: {output_path}")
    print(f"  tested chunk sizes: {tested_chunk_sizes}")
    print(f"  effective tokens:   {tested_effective}")


def plot_truncation_overall(summary_csv: Path, output_path: Path, title: str) -> None:
    s = pd.read_csv(summary_csv).iloc[0]
    total = int(s["total_chunks"])
    exceeded = int(s["chunks_exceeding_256"])
    within = total - exceeded
    trunc_rate = float(s["truncation_rate"])

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    labels = ["<= 256 tokens", "> 256 tokens"]
    counts = [within, exceeded]
    bars = ax.bar(labels, counts, color=[BAR_SECONDARY, BAR_COLOR], width=0.62, edgecolor="black", linewidth=0.6)

    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + max(total * 0.01, 3), f"{c:,}", ha="center", va="bottom", fontsize=10)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Number of chunks", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.2)
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)
    ax.text(
        0.02,
        0.96,
        f"Total chunks: {total:,}\nTruncation rate: {trunc_rate:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "0.75", "boxstyle": "round,pad=0.25"},
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_truncation_by_doc(per_doc_csv: Path, output_path: Path, title: str) -> None:
    df = pd.read_csv(per_doc_csv).copy()
    df["year"] = df["doc_id"].str.extract(r"(\d{4}-\d{4})")
    df = df.sort_values("year")

    x = range(len(df))
    y = df["truncation_rate"] * 100.0

    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ax.bar(x, y, color=BAR_COLOR, width=0.72)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Truncation rate (%)", fontsize=11)
    ax.set_xlabel("Document", fontsize=11)
    ax.set_xticks(list(x))
    ax.set_xticklabels(df["year"], rotation=55, ha="right", fontsize=9)
    ax.set_ylim(0, max(55, y.max() * 1.2))
    ax.grid(axis="y", linewidth=0.5, alpha=0.3)

    mean_val = y.mean()
    ax.axhline(mean_val, linestyle="--", linewidth=1.2, color="black")
    ax.text(len(df) - 0.6, mean_val + 0.6, f"Mean: {mean_val:.1f}%", ha="right", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    token_dir = root / "results" / "token_validation"
    docs_figures_dir = root / "docs" / "figures"
    docs_figures_dir.mkdir(parents=True, exist_ok=True)

    plot_effective_context(docs_figures_dir / "effective_embedding_context_minilm_grampian.png", "Grampian")
    plot_effective_context(docs_figures_dir / "effective_embedding_context_minilm_shetland.png", "Shetland")

    plot_truncation_overall(
        token_dir / "grampian_21_minilm_summary.csv",
        token_dir / "grampian_21_minilm_truncation.png",
        "MiniLM Truncation Summary (Grampian, 21 Documents)",
    )
    plot_truncation_by_doc(
        token_dir / "grampian_21_minilm_per_doc.csv",
        token_dir / "grampian_21_minilm_truncation_by_doc.png",
        "MiniLM Truncation Rate by Document (Grampian)",
    )
    plot_truncation_overall(
        token_dir / "shetland_10_minilm_summary.csv",
        token_dir / "shetland_10_minilm_truncation.png",
        "MiniLM Truncation Summary (Shetland, 10 Documents)",
    )
    plot_truncation_by_doc(
        token_dir / "shetland_10_minilm_per_doc.csv",
        token_dir / "shetland_10_minilm_truncation_by_doc.png",
        "MiniLM Truncation Rate by Document (Shetland)",
    )


if __name__ == "__main__":
    main()
