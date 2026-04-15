from __future__ import annotations

import argparse
import json
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot thesis-ready charts from paired bootstrap retrieval comparison outputs."
    )
    p.add_argument("--input-dir", required=True, help="Directory with paired_bootstrap_summary.json and paired_bootstrap_per_query_deltas.csv")
    p.add_argument("--label", required=True, help="Label used in chart titles (e.g., Grampian 2023-2024)")
    p.add_argument("--output-dir", required=True, help="Output directory for charts")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _metric_rows(summary: dict) -> list[tuple[str, float, float, float]]:
    metrics = summary["metrics"]
    rows: list[tuple[str, float, float, float]] = []
    rows.append(("Hit@1 (Hybrid - Dense)", metrics["hit_at_1"]["observed_delta"], metrics["hit_at_1"]["ci95_low"], metrics["hit_at_1"]["ci95_high"]))
    rows.append(("Hit@3 (Hybrid - Dense)", metrics["hit_at_3"]["observed_delta"], metrics["hit_at_3"]["ci95_low"], metrics["hit_at_3"]["ci95_high"]))
    mrr_key = next(k for k in metrics.keys() if k.startswith("mrr_at_"))
    rows.append((f"MRR@{mrr_key.split('_')[-1]} (Hybrid - Dense)", metrics[mrr_key]["observed_delta"], metrics[mrr_key]["ci95_low"], metrics[mrr_key]["ci95_high"]))
    return rows


def plot_ci(summary: dict, label: str, out_path: Path, dpi: int) -> None:
    rows = _metric_rows(summary)
    labels = [r[0] for r in rows]
    means = np.array([r[1] for r in rows], dtype=float)
    lows = np.array([r[2] for r in rows], dtype=float)
    highs = np.array([r[3] for r in rows], dtype=float)
    x = np.arange(len(labels))
    yerr = np.vstack([means - lows, highs - means])

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--", alpha=0.8)
    ax.errorbar(
        x,
        means,
        yerr=yerr,
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        elinewidth=1.6,
        capsize=4,
        markersize=6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Delta (Hybrid - Dense)")
    ax.set_title("Paired Bootstrap 95% CI for Retrieval Deltas\n" + f"({label})")
    ax.grid(axis="y", alpha=0.3, linewidth=0.7)

    for i, m in enumerate(means):
        ax.text(i, m + (0.01 if m >= 0 else -0.015), f"{m:.3f}", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_distributions(per_query: pd.DataFrame, label: str, out_path: Path, dpi: int) -> None:
    cols = [
        ("delta_hit_at_1_a_minus_b", "Hit@1 delta"),
        ("delta_hit_at_3_a_minus_b", "Hit@3 delta"),
        ("delta_mrr_a_minus_b", "MRR delta"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.8), sharey=False)

    for ax, (col, ttl) in zip(axes, cols):
        x = per_query[col].to_numpy(dtype=float)
        bins = 21 if "mrr" in col else np.arange(-1.05, 1.1, 0.1)
        ax.hist(x, bins=bins, color="#c7c7c7", edgecolor="#666666", linewidth=0.7)
        mean_x = float(np.mean(x)) if len(x) else 0.0
        ax.axvline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        ax.axvline(mean_x, color="#1f77b4", linestyle="-", linewidth=1.2)
        ax.set_title(ttl, fontsize=10)
        ax.set_xlabel("Per-query delta (A - B)")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.text(0.98, 0.93, f"mean={mean_x:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=8)

    axes[0].set_ylabel("Query count")
    fig.suptitle("Per-query Delta Distributions\n" + f"({label})", y=1.03, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = json.loads((input_dir / "paired_bootstrap_summary.json").read_text(encoding="utf-8"))
    per_query = pd.read_csv(input_dir / "paired_bootstrap_per_query_deltas.csv")

    ci_path = out_dir / "paired_bootstrap_ci.png"
    dist_path = out_dir / "paired_bootstrap_delta_distributions.png"
    plot_ci(summary, label=args.label, out_path=ci_path, dpi=int(args.dpi))
    plot_distributions(per_query, label=args.label, out_path=dist_path, dpi=int(args.dpi))

    print("Saved:", ci_path)
    print("Saved:", dist_path)


if __name__ == "__main__":
    main()
