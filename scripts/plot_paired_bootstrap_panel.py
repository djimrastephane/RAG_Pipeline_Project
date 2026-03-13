from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create a combined panel chart of paired-bootstrap retrieval deltas across cohorts."
    )
    p.add_argument(
        "--cohorts",
        nargs="+",
        required=True,
        help="Cohort IDs, e.g. Grampian-2021-2022 Grampian-2022-2023",
    )
    p.add_argument(
        "--input-root",
        default="results/paired_bootstrap_retrieval_compare",
        help="Root containing <cohort>_hybrid_vs_dense/paired_bootstrap_summary.json",
    )
    p.add_argument("--output-path", required=True, help="Output figure path (.png)")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_path = Path(args.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels: list[str] = []
    hit1_mean: list[float] = []
    hit1_lo: list[float] = []
    hit1_hi: list[float] = []
    hit3_mean: list[float] = []
    hit3_lo: list[float] = []
    hit3_hi: list[float] = []
    mrr_mean: list[float] = []
    mrr_lo: list[float] = []
    mrr_hi: list[float] = []
    n_per_cohort: list[int] = []
    n_bootstrap_seen: set[int] = set()

    for cohort in args.cohorts:
        summary_path = input_root / f"{cohort}_hybrid_vs_dense" / "paired_bootstrap_summary.json"
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics = data["metrics"]
        inputs = data.get("inputs", {})
        mrr_key = next(k for k in metrics.keys() if k.startswith("mrr_at_"))

        labels.append(cohort.replace("Grampian-", "").replace("-", "–", 1))
        n_per_cohort.append(int(inputs.get("n_common_queries", 0)))
        n_bootstrap_seen.add(int(inputs.get("n_bootstrap", 0)))
        hit1_mean.append(float(metrics["hit_at_1"]["observed_delta"]))
        hit1_lo.append(float(metrics["hit_at_1"]["ci95_low"]))
        hit1_hi.append(float(metrics["hit_at_1"]["ci95_high"]))
        hit3_mean.append(float(metrics["hit_at_3"]["observed_delta"]))
        hit3_lo.append(float(metrics["hit_at_3"]["ci95_low"]))
        hit3_hi.append(float(metrics["hit_at_3"]["ci95_high"]))
        mrr_mean.append(float(metrics[mrr_key]["observed_delta"]))
        mrr_lo.append(float(metrics[mrr_key]["ci95_low"]))
        mrr_hi.append(float(metrics[mrr_key]["ci95_high"]))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.6), sharex=True, sharey=True)
    all_lows = np.array(hit1_lo + hit3_lo + mrr_lo, dtype=float)
    all_highs = np.array(hit1_hi + hit3_hi + mrr_hi, dtype=float)
    y_min = float(np.min(all_lows))
    y_max = float(np.max(all_highs))
    # Shared y-scale across panels for direct visual comparison.
    y_lo = -0.25
    y_hi = 1.05
    if y_min < y_lo:
        y_lo = np.floor((y_min - 0.05) * 10.0) / 10.0
    if y_max > y_hi:
        y_hi = np.ceil((y_max + 0.05) * 10.0) / 10.0

    # Unified blue palette; significance (CI excludes 0) is encoded by filled vs hollow marker.
    whisker_color = "#4f81bd"
    panel_colors = ["#1f4e79", "#2f6690", "#3d7ea6"]  # three coordinated blue tones
    sig_edge = "#1f4e79"
    nonsig_color = "#555555"
    panels = [
        ("Hit@1 delta (Hybrid - Dense)", np.array(hit1_mean), np.array(hit1_lo), np.array(hit1_hi)),
        ("Hit@3 delta (Hybrid - Dense)", np.array(hit3_mean), np.array(hit3_lo), np.array(hit3_hi)),
        ("MRR@10 delta (Hybrid - Dense)", np.array(mrr_mean), np.array(mrr_lo), np.array(mrr_hi)),
    ]

    for ax, (title, mean, lo, hi), panel_color in zip(axes, panels, panel_colors):
        yerr = np.vstack([mean - lo, hi - mean])
        sig_mask = (lo > 0.0) | (hi < 0.0)
        nonsig_mask = ~sig_mask

        ax.axhline(0.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.errorbar(x, mean, yerr=yerr, fmt="none", ecolor=whisker_color, elinewidth=0.9, capsize=2.0, zorder=2)
        if np.any(nonsig_mask):
            ax.scatter(
                x[nonsig_mask],
                mean[nonsig_mask],
                s=34,
                facecolors="white",
                edgecolors=nonsig_color,
                linewidth=1.1,
                zorder=3,
            )
        if np.any(sig_mask):
            ax.scatter(
                x[sig_mask],
                mean[sig_mask],
                s=34,
                color=panel_color,
                edgecolor=sig_edge,
                linewidth=0.7,
                zorder=4,
            )

        ax.set_title(title, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.3, linewidth=0.7)
        ax.set_ylim(y_lo, y_hi)

        for xi, yi, hi_i, lo_i in zip(x, mean, hi, lo):
            # Place value labels clear of CI whiskers.
            if yi >= 0:
                y_txt = hi_i + 0.015
                va = "bottom"
            else:
                y_txt = lo_i - 0.015
                va = "top"
            # Keep labels inside y-limits so they cannot be clipped at export.
            y_txt = min(max(y_txt, y_lo + 0.02), y_hi - 0.02)
            ax.text(xi, y_txt, f"{yi:.3f}", ha="center", va=va, fontsize=8)

    axes[0].set_ylabel("Observed delta with 95% CI")
    fig.suptitle("Paired Bootstrap Retrieval Comparison Across Grampian Cohorts", y=1.07, fontsize=12)
    n_bootstrap_label = ",".join(str(v) for v in sorted(v for v in n_bootstrap_seen if v > 0)) or "unknown"
    uniq_n = sorted(set(n_per_cohort))
    n_label = f"n={uniq_n[0]} per cohort" if len(uniq_n) == 1 else "n varies by cohort"
    fig.text(
        0.5,
        1.01,
        f"{n_bootstrap_label} resamples · alpha=0.05 · {n_label} · filled=CI excludes 0",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#333333",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
