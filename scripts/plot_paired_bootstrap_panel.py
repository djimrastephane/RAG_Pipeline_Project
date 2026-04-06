from __future__ import annotations

import argparse
import json
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import MultipleLocator


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

    NAVY = "#1a2744"
    WHITE = "#ffffff"
    GREY_GRID = "#d8dde8"
    GREY_MID = "#8c96ab"
    SIGNIF_POS = "#e07b39"
    SIGNIF_NEG = "#c0392b"
    INSIG = "#6b7a99"
    BAND_COL = "#c8d0e0"

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Georgia", "DejaVu Serif"],
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "axes.edgecolor": GREY_MID,
            "axes.linewidth": 0.8,
            "xtick.color": NAVY,
            "ytick.color": NAVY,
            "text.color": NAVY,
            "grid.color": GREY_GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 5.2), sharey=False)
    fig.subplots_adjust(wspace=0.35, left=0.08, right=0.97, top=0.84, bottom=0.18)
    cohort_band_idx = labels.index("2022–2023") if "2022–2023" in labels else None
    panels = [
        ("Hit@1", np.array(hit1_mean), np.array(hit1_lo), np.array(hit1_hi)),
        ("Hit@3", np.array(hit3_mean), np.array(hit3_lo), np.array(hit3_hi)),
        ("MRR@10", np.array(mrr_mean), np.array(mrr_lo), np.array(mrr_hi)),
    ]

    for panel_idx, (ax, (title, mean, lo, hi)) in enumerate(zip(axes, panels)):
        sig_mask = (lo > 0.0) | (hi < 0.0)

        if cohort_band_idx is not None:
            ax.axvspan(
                cohort_band_idx - 0.42,
                cohort_band_idx + 0.42,
                color=BAND_COL,
                alpha=0.35,
                zorder=0,
                lw=0,
            )
        ax.axhline(0.0, color=GREY_MID, linewidth=0.9, linestyle="--", zorder=1)

        for xi, yi, lo_i, hi_i, is_sig in zip(x, mean, lo, hi, sig_mask):
            is_pos = yi >= 0.0
            dot_color = SIGNIF_POS if (is_sig and is_pos) else SIGNIF_NEG if (is_sig and not is_pos) else INSIG
            yerr_i = np.array([[yi - lo_i], [hi_i - yi]], dtype=float)
            ax.errorbar(
                [xi],
                [yi],
                yerr=yerr_i,
                fmt="o",
                color=dot_color,
                ecolor=dot_color,
                elinewidth=1.6,
                capsize=4,
                capthick=1.6,
                markersize=7 if is_sig else 6,
                markerfacecolor=dot_color if is_sig else WHITE,
                markeredgewidth=1.8,
                zorder=2,
            )

            if is_sig:
                offset = 0.025 if yi >= 0 else -0.025
                va = "bottom" if yi >= 0 else "top"
                y_txt = hi_i + offset if yi >= 0 else lo_i + offset
                ax.text(
                    xi,
                    y_txt,
                    f"{yi:+.3f}",
                    ha="center",
                    va=va,
                    fontsize=7.5,
                    color=dot_color,
                    fontweight="bold",
                )

        pad = 0.04
        ymin = float(np.min(lo)) - pad
        ymax = float(np.max(hi)) + pad
        bound = max(abs(ymin), abs(ymax))
        ax.set_ylim(-bound - pad, bound + pad)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.5, rotation=30, ha="right")
        ax.yaxis.set_minor_locator(MultipleLocator(0.025))
        ax.grid(True, axis="y", which="major")
        ax.grid(False, axis="x")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if panel_idx > 0:
            ax.tick_params(axis="y", labelleft=False)

        if cohort_band_idx is not None:
            ax.text(
                cohort_band_idx,
                -bound - pad + 0.03,
                "weakest\ncohort",
                ha="center",
                va="bottom",
                fontsize=7.4,
                color=GREY_MID,
                style="italic",
                linespacing=1.1,
            )
        ax.set_title(f"{title} delta", fontsize=10.5, fontweight="bold", color=NAVY, pad=6)

    axes[0].set_ylabel("Δ metric  (Hybrid − Dense)\nwith 95 % CI", fontsize=9, labelpad=8)
    legend_els = [
        mpatches.Patch(color=SIGNIF_POS, label="Sig. positive  (CI > 0)"),
        mpatches.Patch(color=SIGNIF_NEG, label="Sig. negative  (CI < 0)"),
        mpatches.Patch(color=INSIG, label="Not significant"),
        mpatches.Patch(color=BAND_COL, label="Weakest cohort (2022–23)", alpha=0.7),
    ]
    fig.legend(
        handles=legend_els,
        loc="upper center",
        ncol=4,
        fontsize=8.2,
        frameon=False,
        bbox_to_anchor=(0.52, 1.00),
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.2,
    )
    fig.suptitle("Paired Bootstrap Retrieval Comparison — Grampian Cohorts", fontsize=12.5, fontweight="bold", color=NAVY, y=1.06)
    n_bootstrap_label = ",".join(str(v) for v in sorted(v for v in n_bootstrap_seen if v > 0)) or "unknown"
    uniq_n = sorted(set(n_per_cohort))
    n_label = f"n={uniq_n[0]} per cohort" if len(uniq_n) == 1 else "n varies by cohort"
    fig.text(
        0.52,
        1.01,
        f"{n_bootstrap_label} resamples · α = 0.05 · {n_label} · orange/red markers indicate cohorts where the 95% CI excludes zero",
        ha="center",
        fontsize=8,
        color=GREY_MID,
    )
    fig.savefig(output_path, dpi=int(args.dpi), bbox_inches="tight")
    plt.close(fig)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
