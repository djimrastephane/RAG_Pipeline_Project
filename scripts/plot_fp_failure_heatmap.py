#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot FP1-FP7 failure heatmap with count + percentage labels.")
    p.add_argument("--counts-csv", default="results/fp1_fp7_counts_full50.csv")
    p.add_argument("--output", default="results/fp_failure_heatmap_full50_v2.png")
    p.add_argument("--queries-per-series", type=int, default=50)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def _pretty_series_label(v: str) -> str:
    s = str(v).replace("Grampian ", "")
    # 2022-2023 -> 2022–23 (en-dash + shortened second year)
    if len(s) >= 9 and s[:4].isdigit() and s[4] == "-" and s[5:9].isdigit():
        y1 = s[:4]
        y2 = s[7:9]
        s = f"{y1}\u2013{y2}" + s[9:]
    return s.replace(" Dense", "\nDense").replace(" Hybrid", "\nHybrid")


def plot_heatmap(
    counts_csv: Path,
    output: Path,
    queries_per_series: int = 50,
    dpi: int = 300,
) -> None:
    counts_df = pd.read_csv(counts_csv)
    merged = counts_df.copy()
    denom = max(1, int(queries_per_series))
    merged["pct"] = (merged["count"].astype(float) / float(denom)) * 100.0

    fp_order = ["FP1", "FP2", "FP3", "FP4", "FP5", "FP6", "FP7"]
    series_order = [
        "Grampian 2022-2023 Dense",
        "Grampian 2022-2023 Hybrid",
        "Grampian 2023-2024 Dense",
        "Grampian 2023-2024 Hybrid",
    ]
    rows_present = [x for x in series_order if x in set(merged["series"].unique())]
    if not rows_present:
        rows_present = sorted(merged["series"].unique().tolist())

    heat_counts = (
        merged.pivot(index="series", columns="fp_code", values="count")
        .reindex(index=rows_present, columns=fp_order)
        .fillna(0.0)
    )
    heat_pct = (
        merged.pivot(index="series", columns="fp_code", values="pct")
        .reindex(index=rows_present, columns=fp_order)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    fig.patch.set_facecolor("white")
    im = ax.imshow(heat_counts.values, cmap="Reds", aspect="auto", vmin=0, vmax=max(1.0, float(heat_counts.values.max())))

    ax.set_xticks(np.arange(len(fp_order)))
    ax.set_xticklabels(
        [
            "FP1\nNo chunk\nreturned",
            "FP2\nWrong chunk\nretrieved",
            "FP3\nPartial\nretrieval",
            "FP4\nFormat\nmismatch",
            "FP5\nExtraction\nfailure",
            "FP6\nAnswer\nassembly",
            "FP7\nHallucination",
        ],
        fontsize=8,
        fontweight="semibold",
    )
    ax.set_yticks(np.arange(len(rows_present)))
    ax.set_yticklabels([_pretty_series_label(x) for x in rows_present], fontsize=9)

    ax.set_title(
        f"FP1-FP7 Failure Heatmap (count and % of {denom} queries)",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )

    # Light grid to separate cells.
    ax.set_xticks(np.arange(-0.5, len(fp_order), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows_present), 1), minor=True)
    ax.grid(which="minor", color="#f0f0f0", linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    # Visual separator between year cohorts (Dense/Hybrid pairs).
    if len(rows_present) >= 4:
        ax.hlines(
            y=1.5,
            xmin=-0.5,
            xmax=len(fp_order) - 0.5,
            colors="#7a7a7a",
            linestyles="--",
            linewidth=1.0,
            alpha=0.8,
        )

    for i in range(heat_counts.shape[0]):
        for j in range(heat_counts.shape[1]):
            c = float(heat_counts.iat[i, j])
            p = float(heat_pct.iat[i, j])
            if c <= 0:
                ax.text(j, i, "–", ha="center", va="center", fontsize=10, color="#888888")
                continue
            txt_color = "white" if c >= max(6.0, 0.35 * float(heat_counts.values.max())) else "#222222"
            ax.text(
                j,
                i,
                f"{int(c)}\n{p:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color=txt_color,
                fontweight="semibold" if c >= 8 else "normal",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Failure count", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")


def main() -> None:
    args = parse_args()
    plot_heatmap(
        counts_csv=Path(args.counts_csv),
        output=Path(args.output),
        queries_per_series=int(args.queries_per_series),
        dpi=int(args.dpi),
    )


if __name__ == "__main__":
    main()
