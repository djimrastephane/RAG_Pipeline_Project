from __future__ import annotations

import argparse
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]
LABELS = {
    "280_90_baseline": "280/90 LLM off",
    "280_90_generated": "280/90 LLM on",
    "224_56_baseline": "224/56 LLM off",
    "224_56_generated": "224/56 LLM on",
}
COLORS = {
    "280_90": "#1f4e79",
    "224_56": "#b35a00",
}
PANELS = [
    ("baseline", "LLM off (retrieval only)", ["280_90_baseline", "224_56_baseline"]),
    ("generated", "LLM on (RAG answers)", ["280_90_generated", "224_56_generated"]),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a thesis-ready chart comparing RAGAS runs across chunk settings.")
    p.add_argument(
        "--summary-csv",
        default="results/ragas/chunk_compare_224_vs_280/summary_all_runs.csv",
        help="Input CSV with one row per run.",
    )
    p.add_argument(
        "--out-png",
        default="results/ragas/chunk_compare_224_vs_280/ragas_chunk_comparison_thesis.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--out-csv",
        default="results/ragas/chunk_compare_224_vs_280/ragas_chunk_comparison_plot_data.csv",
        help="Output flattened plot-data CSV path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_csv)
    out_png = Path(args.out_png)
    out_csv = Path(args.out_csv)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(summary_path)
    order = [k for k in LABELS if k in set(df["run"].astype(str))]
    if not order:
        raise ValueError(f"No recognized run labels found in {summary_path}")

    long_rows: list[dict[str, object]] = []
    for run_key in order:
        row = df.loc[df["run"] == run_key].iloc[0]
        for metric in METRICS:
            long_rows.append(
                {
                    "run": run_key,
                    "run_label": LABELS[run_key],
                    "metric": metric,
                    "metric_label": metric.replace("_", " ").title(),
                    "score": float(row[metric]),
                    "non_null": int(row[f"{metric}_non_null"]),
                }
            )
    plot_df = pd.DataFrame(long_rows)
    plot_df.to_csv(out_csv, index=False)

    x = np.arange(len(METRICS))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    for ax, (_, panel_title, panel_runs) in zip(axes, PANELS):
        for i, run_key in enumerate(panel_runs):
            run_df = plot_df.loc[plot_df["run"] == run_key]
            heights = [float(run_df.loc[run_df["metric"] == metric, "score"].iloc[0]) for metric in METRICS]
            non_nulls = [int(run_df.loc[run_df["metric"] == metric, "non_null"].iloc[0]) for metric in METRICS]
            chunk_key = "280_90" if run_key.startswith("280_90") else "224_56"
            bars = ax.bar(
                x + (i - 0.5) * width,
                heights,
                width,
                label=LABELS[run_key].split(" ", 1)[0],
                color=COLORS[chunk_key],
            )
            for bar, non_null in zip(bars, non_nulls):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.015,
                    f"n={non_null}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([metric.replace("_", " ").title() for metric in METRICS], rotation=12, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(panel_title)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.legend(frameon=False)

    axes[0].set_ylabel("RAGAS Mean Score")
    fig.suptitle("RAGAS 75-query chunking comparison", fontsize=14)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=300)
    plt.close(fig)

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
