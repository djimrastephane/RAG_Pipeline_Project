#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRICS = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create visual summary charts for RAGAS outputs.")
    p.add_argument("--per-query-csv", default="results/ragas/run75/ragas_per_query.csv")
    p.add_argument("--input-jsonl", default="results/ragas/ragas_input_75q_15perdoc.jsonl")
    p.add_argument("--out-dir", default="results/ragas/run75/charts")
    p.add_argument("--run-title", default="Distribution")
    p.add_argument("--run-subtitle", default="Generated answer")
    return p.parse_args()


def _load_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def _safe_mean(s: pd.Series) -> float:
    s2 = pd.to_numeric(s, errors="coerce").dropna()
    if s2.empty:
        return float("nan")
    return float(s2.mean())


def main() -> None:
    args = parse_args()
    per_query_path = Path(args.per_query_csv)
    input_jsonl_path = Path(args.input_jsonl)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_query_path)
    if not set(METRICS).issubset(df.columns):
        raise ValueError(f"Expected metric columns {METRICS} in {per_query_path}")

    # join doc_id/difficulty back by row order (same order as dataset input)
    inp = _load_jsonl(input_jsonl_path)
    if len(inp) == len(df):
        for c in ["doc_id", "difficulty", "query_id", "answer_type"]:
            if c in inp.columns:
                df[c] = inp[c]

    # 1) Metric means with non-null annotations
    means = []
    non_null = []
    for m in METRICS:
        col = pd.to_numeric(df[m], errors="coerce")
        means.append(float(col.mean(skipna=True)))
        non_null.append(int(col.notna().sum()))

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    bars = ax.bar(METRICS, means, color=["#d95f02", "#7570b3", "#1b9e77", "#66a61e"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("RAGAS Mean Scores (75 queries)")
    for b, n in zip(bars, non_null):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"n={n}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "ragas_metric_means.png", dpi=200)
    plt.close(fig)

    # 2) Distribution boxplot
    vals = []
    for m in METRICS:
        s = pd.to_numeric(df[m], errors="coerce").dropna()
        # RAGAS metrics are expected in [0, 1]; clip tiny numeric spillover.
        s = s.clip(lower=0.0, upper=1.0)
        vals.append(s.values)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    bp = ax.boxplot(vals, labels=METRICS, patch_artist=True, showfliers=True, whis=1.5)
    colors = ["#d95f02", "#7570b3", "#1b9e77", "#66a61e"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("RAGAS Score Distributions (75 queries)")
    fig.tight_layout()
    fig.savefig(out_dir / "ragas_metric_boxplot.png", dpi=200)
    plt.close(fig)

    # 2b) Violin + jitter distribution plot used in thesis charts
    colors = ["#e6a15d", "#a9a8c9", "#7fc7b4", "#a8c97e"]
    point_colors = ["#e69138", "#7f7fbf", "#49b3a2", "#7aa342"]
    positions = np.arange(1, len(METRICS) + 1)
    fig, ax = plt.subplots(figsize=(10.105, 5.72))
    fig.suptitle(args.run_title, fontsize=18, y=0.97)

    violin = ax.violinplot(
        vals,
        positions=positions,
        widths=0.8,
        showmeans=True,
        showmedians=False,
        showextrema=False,
    )
    for body, color in zip(violin["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("#8a8a8a")
        body.set_alpha(0.72)
        body.set_linewidth(1.0)
    violin["cmeans"].set_color("#5f5f5f")
    violin["cmeans"].set_linewidth(1.6)

    rng = np.random.default_rng(42)
    for i, (series, color) in enumerate(zip(vals, point_colors), start=1):
        jitter = rng.normal(loc=0.0, scale=0.035, size=len(series))
        ax.scatter(
            np.full(len(series), i) + jitter,
            series,
            s=7,
            color=color,
            alpha=0.45,
            edgecolors="none",
            zorder=3,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(METRICS, rotation=18, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score (0-1)")
    ax.set_title(f"RAGAS Distributions ({len(df)} queries) - {args.run_subtitle}", fontsize=11, pad=4)
    ax.grid(axis="y", linestyle="-", alpha=0.18)
    fig.tight_layout()
    fig.savefig(out_dir / "ragas_metric_violin_jitter.png", dpi=200)
    plt.close(fig)

    # 3) Non-null coverage
    coverage = [n / max(1, len(df)) for n in non_null]
    fig, ax = plt.subplots(figsize=(8.2, 4.3))
    bars = ax.bar(METRICS, coverage, color="#4c78a8")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Non-null ratio")
    ax.set_title("RAGAS Metric Coverage (non-null)")
    for b, n in zip(bars, non_null):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{n}/{len(df)}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "ragas_metric_coverage.png", dpi=200)
    plt.close(fig)

    # 4) By-doc heatmap-like matrix
    if "doc_id" in df.columns:
        agg = (
            df.groupby("doc_id")[METRICS]
            .agg(lambda s: _safe_mean(s))
            .reindex(sorted(df["doc_id"].dropna().unique().tolist()))
        )
        arr = agg.to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(8.8, max(3.2, 0.45 * len(agg) + 1.8)))
        im = ax.imshow(arr, aspect="auto", vmin=0, vmax=1, cmap="YlGnBu")
        ax.set_xticks(np.arange(len(METRICS)))
        ax.set_xticklabels(METRICS, rotation=20, ha="right")
        ax.set_yticks(np.arange(len(agg.index)))
        ax.set_yticklabels(agg.index)
        ax.set_title("RAGAS Means by Document")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                if np.isfinite(arr[i, j]):
                    ax.text(j, i, f"{arr[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("Score")
        fig.tight_layout()
        fig.savefig(out_dir / "ragas_by_doc_heatmap.png", dpi=220)
        plt.close(fig)
        agg.reset_index().to_csv(out_dir / "ragas_by_doc_summary.csv", index=False)

    summary = {
        "n_queries": int(len(df)),
        "metrics_mean": {m: float(means[i]) for i, m in enumerate(METRICS)},
        "metrics_non_null": {m: int(non_null[i]) for i, m in enumerate(METRICS)},
        "charts": [
            "ragas_metric_means.png",
            "ragas_metric_boxplot.png",
            "ragas_metric_violin_jitter.png",
            "ragas_metric_coverage.png",
            "ragas_by_doc_heatmap.png",
        ],
    }
    (out_dir / "ragas_visual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote charts in {out_dir}")


if __name__ == "__main__":
    main()
