#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import _matplotlib_env
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a thesis-ready slide image for RAGAS comparison.")
    p.add_argument("--baseline-summary", default="results/ragas/run75/ragas_summary.json")
    p.add_argument("--generated-summary", default="results/ragas/run75_generated/ragas_summary.json")
    p.add_argument("--compare-chart", default="results/ragas/comparison_75/ragas_75_baseline_vs_generated.png")
    p.add_argument("--violin-chart", default="results/ragas/run75_generated/charts/ragas_metric_violin_jitter.png")
    p.add_argument("--out-dir", default="results/ragas/comparison_75")
    return p.parse_args()


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(Path(args.baseline_summary).read_text(encoding="utf-8"))
    generated = json.loads(Path(args.generated_summary).read_text(encoding="utf-8"))
    b = baseline["metrics_mean"]
    g = generated["metrics_mean"]
    delta = {k: float(g[k]) - float(b[k]) for k in b.keys()}

    comp_img = mpimg.imread(str(Path(args.compare_chart)))
    vio_img = mpimg.imread(str(Path(args.violin_chart)))

    fig = plt.figure(figsize=(15, 8.5))
    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        width_ratios=[1.05, 1.0],
        height_ratios=[0.20, 0.67, 0.13],
        hspace=0.12,
        wspace=0.08,
    )

    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")
    ax_title.text(
        0.0,
        0.84,
        "RAGAS Evaluation: Baseline vs Generated Answers (75 Queries)",
        fontsize=22,
        weight="bold",
        ha="left",
        va="center",
    )
    ax_title.text(
        0.0,
        0.42,
        "Matched setup: 15 queries/doc across 5 Grampian reports, same retrieval config (RRF default).",
        fontsize=12,
        color="#333333",
        ha="left",
        va="center",
    )

    ax_left = fig.add_subplot(gs[1, 0])
    ax_left.imshow(comp_img)
    ax_left.axis("off")
    ax_left.set_title("Aggregate Metric Comparison", fontsize=13, pad=8)

    ax_right = fig.add_subplot(gs[1, 1])
    ax_right.imshow(vio_img)
    ax_right.axis("off")
    ax_right.set_title("Distribution (Generated-Answer Run)", fontsize=13, pad=8)

    ax_bottom = fig.add_subplot(gs[2, :])
    ax_bottom.axis("off")
    bullets = [
        f"Answer quality improved materially with generation: answer_relevancy +{_fmt(delta['answer_relevancy'])}, faithfulness +{_fmt(delta['faithfulness'])}.",
        f"Retrieval-oriented metrics stayed stable: context_precision +{_fmt(delta['context_precision'])}, context_recall {delta['context_recall']:+.3f}.",
        "Interpretation: retrieval is already strong; generation mainly raises final answer quality. Non-null coverage <100% reflects evaluator timeouts/parsing failures.",
    ]
    y = 0.86
    for bline in bullets:
        ax_bottom.text(0.01, y, f"• {bline}", fontsize=12, ha="left", va="top")
        y -= 0.31

    out_png = out_dir / "ragas_thesis_slide.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    out_md = out_dir / "ragas_thesis_slide_notes.md"
    out_md.write_text(
        "\n".join(
            [
                "# RAGAS Thesis Slide Notes",
                "",
                "- Setup: 75 queries, 15 per Grampian document, same retrieval configuration (RRF default).",
                f"- answer_relevancy: {b['answer_relevancy']:.3f} -> {g['answer_relevancy']:.3f} (delta {delta['answer_relevancy']:+.3f})",
                f"- faithfulness: {b['faithfulness']:.3f} -> {g['faithfulness']:.3f} (delta {delta['faithfulness']:+.3f})",
                f"- context_precision: {b['context_precision']:.3f} -> {g['context_precision']:.3f} (delta {delta['context_precision']:+.3f})",
                f"- context_recall: {b['context_recall']:.3f} -> {g['context_recall']:.3f} (delta {delta['context_recall']:+.3f})",
                "",
                "Key message: generated answers materially improved answer-level quality while retrieval-level metrics remained stable.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Wrote {out_png}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
