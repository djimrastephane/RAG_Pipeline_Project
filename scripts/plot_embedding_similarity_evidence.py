#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import _matplotlib_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate paper-ready embedding similarity charts for a processed document."
    )
    p.add_argument("--doc-id", required=True, help="Document id under data_processed/")
    p.add_argument("--data-root", default="data_processed", help="Root containing processed doc folders.")
    p.add_argument("--model", default="models/all-MiniLM-L6-v2", help="Embedding model path.")
    p.add_argument("--out-dir", default="results", help="Output directory.")
    p.add_argument("--sample-size", type=int, default=20000, help="Random chunk-pair sample size.")
    p.add_argument("--topk", type=int, default=5, help="Top-k retrieved chunk similarities per query.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    return p.parse_args()


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def load_eval_questions(eval_path: Path) -> list[str]:
    if not eval_path.exists():
        return []
    obj: Any = json.loads(eval_path.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = []
    if isinstance(obj, list):
        items = [x for x in obj if isinstance(x, dict)]
    elif isinstance(obj, dict):
        q = obj.get("queries")
        if isinstance(q, list):
            items = [x for x in q if isinstance(x, dict)]
    out: list[str] = []
    for it in items:
        q = str(it.get("question") or "").strip()
        if q:
            out.append(q)
    return out


def sample_random_cosine_pairs(emb: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    n = int(emb.shape[0])
    if n < 2:
        return np.array([], dtype=np.float32)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=sample_size)
    j = rng.integers(0, n, size=sample_size)
    same = i == j
    while np.any(same):
        j[same] = rng.integers(0, n, size=int(np.sum(same)))
        same = i == j
    return np.sum(emb[i] * emb[j], axis=1)


def compute_retrieved_similarities(
    emb: np.ndarray,
    questions: list[str],
    model: SentenceTransformer,
    topk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not questions:
        return (
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )
    q_emb = model.encode(questions, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
    q_emb = l2_normalize(q_emb).astype(np.float32)
    scores = q_emb @ emb.T
    k = max(1, min(int(topk), emb.shape[0]))
    top_idx = np.argpartition(scores, -k, axis=1)[:, -k:]
    top_scores = np.take_along_axis(scores, top_idx, axis=1)
    top_scores = np.sort(top_scores, axis=1)[:, ::-1]
    top1 = top_scores[:, 0]
    if top_scores.shape[1] > 1:
        top2 = top_scores[:, 1]
        margins = top1 - top2
    else:
        margins = np.array([], dtype=np.float32)
    flat_topk = top_scores.reshape(-1)
    return flat_topk.astype(np.float32), top1.astype(np.float32), margins.astype(np.float32)


def _format_report_label(label: str) -> str:
    """Convert e.g. Grampian-2022-2023 -> Grampian 2022–2023."""
    text = str(label).replace("_", " ")
    m = re.match(r"^(.*?)-(\d{4})-(\d{4})$", text)
    if m:
        prefix = m.group(1).replace("-", " ").strip()
        return f"{prefix} {m.group(2)}\u2013{m.group(3)}".strip()
    return text.replace("-", " ").strip()


def plot_similarity_margin_histogram(
    margins: list[float] | np.ndarray,
    report_label: str,
    output_path: str | Path,
    bins: int = 32,
    dpi: int = 300,
) -> None:
    """Plot a thesis-ready histogram of retrieval similarity margins."""
    vals = np.asarray(margins, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("margins is empty; cannot plot similarity margin histogram.")

    mean_val = float(np.mean(vals))
    median_val = float(np.median(vals))
    q1 = float(np.quantile(vals, 0.25))

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    counts, _, _ = ax.hist(
        vals,
        bins=int(bins),
        color="#7f8c8d",
        edgecolor="#4d5656",
        linewidth=0.6,
        alpha=0.85,
    )

    # Small margins indicate weak separation between top-1 and top-2 retrieved chunks,
    # suggesting ranking uncertainty.
    x_min = min(0.0, float(np.min(vals)))
    x_max = max(float(np.max(vals)) * 1.05, q1 * 1.35)
    ax.set_xlim(x_min, x_max)
    ax.axvspan(x_min, q1, color="#efc27b", alpha=0.24, zorder=0)
    y_max = float(np.max(counts)) if counts.size else 1.0
    ax.text(
        x_min + (q1 - x_min) * 0.08,
        y_max * 0.95,
        "Ambiguous retrieval zone (<= Q1)",
        fontsize=9,
        color="#2c3e50",
        ha="left",
        va="top",
    )

    q1_line = ax.axvline(q1, linestyle="-.", linewidth=1.6, color="#b9770e", label=f"Q1: {q1:.3f}")
    mean_line = ax.axvline(mean_val, linestyle="-", linewidth=1.8, color="#8e44ad", label=f"Mean: {mean_val:.3f}")
    median_line = ax.axvline(median_val, linestyle="--", linewidth=1.8, color="#1f618d", label=f"Median: {median_val:.3f}")

    # Single legend box: threshold marker + line-handle entries for distribution summaries.
    counts_h1 = Patch(facecolor="none", edgecolor="none", label="Ambiguous zone defined by Q1")
    legend = ax.legend(
        handles=[counts_h1, q1_line, mean_line, median_line],
        title="Margin summary",
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        frameon=True,
        framealpha=0.98,
        facecolor="#fff8e6",
        edgecolor="#c8a96a",
        fontsize=9.0,
        title_fontsize=9.4,
        handlelength=2.2,
        borderpad=0.45,
        labelspacing=0.45,
    )
    legend.get_title().set_weight("bold")

    ax.set_title("Similarity Margin Distribution", fontsize=13, pad=14)
    ax.text(
        0.5,
        1.01,
        f"(Dense retriever · {_format_report_label(report_label)})",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#4d4d4d",
    )
    ax.set_xlabel("Similarity margin (top-1 minus top-2)", fontsize=11)
    ax.set_ylabel("Query count", fontsize=11)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight")
    plt.close(fig)


def plot_similarity_margin_ecdf(
    margins: list[float] | np.ndarray,
    report_label: str,
    output_path: str | Path,
    dpi: int = 300,
) -> None:
    """Plot thesis-ready ECDF of similarity margins with a Q1-based ambiguous zone."""
    vals = np.asarray(margins, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        raise ValueError("margins is empty; cannot plot ECDF.")

    xs = np.sort(vals)
    ys = np.arange(1, xs.size + 1, dtype=np.float64) / float(xs.size)

    q1 = float(np.quantile(vals, 0.25))
    p_q1 = float(np.mean(vals <= q1))

    fig, ax = plt.subplots(figsize=(8.4, 5.2))
    fig.patch.set_edgecolor("none")

    x_min = min(0.0, float(np.min(vals)))
    x_max = max(float(np.max(vals)) * 1.05, q1 * 1.4)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, 1.02)

    ax.axvspan(x_min, q1, color="#efc27b", alpha=0.24, zorder=0)
    ax.axvline(q1, linestyle="--", linewidth=1.3, color="#b9770e", zorder=2)
    ax.text(
        x_min + (q1 - x_min) * 0.10,
        0.95,
        "Ambiguous retrieval zone (<= Q1)",
        fontsize=9,
        color="#2c3e50",
        ha="left",
        va="top",
    )

    # ECDF line (thinner for clarity against annotations).
    ax.step(xs, ys, where="post", color="#2c3e50", linewidth=1.5, zorder=3)

    ax.hlines(y=p_q1, xmin=x_min, xmax=q1, colors="#7f8c8d", linestyles="-", linewidth=1.0)

    ax.plot(
        [q1],
        [p_q1],
        marker="o",
        markersize=5.0,
        markerfacecolor="white",
        markeredgecolor="#b9770e",
        markeredgewidth=1.3,
        linestyle="None",
        zorder=4,
    )

    ax.annotate(
        "Quartile boundary",
        xy=(q1, p_q1),
        xytext=(8, 8),
        textcoords="offset points",
        fontsize=9,
        color="#4d4d4d",
        ha="left",
        va="bottom",
    )
    ax.annotate(
        f"Q1 = {q1:.3f}",
        xy=(q1, 0.0),
        xytext=(0, -14),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=8.5,
        color="#b9770e",
        clip_on=False,
    )

    ax.set_title("ECDF of Similarity Margins", fontsize=13, pad=14)
    ax.text(
        0.5,
        1.01,
        f"(Dense retriever · {_format_report_label(report_label)})",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color="#4d4d4d",
    )
    ax.set_xlabel("Similarity margin (top-1 minus top-2)", fontsize=11)
    ax.set_ylabel("Cumulative proportion", fontsize=11)
    ax.grid(True, alpha=0.3, linewidth=0.8, color="#d5d8dc")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    doc_dir = Path(args.data_root) / args.doc_id
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = doc_dir / "embeddings.npy"
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing embeddings: {emb_path}")
    emb = np.load(emb_path).astype(np.float32, copy=False)
    emb = l2_normalize(emb).astype(np.float32)

    eval_questions = load_eval_questions(doc_dir / "eval_set.json")
    model = SentenceTransformer(str(args.model))

    random_cos = sample_random_cosine_pairs(emb, sample_size=int(args.sample_size), seed=int(args.seed))
    retrieved_topk_cos, retrieved_top1_cos, retrieved_margins = compute_retrieved_similarities(
        emb=emb,
        questions=eval_questions,
        model=model,
        topk=int(args.topk),
    )

    # 1) Chunk-chunk random pair cosine histogram
    fig1, ax1 = plt.subplots(figsize=(8.0, 5.0))
    ax1.hist(random_cos, bins=55, alpha=0.85)
    ax1.set_title(f"Distribution of cosine similarities between chunk embeddings ({args.doc_id})")
    ax1.set_xlabel("Cosine similarity")
    ax1.set_ylabel("Frequency")
    fig1.tight_layout()
    out_hist_random = out_dir / f"vector_similarity_hist_chunk_pairs_{args.doc_id}.png"
    fig1.savefig(out_hist_random, dpi=220)
    plt.close(fig1)

    # 2) Random vs retrieved similarities overlay
    fig2, ax2 = plt.subplots(figsize=(8.0, 5.0))
    bins = np.linspace(-0.1, 1.0, 60)
    ax2.hist(random_cos, bins=bins, density=True, alpha=0.50, label="Random chunk pairs")
    if len(retrieved_topk_cos) > 0:
        ax2.hist(
            retrieved_topk_cos,
            bins=bins,
            density=True,
            alpha=0.50,
            label=f"Retrieved chunk similarities (eval top-{int(args.topk)})",
        )
    ax2.set_title(f"Random vs retrieved chunk similarity distributions ({args.doc_id})")
    ax2.set_xlabel("Cosine similarity")
    ax2.set_ylabel("Density")
    ax2.legend()
    fig2.tight_layout()
    out_overlay = out_dir / f"vector_similarity_random_vs_retrieved_{args.doc_id}.png"
    fig2.savefig(out_overlay, dpi=220)
    plt.close(fig2)

    # 3) Top-1 retrieved similarity histogram
    out_top1 = out_dir / f"vector_similarity_hist_retrieved_top1_{args.doc_id}.png"
    if len(retrieved_top1_cos) > 0:
        fig3, ax3 = plt.subplots(figsize=(8.0, 5.0))
        ax3.hist(retrieved_top1_cos, bins=40, alpha=0.85)
        ax3.set_title(f"Distribution of top-1 retrieved chunk similarities ({args.doc_id})")
        ax3.set_xlabel("Cosine similarity")
        ax3.set_ylabel("Frequency")
        fig3.tight_layout()
        fig3.savefig(out_top1, dpi=220)
        plt.close(fig3)

    out_margin = out_dir / f"similarity_margin_histogram_{args.doc_id}.png"
    out_margin_ecdf = out_dir / f"similarity_margin_ecdf_{args.doc_id}.png"
    if len(retrieved_margins) > 0:
        plot_similarity_margin_histogram(
            margins=retrieved_margins,
            report_label=str(args.doc_id),
            output_path=out_margin,
        )
        plot_similarity_margin_ecdf(
            margins=retrieved_margins,
            report_label=str(args.doc_id),
            output_path=out_margin_ecdf,
        )

    summary = {
        "doc_id": args.doc_id,
        "n_chunks": int(emb.shape[0]),
        "n_eval_questions": int(len(eval_questions)),
        "topk": int(args.topk),
        "seed": int(args.seed),
        "sample_size_random_pairs": int(len(random_cos)),
        "random_cos_mean": float(np.mean(random_cos)) if len(random_cos) else None,
        "random_cos_std": float(np.std(random_cos)) if len(random_cos) else None,
        "retrieved_topk_mean": float(np.mean(retrieved_topk_cos)) if len(retrieved_topk_cos) else None,
        "retrieved_topk_std": float(np.std(retrieved_topk_cos)) if len(retrieved_topk_cos) else None,
        "retrieved_top1_mean": float(np.mean(retrieved_top1_cos)) if len(retrieved_top1_cos) else None,
        "retrieved_top1_std": float(np.std(retrieved_top1_cos)) if len(retrieved_top1_cos) else None,
        "retrieval_margin_mean": float(np.mean(retrieved_margins)) if len(retrieved_margins) else None,
        "retrieval_margin_median": float(np.median(retrieved_margins)) if len(retrieved_margins) else None,
        "retrieval_margin_lt_0_01_count": int(np.sum(retrieved_margins < 0.01)) if len(retrieved_margins) else 0,
        "retrieval_margin_lt_0_01_pct": (
            float(100.0 * np.mean(retrieved_margins < 0.01)) if len(retrieved_margins) else 0.0
        ),
        "retrieval_margin_lt_0_02_count": int(np.sum(retrieved_margins < 0.02)) if len(retrieved_margins) else 0,
        "retrieval_margin_lt_0_02_pct": (
            float(100.0 * np.mean(retrieved_margins < 0.02)) if len(retrieved_margins) else 0.0
        ),
        "delta_retrieved_topk_minus_random_mean": (
            float(np.mean(retrieved_topk_cos) - np.mean(random_cos))
            if len(retrieved_topk_cos) and len(random_cos)
            else None
        ),
    }
    summary_df = pd.DataFrame([summary])
    out_summary = out_dir / f"vector_similarity_random_vs_retrieved_summary_{args.doc_id}.csv"
    summary_df.to_csv(out_summary, index=False)

    print(f"Wrote {out_hist_random}")
    print(f"Wrote {out_overlay}")
    if len(retrieved_top1_cos) > 0:
        print(f"Wrote {out_top1}")
    if len(retrieved_margins) > 0:
        print(f"Wrote {out_margin}")
        print(f"Wrote {out_margin_ecdf}")
    print(f"Wrote {out_summary}")


if __name__ == "__main__":
    main()
    def _format_report_label(label: str) -> str:
        # Convert e.g. "Grampian-2022-2023" -> "Grampian 2022–2023"
        text = str(label).replace("_", " ")
        m = re.match(r"^(.*?)-(\d{4})-(\d{4})$", text)
        if m:
            prefix = m.group(1).replace("-", " ").strip()
            return f"{prefix} {m.group(2)}\u2013{m.group(3)}".strip()
        return text.replace("-", " ").strip()
