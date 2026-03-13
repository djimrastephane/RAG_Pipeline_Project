#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot relevant vs non-relevant similarity distributions using eval labels."
    )
    p.add_argument("--doc-id", required=True, help="Document id under data_processed/")
    p.add_argument("--data-root", default="data_processed", help="Root with processed doc folders")
    p.add_argument("--model", default="models/all-MiniLM-L6-v2", help="Embedding model path")
    p.add_argument("--out-dir", default="results", help="Output directory")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument(
        "--negative-sample-ratio",
        type=float,
        default=1.0,
        help="How many non-relevant samples per relevant sample.",
    )
    return p.parse_args()


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def _load_eval_items(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    obj: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        q = obj.get("queries")
        if isinstance(q, list):
            return [x for x in q if isinstance(x, dict)]
    return []


def _to_page_set(v: Any) -> set[int]:
    out: set[int] = set()
    if isinstance(v, (list, tuple, np.ndarray, set)):
        for x in list(v):
            try:
                out.add(int(x))
            except Exception:
                continue
        return out
    if isinstance(v, dict) and "element" in v:
        try:
            out.add(int(v.get("element")))
        except Exception:
            pass
        return out
    try:
        out.add(int(v))
    except Exception:
        pass
    return out


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(int(args.seed))
    doc_dir = Path(args.data_root) / args.doc_id
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_path = doc_dir / "embeddings.npy"
    meta_path = doc_dir / "chunk_meta.parquet"
    eval_path = doc_dir / "eval_set.json"
    if not emb_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing required artifacts for {args.doc_id}")

    emb = np.load(emb_path).astype(np.float32, copy=False)
    emb = l2_normalize(emb).astype(np.float32)
    meta = pd.read_parquet(meta_path)
    n = min(len(meta), emb.shape[0])
    emb = emb[:n]
    meta = meta.iloc[:n].reset_index(drop=True)
    pages_per_chunk = [_to_page_set(v) for v in meta.get("pages", [[]] * n).tolist()]

    eval_items = _load_eval_items(eval_path)
    if not eval_items:
        raise ValueError(f"No eval_set queries found for {args.doc_id}; cannot build relevance distributions.")

    questions: list[str] = []
    expected_sets: list[set[int]] = []
    for it in eval_items:
        q = str(it.get("question") or "").strip()
        eps = _to_page_set(it.get("expected_pages"))
        if q and eps:
            questions.append(q)
            expected_sets.append(eps)

    if not questions:
        raise ValueError(f"No usable eval questions with expected pages for {args.doc_id}.")

    model = SentenceTransformer(str(args.model))
    q_emb = model.encode(questions, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
    q_emb = l2_normalize(q_emb).astype(np.float32)
    score_matrix = q_emb @ emb.T

    relevant_scores: list[float] = []
    nonrelevant_scores: list[float] = []
    used_queries = 0
    neg_ratio = max(0.1, float(args.negative_sample_ratio))

    for qi, exp_pages in enumerate(expected_sets):
        rel_idx = [i for i, pset in enumerate(pages_per_chunk) if pset and (pset & exp_pages)]
        if not rel_idx:
            continue
        used_queries += 1
        scores = score_matrix[qi]
        relevant_scores.extend(scores[rel_idx].tolist())

        non_idx = np.array([i for i in range(n) if i not in set(rel_idx)], dtype=np.int32)
        if len(non_idx) == 0:
            continue
        take = min(len(non_idx), max(1, int(round(len(rel_idx) * neg_ratio))))
        sampled = rng.choice(non_idx, size=take, replace=False)
        nonrelevant_scores.extend(scores[sampled].tolist())

    rel = np.asarray(relevant_scores, dtype=np.float32)
    non = np.asarray(nonrelevant_scores, dtype=np.float32)
    if len(rel) == 0 or len(non) == 0:
        raise ValueError(f"Insufficient relevant/non-relevant samples for {args.doc_id}.")

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    bins = np.linspace(-0.1, 1.0, 60)
    ax.hist(rel, bins=bins, density=True, alpha=0.55, label="Relevant chunks")
    ax.hist(non, bins=bins, density=True, alpha=0.55, label="Non-relevant chunks")
    rel_mean = float(np.mean(rel))
    non_mean = float(np.mean(non))
    ax.axvline(
        rel_mean,
        linestyle="--",
        linewidth=2.0,
        color="#1b9e77",
        label=f"Relevant mean = {rel_mean:.3f}",
    )
    ax.axvline(
        non_mean,
        linestyle="--",
        linewidth=2.0,
        color="#d95f02",
        label=f"Non-relevant mean = {non_mean:.3f}",
    )
    ax.set_title(f"Similarity distribution: relevant vs non-relevant chunks ({args.doc_id})")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig_path = out_dir / f"vector_similarity_relevant_vs_nonrelevant_{args.doc_id}.png"
    fig.savefig(fig_path, dpi=240)
    plt.close(fig)

    summary = pd.DataFrame(
        [
            {
                "doc_id": args.doc_id,
                "n_chunks": int(n),
                "n_eval_queries_total": int(len(eval_items)),
                "n_eval_queries_used": int(used_queries),
                "n_relevant_samples": int(len(rel)),
                "n_nonrelevant_samples": int(len(non)),
                "relevant_mean": float(np.mean(rel)),
                "relevant_std": float(np.std(rel)),
                "nonrelevant_mean": float(np.mean(non)),
                "nonrelevant_std": float(np.std(non)),
                "delta_relevant_minus_nonrelevant_mean": float(np.mean(rel) - np.mean(non)),
                "negative_sample_ratio": float(neg_ratio),
                "seed": int(args.seed),
            }
        ]
    )
    summary_path = out_dir / f"vector_similarity_relevant_vs_nonrelevant_summary_{args.doc_id}.csv"
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {fig_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
