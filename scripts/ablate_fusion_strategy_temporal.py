"""
Compare hybrid fusion strategies across temporal document transitions.

Strategies:
- rrf: reciprocal-rank fusion over dense + BM25 ranked lists
- score_fusion: weighted sum of per-query min-max normalized dense and BM25 scores

Protocol:
- Build consecutive temporal transitions from available Grampian doc folders.
- For each transition (train_doc -> test_doc), sweep bm25_weight grid with fixed
  rrf_k and dense_weight.
- Select weight on train (Hit@1, then MRR@10, then lower bm25_weight).
- Evaluate selected weight on test.
- Report per-fold and aggregate comparisons for both strategies.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import _matplotlib_env
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
src_path = repo_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from scripts.retrieval_eval_bm25 import BM25Index, get_retrieved_pages, tokenize


DOC_ID_RE = re.compile(r"^Grampian-(\d{4})-(\d{4})$")


@dataclass
class QueryPack:
    query_id: str
    expected_pages: set[int]
    dense_ranked: list[int]
    bm25_ranked: list[int]
    dense_score_map: dict[int, float]
    bm25_score_map: dict[int, float]


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def parse_float_list(s: str) -> list[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("weight grid cannot be empty")
    return vals


def parse_year_start(doc_id: str) -> int:
    m = DOC_ID_RE.match(doc_id)
    if not m:
        return -1
    return int(m.group(1))


def _has_required_artifacts(doc_dir: Path) -> bool:
    required = [
        doc_dir / "faiss.index",
        doc_dir / "chunk_meta.parquet",
        doc_dir / "chunks.parquet",
        doc_dir / "eval_set.json",
    ]
    return all(p.exists() for p in required)


def build_temporal_transitions(run_root: Path) -> list[tuple[str, str, str]]:
    docs: list[str] = []
    for d in sorted(p for p in run_root.iterdir() if p.is_dir()):
        if DOC_ID_RE.match(d.name) and _has_required_artifacts(d):
            docs.append(d.name)
    docs = sorted(docs, key=parse_year_start)
    out: list[tuple[str, str, str]] = []
    for i in range(len(docs) - 1):
        tr = docs[i]
        te = docs[i + 1]
        label = f"fold_{i+1}_{tr}_to_{te}"
        out.append((label, tr, te))
    return out


def load_eval_items(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("queries"), list):
        return obj["queries"]
    return []


def normalize_map(score_map: dict[int, float], candidates: list[int]) -> dict[int, float]:
    vals = np.array([score_map.get(i, np.nan) for i in candidates], dtype=np.float32)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return {i: 0.0 for i in candidates}
    lo = float(np.min(vals[finite]))
    hi = float(np.max(vals[finite]))
    if hi <= lo:
        return {i: 0.0 for i in candidates}
    out: dict[int, float] = {}
    for i in candidates:
        s = score_map.get(i)
        if s is None:
            out[i] = 0.0
        else:
            out[i] = float((float(s) - lo) / (hi - lo))
    return out


def rrf_rank(
    dense_ranked: list[int],
    bm25_ranked: list[int],
    dense_weight: float,
    bm25_weight: float,
    rrf_k: int,
) -> list[int]:
    s: dict[int, float] = {}
    for rank, idx in enumerate(dense_ranked, start=1):
        s[idx] = s.get(idx, 0.0) + (dense_weight / float(rrf_k + rank))
    for rank, idx in enumerate(bm25_ranked, start=1):
        s[idx] = s.get(idx, 0.0) + (bm25_weight / float(rrf_k + rank))
    return [idx for idx, _ in sorted(s.items(), key=lambda kv: kv[1], reverse=True)]


def weighted_score_rank(
    dense_score_map: dict[int, float],
    bm25_score_map: dict[int, float],
    dense_weight: float,
    bm25_weight: float,
) -> list[int]:
    candidates = sorted(list(set(dense_score_map.keys()).union(set(bm25_score_map.keys()))))
    dn = normalize_map(dense_score_map, candidates)
    bn = normalize_map(bm25_score_map, candidates)
    scores = {i: (dense_weight * dn[i]) + (bm25_weight * bn[i]) for i in candidates}
    return [idx for idx, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


def page_hit_at_1(expected_pages: set[int], meta: pd.DataFrame, ranked_idxs: list[int]) -> float:
    if not ranked_idxs:
        return 0.0
    top = ranked_idxs[:1]
    pages: list[int] = []
    for idx in top:
        pages.extend(get_retrieved_pages(meta.iloc[idx]))
    return 1.0 if expected_pages.intersection(set(pages)) else 0.0


def page_mrr_at_10(expected_pages: set[int], meta: pd.DataFrame, ranked_idxs: list[int]) -> float:
    top = ranked_idxs[:10]
    pages: list[int] = []
    for idx in top:
        pages.extend(get_retrieved_pages(meta.iloc[idx]))
    dedup: list[int] = []
    seen: set[int] = set()
    for p in pages:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    for i, p in enumerate(dedup, start=1):
        if p in expected_pages:
            return 1.0 / float(i)
    return 0.0


def eval_packs(
    packs: list[QueryPack],
    meta: pd.DataFrame,
    strategy: str,
    dense_weight: float,
    bm25_weight: float,
    rrf_k: int,
) -> tuple[float, float]:
    hits: list[float] = []
    mrrs: list[float] = []
    for p in packs:
        if strategy == "rrf":
            ranked = rrf_rank(p.dense_ranked, p.bm25_ranked, dense_weight, bm25_weight, rrf_k)
        elif strategy == "score_fusion":
            ranked = weighted_score_rank(p.dense_score_map, p.bm25_score_map, dense_weight, bm25_weight)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        hits.append(page_hit_at_1(p.expected_pages, meta, ranked))
        mrrs.append(page_mrr_at_10(p.expected_pages, meta, ranked))
    return (float(np.mean(hits)) if hits else 0.0, float(np.mean(mrrs)) if mrrs else 0.0)


def build_query_packs(
    data_dir: Path,
    model: SentenceTransformer,
    max_k_search: int,
    bm25_k1: float,
    bm25_b: float,
) -> tuple[pd.DataFrame, list[QueryPack]]:
    index_path = data_dir / "faiss.index"
    meta_path = data_dir / "chunk_meta.parquet"
    chunks_path = data_dir / "chunks.parquet"
    eval_path = data_dir / "eval_set.json"
    for p in [index_path, meta_path, chunks_path, eval_path]:
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    meta = pd.read_parquet(meta_path)
    chunks = pd.read_parquet(chunks_path)
    eval_items = load_eval_items(eval_path)
    if not eval_items:
        raise ValueError(f"Empty eval_set: {eval_path}")

    index = faiss.read_index(str(index_path))
    k_search = min(max_k_search, len(meta))
    questions = [str(x.get("question", "")).strip() for x in eval_items]
    q_emb = model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=False,
    ).astype("float32")
    q_emb = l2_normalize(q_emb).astype("float32")

    text_by_id: dict[str, str] = {}
    if "chunk_id_global" in chunks.columns:
        for _, row in chunks.iterrows():
            cid = str(row.get("chunk_id_global") or "")
            if cid:
                text_by_id[cid] = str(row.get("chunk_text") or "")
    if "chunk_id" in chunks.columns:
        for _, row in chunks.iterrows():
            cid = str(row.get("chunk_id") or "")
            if cid and cid not in text_by_id:
                text_by_id[cid] = str(row.get("chunk_text") or "")

    corpus_texts: list[str] = []
    for _, r in meta.iterrows():
        cid = str(r.get("chunk_id_global") or r.get("chunk_id") or "")
        corpus_texts.append(text_by_id.get(cid, ""))
    bm25 = BM25Index([tokenize(t) for t in corpus_texts], k1=bm25_k1, b=bm25_b)

    packs: list[QueryPack] = []
    for qi, item in enumerate(eval_items):
        q = questions[qi]
        if not q:
            continue
        expected_raw = item.get("expected_pages", [])
        expected = set(int(x) for x in expected_raw) if isinstance(expected_raw, list) else set()

        dense_scores, dense_idxs = index.search(q_emb[qi : qi + 1], k_search)
        dense_ranked = [int(x) for x in dense_idxs[0].tolist()]
        dense_score_map = {int(i): float(s) for i, s in zip(dense_idxs[0].tolist(), dense_scores[0].tolist())}

        bm25_scores = bm25.score_query(tokenize(q))
        bm25_ranked_pairs = sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:k_search]
        bm25_ranked = [int(i) for i, _ in bm25_ranked_pairs]
        bm25_score_map = {int(i): float(s) for i, s in bm25_ranked_pairs}

        packs.append(
            QueryPack(
                query_id=str(item.get("query_id", "")),
                expected_pages=expected,
                dense_ranked=dense_ranked,
                bm25_ranked=bm25_ranked,
                dense_score_map=dense_score_map,
                bm25_score_map=bm25_score_map,
            )
        )
    return meta, packs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablate RRF vs score-fusion across temporal folds.")
    p.add_argument("--run-root", default="data_processed")
    p.add_argument("--model", default="models/all-MiniLM-L6-v2")
    p.add_argument("--dense-weight", type=float, default=0.5)
    p.add_argument("--bm25-grid", default="0.5,0.75,1.0,1.25,1.5,2.0")
    p.add_argument("--rrf-k", type=int, default=20)
    p.add_argument("--max-k-search", type=int, default=200)
    p.add_argument("--bm25-k1", type=float, default=1.5)
    p.add_argument("--bm25-b", type=float, default=0.75)
    p.add_argument(
        "--out-dir",
        default="results/ablations/ablation_thesis_5docs_q50/final_selection/fusion_strategy_temporal_compare_2026-03-01",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bm25_grid = parse_float_list(args.bm25_grid)
    transitions = build_temporal_transitions(run_root)
    if not transitions:
        raise ValueError(f"No consecutive Grampian doc folders found under {run_root}")

    model = SentenceTransformer(str(args.model))

    cache: dict[str, tuple[pd.DataFrame, list[QueryPack]]] = {}
    for _, tr_doc, te_doc in transitions:
        for doc_id in [tr_doc, te_doc]:
            if doc_id in cache:
                continue
            data_dir = run_root / doc_id
            meta, packs = build_query_packs(
                data_dir=data_dir,
                model=model,
                max_k_search=int(args.max_k_search),
                bm25_k1=float(args.bm25_k1),
                bm25_b=float(args.bm25_b),
            )
            cache[doc_id] = (meta, packs)

    sweep_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    for fold_label, tr_doc, te_doc in transitions:
        tr_meta, tr_packs = cache[tr_doc]
        te_meta, te_packs = cache[te_doc]

        for strategy in ["rrf", "score_fusion"]:
            for bw in bm25_grid:
                tr_hit1, tr_mrr10 = eval_packs(
                    packs=tr_packs,
                    meta=tr_meta,
                    strategy=strategy,
                    dense_weight=float(args.dense_weight),
                    bm25_weight=float(bw),
                    rrf_k=int(args.rrf_k),
                )
                te_hit1, te_mrr10 = eval_packs(
                    packs=te_packs,
                    meta=te_meta,
                    strategy=strategy,
                    dense_weight=float(args.dense_weight),
                    bm25_weight=float(bw),
                    rrf_k=int(args.rrf_k),
                )
                sweep_rows.append(
                    {
                        "fold": fold_label,
                        "train_doc_id": tr_doc,
                        "test_doc_id": te_doc,
                        "strategy": strategy,
                        "bm25_weight": float(bw),
                        "train_hit1": tr_hit1,
                        "train_mrr10": tr_mrr10,
                        "test_hit1": te_hit1,
                        "test_mrr10": te_mrr10,
                    }
                )

            g = pd.DataFrame([r for r in sweep_rows if r["fold"] == fold_label and r["strategy"] == strategy]).copy()
            tr_best = g.sort_values(["train_hit1", "train_mrr10", "bm25_weight"], ascending=[False, False, True]).iloc[0]
            te_best = g.sort_values(["test_hit1", "test_mrr10", "bm25_weight"], ascending=[False, False, True]).iloc[0]
            at_sel = g[g["bm25_weight"] == float(tr_best["bm25_weight"])].iloc[0]
            selection_rows.append(
                {
                    "fold": fold_label,
                    "train_doc_id": tr_doc,
                    "test_doc_id": te_doc,
                    "strategy": strategy,
                    "selected_bm25_weight_on_train": float(tr_best["bm25_weight"]),
                    "train_hit1_at_selected": float(tr_best["train_hit1"]),
                    "train_mrr10_at_selected": float(tr_best["train_mrr10"]),
                    "test_hit1_at_selected": float(at_sel["test_hit1"]),
                    "test_mrr10_at_selected": float(at_sel["test_mrr10"]),
                    "best_test_bm25_weight": float(te_best["bm25_weight"]),
                    "best_test_hit1": float(te_best["test_hit1"]),
                    "best_test_mrr10": float(te_best["test_mrr10"]),
                }
            )

    sweep_df = pd.DataFrame(sweep_rows)
    sel_df = pd.DataFrame(selection_rows)

    agg = (
        sel_df.groupby("strategy", as_index=False)
        .agg(
            mean_test_hit1_at_selected=("test_hit1_at_selected", "mean"),
            mean_test_mrr10_at_selected=("test_mrr10_at_selected", "mean"),
            mean_best_test_hit1=("best_test_hit1", "mean"),
            mean_best_test_mrr10=("best_test_mrr10", "mean"),
        )
        .sort_values(["mean_test_hit1_at_selected", "mean_test_mrr10_at_selected"], ascending=False)
    )

    sweep_path = out_dir / "fusion_strategy_temporal_sweep.csv"
    sel_path = out_dir / "fusion_strategy_temporal_selection.csv"
    agg_path = out_dir / "fusion_strategy_temporal_aggregate.csv"
    summary_path = out_dir / "fusion_strategy_temporal_summary.json"
    sweep_df.to_csv(sweep_path, index=False)
    sel_df.to_csv(sel_path, index=False)
    agg.to_csv(agg_path, index=False)

    summary = {
        "fixed_params": {
            "dense_weight": float(args.dense_weight),
            "rrf_k_for_rrf_strategy": int(args.rrf_k),
            "bm25_k1": float(args.bm25_k1),
            "bm25_b": float(args.bm25_b),
            "max_k_search": int(args.max_k_search),
        },
        "bm25_weight_grid": bm25_grid,
        "transitions": [
            {"fold": f, "train_doc_id": tr, "test_doc_id": te}
            for (f, tr, te) in transitions
        ],
        "aggregate": agg.to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Chart 1: test hit@1 at selected train weight across transitions
    fold_order = [x[0] for x in transitions]
    plot_df = sel_df.copy()
    plot_df["fold"] = pd.Categorical(plot_df["fold"], categories=fold_order, ordered=True)
    plot_df = plot_df.sort_values("fold")
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for strategy, g in plot_df.groupby("strategy"):
        ax.plot(g["fold"].astype(str), g["test_hit1_at_selected"], marker="o", linewidth=2, label=strategy)
    ax.set_title("RRF vs Score-Fusion: Test Hit@1 Across Temporal Transitions")
    ax.set_xlabel("Temporal transition (train -> test)")
    ax.set_ylabel("Test Hit@1 at train-selected weight")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    chart1 = out_dir / "chart_fusion_strategy_test_hit1_by_transition.png"
    fig.savefig(chart1, dpi=180)
    plt.close(fig)

    # Chart 2: mean test hit@1 by bm25_weight per strategy
    plot2 = (
        sweep_df.groupby(["strategy", "bm25_weight"], as_index=False)
        .agg(mean_test_hit1=("test_hit1", "mean"), mean_test_mrr10=("test_mrr10", "mean"))
        .sort_values(["strategy", "bm25_weight"])
    )
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for strategy, g in plot2.groupby("strategy"):
        ax.plot(g["bm25_weight"], g["mean_test_hit1"], marker="o", linewidth=2, label=strategy)
    ax.set_title("Mean Test Hit@1 vs BM25 Weight")
    ax.set_xlabel("BM25 weight")
    ax.set_ylabel("Mean test Hit@1")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    chart2 = out_dir / "chart_fusion_strategy_mean_hit1_by_bm25_weight.png"
    fig.savefig(chart2, dpi=180)
    plt.close(fig)

    md = [
        "# Fusion Strategy Temporal Comparison",
        "",
        "Compared strategies:",
        "- `rrf`: reciprocal-rank fusion",
        "- `score_fusion`: weighted sum of normalized dense and BM25 scores",
        "",
        "## Aggregate (across transitions)",
        "",
        agg.to_markdown(index=False),
        "",
        "## Per-fold selection summary",
        "",
        sel_df.to_markdown(index=False),
        "",
        "## Artifacts",
        f"- `{sweep_path}`",
        f"- `{sel_path}`",
        f"- `{agg_path}`",
        f"- `{summary_path}`",
        f"- `{chart1}`",
        f"- `{chart2}`",
    ]
    (out_dir / "fusion_strategy_temporal_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("Saved:")
    print("-", sweep_path)
    print("-", sel_path)
    print("-", agg_path)
    print("-", summary_path)
    print("-", out_dir / "fusion_strategy_temporal_report.md")
    print("-", chart1)
    print("-", chart2)


if __name__ == "__main__":
    main()
