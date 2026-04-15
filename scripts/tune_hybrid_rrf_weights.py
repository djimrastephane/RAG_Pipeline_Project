"""
Tune hybrid RRF fusion weights for Dense + BM25 retrieval.

This script:
- Loads one or more processed retrieval data directories.
- Computes dense and BM25 rank lists per query once.
- Runs a deterministic train/val/test split by query_id hash.
- Sweeps dense_weight, bm25_weight, and rrf_k.
- Selects best config on validation (Hit@1, then MRR@10, then Hit@3).
- Reports selected config and compares it to baseline (1.0/1.0, rrf_k=60) on test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
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


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def parse_k_list(val: str) -> list[int]:
    out = [int(x.strip()) for x in val.split(",") if x.strip()]
    if not out:
        raise ValueError("k-list must contain at least one integer")
    if min(out) <= 0:
        raise ValueError("k-list values must be > 0")
    return out


def parse_float_list(val: str) -> list[float]:
    out = [float(x.strip()) for x in val.split(",") if x.strip()]
    if not out:
        raise ValueError("float grid must contain at least one value")
    return out


def parse_int_list(val: str) -> list[int]:
    out = [int(x.strip()) for x in val.split(",") if x.strip()]
    if not out:
        raise ValueError("int grid must contain at least one value")
    return out


def stable_bucket(text: str) -> float:
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    val = int(h[:8], 16)
    return (val % 10_000) / 10_000.0


def split_name(query_id: str, train_frac: float, val_frac: float) -> str:
    x = stable_bucket(query_id)
    if x < train_frac:
        return "train"
    if x < (train_frac + val_frac):
        return "val"
    return "test"


def rrf_fuse(
    dense_ranked: list[int],
    bm25_ranked: list[int],
    rrf_k: int,
    dense_weight: float,
    bm25_weight: float,
) -> list[int]:
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + (dense_weight / float(rrf_k + rank))
    for rank, idx in enumerate(bm25_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + (bm25_weight / float(rrf_k + rank))
    return [idx for idx, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]


@dataclass
class QueryPack:
    dataset: str
    query_id: str
    expected_pages: set[int]
    dense_ranked: list[int]
    bm25_ranked: list[int]
    split: str


def page_hit_at_k(expected_pages: set[int], meta: pd.DataFrame, ranked_idxs: list[int], k: int) -> float:
    top = ranked_idxs[:k]
    pages: list[int] = []
    for idx in top:
        pages.extend(get_retrieved_pages(meta.iloc[idx]))
    return 1.0 if expected_pages.intersection(set(pages)) else 0.0


def page_mrr_at_k(expected_pages: set[int], meta: pd.DataFrame, ranked_idxs: list[int], k: int) -> float:
    top = ranked_idxs[:k]
    pages: list[int] = []
    for idx in top:
        pages.extend(get_retrieved_pages(meta.iloc[idx]))
    seen: set[int] = set()
    dedup_pages: list[int] = []
    for p in pages:
        if p not in seen:
            seen.add(p)
            dedup_pages.append(p)
    for i, p in enumerate(dedup_pages, start=1):
        if p in expected_pages:
            return 1.0 / float(i)
    return 0.0


def eval_config(
    packs: list[QueryPack],
    meta_by_dataset: dict[str, pd.DataFrame],
    split: str,
    dense_weight: float,
    bm25_weight: float,
    rrf_k: int,
    k_list: list[int],
) -> dict[str, Any]:
    rows = [p for p in packs if p.split == split]
    out: dict[str, Any] = {
        "split": split,
        "num_queries": len(rows),
    }
    if not rows:
        for k in k_list:
            out[f"hit@{k}"] = 0.0
            out[f"mrr@{k}"] = 0.0
        return out

    for k in k_list:
        hits: list[float] = []
        mrrs: list[float] = []
        for p in rows:
            ranked = rrf_fuse(
                dense_ranked=p.dense_ranked,
                bm25_ranked=p.bm25_ranked,
                rrf_k=rrf_k,
                dense_weight=dense_weight,
                bm25_weight=bm25_weight,
            )
            meta = meta_by_dataset[p.dataset]
            hits.append(page_hit_at_k(p.expected_pages, meta, ranked, k))
            mrrs.append(page_mrr_at_k(p.expected_pages, meta, ranked, k))
        out[f"hit@{k}"] = float(np.mean(hits)) if hits else 0.0
        out[f"mrr@{k}"] = float(np.mean(mrrs)) if mrrs else 0.0
    return out


def load_eval_items(eval_set_path: Path) -> list[dict[str, Any]]:
    obj = json.loads(eval_set_path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("queries"), list):
        return obj["queries"]
    return []


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tune hybrid Dense/BM25 RRF weights.")
    p.add_argument(
        "--run-root",
        required=True,
        help="Root dir containing ablation run folders (e.g. results/ablations/ablation_thesis_5docs_q50).",
    )
    p.add_argument(
        "--run-filter",
        default="chunk_280_90_seg_off_dense_rerank_on",
        help="Substring filter for selecting run folders under --run-root.",
    )
    p.add_argument(
        "--model",
        default="models/all-MiniLM-L6-v2",
        help="Sentence-transformers model path/name.",
    )
    p.add_argument(
        "--k-list",
        default="1,3,5,10",
        help="Evaluation k list.",
    )
    p.add_argument(
        "--dense-grid",
        default="0.5,0.75,1.0,1.25,1.5,2.0",
        help="Comma-separated dense weights.",
    )
    p.add_argument(
        "--bm25-grid",
        default="0.5,0.75,1.0,1.25,1.5,2.0",
        help="Comma-separated bm25 weights.",
    )
    p.add_argument(
        "--rrf-k-grid",
        default="20,40,60,80,100",
        help="Comma-separated rrf_k values.",
    )
    p.add_argument(
        "--train-frac",
        type=float,
        default=0.6,
        help="Train split fraction.",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help="Validation split fraction.",
    )
    p.add_argument(
        "--max-k-search",
        type=int,
        default=200,
        help="Depth for dense/BM25 ranking lists.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for tuning artifacts.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    k_list = parse_k_list(args.k_list)
    max_k = max(k_list)
    dense_grid = parse_float_list(args.dense_grid)
    bm25_grid = parse_float_list(args.bm25_grid)
    rrf_k_grid = parse_int_list(args.rrf_k_grid)

    if args.train_frac <= 0 or args.val_frac <= 0 or (args.train_frac + args.val_frac) >= 1.0:
        raise ValueError("train_frac and val_frac must be >0 and sum to <1.0")

    run_dirs = sorted(
        d for d in run_root.iterdir() if d.is_dir() and args.run_filter in d.name
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run dirs matching '{args.run_filter}' under {run_root}")

    model = SentenceTransformer(str(args.model))

    packs: list[QueryPack] = []
    meta_by_dataset: dict[str, pd.DataFrame] = {}
    dataset_rows: list[dict[str, Any]] = []

    for run_dir in run_dirs:
        doc_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
        if not doc_dirs:
            continue
        data_dir = doc_dirs[0]
        dataset = data_dir.name

        index_path = data_dir / "faiss.index"
        meta_path = data_dir / "chunk_meta.parquet"
        chunks_path = data_dir / "chunks.parquet"
        eval_set_path = data_dir / "eval_set.json"
        if not (index_path.exists() and meta_path.exists() and chunks_path.exists() and eval_set_path.exists()):
            continue

        index = faiss.read_index(str(index_path))
        meta = pd.read_parquet(meta_path)
        chunks = pd.read_parquet(chunks_path)
        eval_items = load_eval_items(eval_set_path)
        if not eval_items:
            continue

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
        for _, row in meta.iterrows():
            cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
            corpus_texts.append(text_by_id.get(cid, ""))
        bm25 = BM25Index([tokenize(t) for t in corpus_texts], k1=1.5, b=0.75)

        questions = [str(x.get("question", "")).strip() for x in eval_items]
        q_emb = model.encode(
            questions,
            convert_to_numpy=True,
            normalize_embeddings=False,
            show_progress_bar=False,
        ).astype("float32")
        q_emb = l2_normalize(q_emb).astype("float32")

        max_k_search = min(max(args.max_k_search, max_k), len(meta))
        dense_scores, dense_idxs = index.search(q_emb, max_k_search)

        meta_by_dataset[dataset] = meta
        added = 0
        for qi, item in enumerate(eval_items):
            qid = str(item.get("query_id", "")).strip()
            if not qid:
                continue
            expected_raw = item.get("expected_pages", [])
            expected_pages = {int(x) for x in expected_raw if str(x).isdigit()} if isinstance(expected_raw, list) else set()
            if not expected_pages:
                continue

            bm25_scores = bm25.score_query(tokenize(questions[qi]))
            bm25_ranked = [
                idx for idx, _ in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:max_k_search]
            ]
            packs.append(
                QueryPack(
                    dataset=dataset,
                    query_id=qid,
                    expected_pages=expected_pages,
                    dense_ranked=[int(x) for x in dense_idxs[qi].tolist()],
                    bm25_ranked=[int(x) for x in bm25_ranked],
                    split=split_name(qid, args.train_frac, args.val_frac),
                )
            )
            added += 1

        dataset_rows.append(
            {
                "dataset": dataset,
                "run_dir": str(run_dir),
                "data_dir": str(data_dir),
                "num_queries_loaded": int(added),
                "num_chunks": int(len(meta)),
            }
        )

    if not packs:
        raise RuntimeError("No queries loaded for tuning.")

    grid_rows: list[dict[str, Any]] = []
    for rrf_k in rrf_k_grid:
        for dw in dense_grid:
            for bw in bm25_grid:
                row = {
                    "rrf_k": int(rrf_k),
                    "dense_weight": float(dw),
                    "bm25_weight": float(bw),
                }
                val_metrics = eval_config(
                    packs=packs,
                    meta_by_dataset=meta_by_dataset,
                    split="val",
                    dense_weight=float(dw),
                    bm25_weight=float(bw),
                    rrf_k=int(rrf_k),
                    k_list=k_list,
                )
                row.update({f"val_{k}": v for k, v in val_metrics.items() if k != "split"})
                grid_rows.append(row)

    grid_df = pd.DataFrame(grid_rows)
    sort_cols = []
    if "val_hit@1" in grid_df.columns:
        sort_cols.append("val_hit@1")
    if "val_mrr@10" in grid_df.columns:
        sort_cols.append("val_mrr@10")
    if "val_hit@3" in grid_df.columns:
        sort_cols.append("val_hit@3")
    if not sort_cols:
        sort_cols = [c for c in grid_df.columns if c.startswith("val_hit@")]
    grid_df = grid_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    best = grid_df.iloc[0].to_dict()
    best_rrf_k = int(best["rrf_k"])
    best_dw = float(best["dense_weight"])
    best_bw = float(best["bm25_weight"])

    baseline_rrf_k = 60
    baseline_dw = 1.0
    baseline_bw = 1.0

    best_test = eval_config(
        packs=packs,
        meta_by_dataset=meta_by_dataset,
        split="test",
        dense_weight=best_dw,
        bm25_weight=best_bw,
        rrf_k=best_rrf_k,
        k_list=k_list,
    )
    base_test = eval_config(
        packs=packs,
        meta_by_dataset=meta_by_dataset,
        split="test",
        dense_weight=baseline_dw,
        bm25_weight=baseline_bw,
        rrf_k=baseline_rrf_k,
        k_list=k_list,
    )

    comparison_rows: list[dict[str, Any]] = []
    keys = sorted({k for k in best_test.keys() if k != "split"} | {k for k in base_test.keys() if k != "split"})
    for k in keys:
        b = float(best_test.get(k, 0.0))
        o = float(base_test.get(k, 0.0))
        comparison_rows.append(
            {
                "metric": k,
                "best_config_value": b,
                "baseline_value": o,
                "delta_best_minus_baseline": b - o,
            }
        )

    outputs = {
        "run_root": str(run_root),
        "run_filter": args.run_filter,
        "model": str(args.model),
        "k_list": k_list,
        "train_frac": float(args.train_frac),
        "val_frac": float(args.val_frac),
        "test_frac": float(1.0 - args.train_frac - args.val_frac),
        "num_queries_total": int(len(packs)),
        "num_train": int(sum(1 for p in packs if p.split == "train")),
        "num_val": int(sum(1 for p in packs if p.split == "val")),
        "num_test": int(sum(1 for p in packs if p.split == "test")),
        "best_config_on_val": {
            "rrf_k": best_rrf_k,
            "dense_weight": best_dw,
            "bm25_weight": best_bw,
            "dense_to_bm25_ratio": (best_dw / best_bw) if best_bw != 0 else None,
        },
        "baseline_config": {
            "rrf_k": baseline_rrf_k,
            "dense_weight": baseline_dw,
            "bm25_weight": baseline_bw,
        },
        "best_val_row": best,
        "test_metrics_best": best_test,
        "test_metrics_baseline": base_test,
    }

    dataset_df = pd.DataFrame(dataset_rows)
    comp_df = pd.DataFrame(comparison_rows)

    grid_csv = out_dir / "hybrid_weight_sweep_grid.csv"
    datasets_csv = out_dir / "hybrid_weight_sweep_datasets.csv"
    comparison_csv = out_dir / "hybrid_weight_sweep_test_comparison.csv"
    summary_json = out_dir / "hybrid_weight_sweep_summary.json"

    grid_df.to_csv(grid_csv, index=False)
    dataset_df.to_csv(datasets_csv, index=False)
    comp_df.to_csv(comparison_csv, index=False)
    summary_json.write_text(json.dumps(outputs, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {grid_csv}")
    print(f"Saved: {datasets_csv}")
    print(f"Saved: {comparison_csv}")
    print(f"Saved: {summary_json}")
    print(
        "Best on val: "
        f"rrf_k={best_rrf_k}, dense_weight={best_dw}, bm25_weight={best_bw}, "
        f"ratio={best_dw / best_bw if best_bw != 0 else 'inf'}"
    )


if __name__ == "__main__":
    main()
