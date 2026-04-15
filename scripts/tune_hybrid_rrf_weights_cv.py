"""
Repeated nested CV tuning for hybrid Dense+BM25 RRF weights.

Workflow:
- Load ablation data dirs matching run-filter.
- Precompute dense and BM25 ranked lists per query.
- For each repeat/fold:
  - test fold = f
  - val fold = (f + 1) % n_folds
  - train folds = remaining (kept for protocol completeness)
  - choose best config on val using Hit@1, then MRR@10, then Hit@3
  - evaluate chosen config on test
- Also aggregate every config directly on all test folds.
- Recommend one stable production config from CV aggregate:
  max mean Hit@1, then max mean MRR@10, then min std Hit@1, then min std MRR@10.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def parse_k_list(val: str) -> list[int]:
    out = [int(x.strip()) for x in val.split(",") if x.strip()]
    if not out or min(out) <= 0:
        raise ValueError("k-list must contain positive integers")
    return out


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def fold_bucket(query_id: str, repeat_id: int, n_folds: int) -> int:
    h = hashlib.sha1(f"{repeat_id}|{query_id}".encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n_folds


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


def eval_rows(
    packs: list[QueryPack],
    meta_by_dataset: dict[str, pd.DataFrame],
    dense_weight: float,
    bm25_weight: float,
    rrf_k: int,
    k_list: list[int],
) -> dict[str, float]:
    if not packs:
        out: dict[str, float] = {"num_queries": 0.0}
        for k in k_list:
            out[f"hit@{k}"] = 0.0
            out[f"mrr@{k}"] = 0.0
        return out

    out = {"num_queries": float(len(packs))}
    for k in k_list:
        hits: list[float] = []
        mrrs: list[float] = []
        for p in packs:
            ranked = rrf_fuse(p.dense_ranked, p.bm25_ranked, rrf_k, dense_weight, bm25_weight)
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
    p = argparse.ArgumentParser(description="Repeated nested CV tuning for hybrid RRF weights.")
    p.add_argument("--run-root", required=True)
    p.add_argument("--run-filter", default="chunk_280_90_seg_off_dense_rerank_on")
    p.add_argument("--model", default="models/all-MiniLM-L6-v2")
    p.add_argument("--k-list", default="1,3,5,10")
    p.add_argument("--dense-grid", default="0.5,0.75,1.0,1.25,1.5,2.0")
    p.add_argument("--bm25-grid", default="0.5,0.75,1.0,1.25,1.5,2.0")
    p.add_argument("--rrf-k-grid", default="20,40,60,80,100")
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--max-k-search", type=int, default=200)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.n_folds < 3:
        raise ValueError("n-folds must be >= 3")
    if args.n_repeats < 1:
        raise ValueError("n-repeats must be >= 1")

    k_list = parse_k_list(args.k_list)
    dense_grid = parse_float_list(args.dense_grid)
    bm25_grid = parse_float_list(args.bm25_grid)
    rrf_k_grid = parse_int_list(args.rrf_k_grid)
    max_k = max(k_list)

    configs: list[tuple[int, float, float]] = []
    for rrf_k in rrf_k_grid:
        for dw in dense_grid:
            for bw in bm25_grid:
                configs.append((int(rrf_k), float(dw), float(bw)))

    run_dirs = sorted([d for d in run_root.iterdir() if d.is_dir() and args.run_filter in d.name])
    if not run_dirs:
        raise FileNotFoundError(f"No run directories matching '{args.run_filter}' under {run_root}")

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
        k_search = min(max(args.max_k_search, max_k), len(meta))
        _, dense_idxs = index.search(q_emb, k_search)

        meta_by_dataset[dataset] = meta
        kept = 0
        for qi, item in enumerate(eval_items):
            qid = str(item.get("query_id", "")).strip()
            if not qid:
                continue
            expected_raw = item.get("expected_pages", [])
            expected_pages = {int(x) for x in expected_raw if str(x).isdigit()} if isinstance(expected_raw, list) else set()
            if not expected_pages:
                continue
            bm25_scores = bm25.score_query(tokenize(questions[qi]))
            bm25_ranked = [idx for idx, _ in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True)[:k_search]]
            packs.append(
                QueryPack(
                    dataset=dataset,
                    query_id=qid,
                    expected_pages=expected_pages,
                    dense_ranked=[int(x) for x in dense_idxs[qi].tolist()],
                    bm25_ranked=[int(x) for x in bm25_ranked],
                )
            )
            kept += 1

        dataset_rows.append(
            {
                "dataset": dataset,
                "run_dir": str(run_dir),
                "data_dir": str(data_dir),
                "num_queries_loaded": int(kept),
                "num_chunks": int(len(meta)),
            }
        )

    if not packs:
        raise RuntimeError("No queries loaded.")

    split_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    test_rows_all_configs: list[dict[str, Any]] = []

    for rep in range(args.n_repeats):
        for fold in range(args.n_folds):
            val_fold = (fold + 1) % args.n_folds
            test_q = [p for p in packs if fold_bucket(p.query_id, rep, args.n_folds) == fold]
            val_q = [p for p in packs if fold_bucket(p.query_id, rep, args.n_folds) == val_fold]
            train_q = [
                p
                for p in packs
                if fold_bucket(p.query_id, rep, args.n_folds) not in {fold, val_fold}
            ]
            if not test_q or not val_q or not train_q:
                continue

            # Evaluate all configs on val and test folds
            cfg_eval_rows: list[dict[str, Any]] = []
            for rrf_k, dw, bw in configs:
                val_metrics = eval_rows(val_q, meta_by_dataset, dw, bw, rrf_k, k_list)
                test_metrics = eval_rows(test_q, meta_by_dataset, dw, bw, rrf_k, k_list)
                row = {
                    "repeat": rep,
                    "fold": fold,
                    "rrf_k": rrf_k,
                    "dense_weight": dw,
                    "bm25_weight": bw,
                    "val_num_queries": int(val_metrics.get("num_queries", 0.0)),
                    "test_num_queries": int(test_metrics.get("num_queries", 0.0)),
                }
                for k, v in val_metrics.items():
                    if k != "num_queries":
                        row[f"val_{k}"] = v
                for k, v in test_metrics.items():
                    if k != "num_queries":
                        row[f"test_{k}"] = v
                cfg_eval_rows.append(row)
                test_rows_all_configs.append(row.copy())

            cfg_df = pd.DataFrame(cfg_eval_rows)
            sort_cols = []
            if "val_hit@1" in cfg_df.columns:
                sort_cols.append("val_hit@1")
            if "val_mrr@10" in cfg_df.columns:
                sort_cols.append("val_mrr@10")
            if "val_hit@3" in cfg_df.columns:
                sort_cols.append("val_hit@3")
            cfg_df = cfg_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)
            best = cfg_df.iloc[0].to_dict()
            selected_rows.append(best)
            split_rows.append(
                {
                    "repeat": rep,
                    "fold": fold,
                    "num_train": len(train_q),
                    "num_val": len(val_q),
                    "num_test": len(test_q),
                    "selected_rrf_k": int(best["rrf_k"]),
                    "selected_dense_weight": float(best["dense_weight"]),
                    "selected_bm25_weight": float(best["bm25_weight"]),
                    "selected_val_hit@1": float(best.get("val_hit@1", 0.0)),
                    "selected_val_mrr@10": float(best.get("val_mrr@10", 0.0)),
                    "selected_test_hit@1": float(best.get("test_hit@1", 0.0)),
                    "selected_test_mrr@10": float(best.get("test_mrr@10", 0.0)),
                }
            )

    if not selected_rows:
        raise RuntimeError("No valid CV splits were evaluated.")

    all_test_df = pd.DataFrame(test_rows_all_configs)
    agg = (
        all_test_df.groupby(["rrf_k", "dense_weight", "bm25_weight"], as_index=False)
        .agg(
            mean_test_hit_at_1=("test_hit@1", "mean"),
            std_test_hit_at_1=("test_hit@1", "std"),
            mean_test_mrr_at_10=("test_mrr@10", "mean"),
            std_test_mrr_at_10=("test_mrr@10", "std"),
            mean_test_hit_at_3=("test_hit@3", "mean"),
            std_test_hit_at_3=("test_hit@3", "std"),
            count_splits=("test_hit@1", "count"),
        )
    )
    agg["std_test_hit_at_1"] = agg["std_test_hit_at_1"].fillna(0.0)
    agg["std_test_mrr_at_10"] = agg["std_test_mrr_at_10"].fillna(0.0)
    agg["std_test_hit_at_3"] = agg["std_test_hit_at_3"].fillna(0.0)

    agg_sorted = agg.sort_values(
        [
            "mean_test_hit_at_1",
            "mean_test_mrr_at_10",
            "std_test_hit_at_1",
            "std_test_mrr_at_10",
        ],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    prod = agg_sorted.iloc[0].to_dict()

    sel_df = pd.DataFrame(selected_rows)
    sel_freq = (
        sel_df.groupby(["rrf_k", "dense_weight", "bm25_weight"], as_index=False)
        .size()
        .rename(columns={"size": "selected_count"})
        .sort_values("selected_count", ascending=False)
        .reset_index(drop=True)
    )

    split_df = pd.DataFrame(split_rows)
    nested_summary = {
        "num_splits": int(len(split_df)),
        "mean_selected_test_hit@1": float(split_df["selected_test_hit@1"].mean()),
        "std_selected_test_hit@1": float(split_df["selected_test_hit@1"].std(ddof=0)),
        "mean_selected_test_mrr@10": float(split_df["selected_test_mrr@10"].mean()),
        "std_selected_test_mrr@10": float(split_df["selected_test_mrr@10"].std(ddof=0)),
    }

    summary = {
        "run_root": str(run_root),
        "run_filter": args.run_filter,
        "model": str(args.model),
        "k_list": k_list,
        "n_folds": int(args.n_folds),
        "n_repeats": int(args.n_repeats),
        "num_queries_total": int(len(packs)),
        "num_datasets": int(len(dataset_rows)),
        "num_configs": int(len(configs)),
        "recommended_production_config": {
            "rrf_k": int(prod["rrf_k"]),
            "dense_weight": float(prod["dense_weight"]),
            "bm25_weight": float(prod["bm25_weight"]),
            "dense_to_bm25_ratio": float(prod["dense_weight"] / prod["bm25_weight"])
            if float(prod["bm25_weight"]) != 0.0
            else None,
            "mean_test_hit@1": float(prod["mean_test_hit_at_1"]),
            "std_test_hit@1": float(prod["std_test_hit_at_1"]),
            "mean_test_mrr@10": float(prod["mean_test_mrr_at_10"]),
            "std_test_mrr@10": float(prod["std_test_mrr_at_10"]),
            "count_splits": int(prod["count_splits"]),
        },
        "nested_selection_policy_summary": nested_summary,
    }

    datasets_csv = out_dir / "hybrid_weight_cv_datasets.csv"
    all_configs_csv = out_dir / "hybrid_weight_cv_all_configs_by_split.csv"
    selected_csv = out_dir / "hybrid_weight_cv_selected_per_split.csv"
    selected_freq_csv = out_dir / "hybrid_weight_cv_selected_frequency.csv"
    agg_csv = out_dir / "hybrid_weight_cv_config_aggregate.csv"
    summary_json = out_dir / "hybrid_weight_cv_summary.json"

    pd.DataFrame(dataset_rows).to_csv(datasets_csv, index=False)
    all_test_df.to_csv(all_configs_csv, index=False)
    split_df.to_csv(selected_csv, index=False)
    sel_freq.to_csv(selected_freq_csv, index=False)
    agg_sorted.to_csv(agg_csv, index=False)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Saved: {datasets_csv}")
    print(f"Saved: {all_configs_csv}")
    print(f"Saved: {selected_csv}")
    print(f"Saved: {selected_freq_csv}")
    print(f"Saved: {agg_csv}")
    print(f"Saved: {summary_json}")
    print(
        "Recommended production config: "
        f"rrf_k={int(prod['rrf_k'])}, dense_weight={float(prod['dense_weight'])}, bm25_weight={float(prod['bm25_weight'])}, "
        f"mean_test_hit@1={float(prod['mean_test_hit_at_1']):.4f}, mean_test_mrr@10={float(prod['mean_test_mrr_at_10']):.4f}"
    )


if __name__ == "__main__":
    main()
