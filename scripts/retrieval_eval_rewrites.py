"""
retrieval_eval_rewrites.py

Evaluate top-k retrieval with query rewriting.

This script extends retrieval_eval.py by allowing each query to have multiple
rewrite variants. For each query_id, it retrieves candidates for:
- the original question
- each rewrite variant

It then merges retrieval results by:
- keeping the maximum similarity score per chunk across variants
- sorting chunks by that max score (descending)
- computing page-level metrics on the merged top-k list

Inputs (in DATA_DIR):
- faiss.index
- chunk_meta.parquet
- eval_set_rewrites.json

Outputs (written to DATA_DIR):
- retrieval_results_rewrites.json
- retrieval_metrics_rewrites.json
- retrieval_summary_rewrites.csv

How to run
python retrieval_eval_rewrites.py

eval_set_rewrites.json schema (example)
[
  {
    "query_id": "Q001",
    "question": "What is the reporting period end date?",
    "rewrites": [
      "For the period ended, what is the date?",
      "What date is shown after the phrase 'For the period ended'?"
    ],
    "expected_pages": [1],
    "answer_type": "date"
  }
]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

try:
    import faiss
except Exception as e:
    raise RuntimeError(
        "FAISS is not installed.\n"
        "Fix:\n"
        "  pip install faiss-cpu\n"
    ) from e

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise RuntimeError(
        "sentence-transformers is not installed.\n"
        "Fix:\n"
        "  pip install sentence-transformers\n"
    ) from e


# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = Path(
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed/"
    "nhs-england-annual-report-and-accounts-2023-to-2024"
)

INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "chunk_meta.parquet"

# You create this file (similar to eval_set.json but with a "rewrites" list)
EVAL_SET_PATH = DATA_DIR / "eval_set_rewrites.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Evaluate multiple k values in one run
K_LIST = [1, 3, 5, 10]

# Outputs
RESULTS_JSON = DATA_DIR / "retrieval_results_rewrites.json"
METRICS_JSON = DATA_DIR / "retrieval_metrics_rewrites.json"
SUMMARY_CSV = DATA_DIR / "retrieval_summary_rewrites.csv"

# Retrieval depth per variant (we need enough candidates so merging helps)
# Use the largest k you evaluate, but you can set a higher number if desired.
MAX_K_PER_VARIANT = 10


# =============================================================================
# HELPERS
# =============================================================================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _env_or_default(name: str, default: str) -> str:
    """Return environment value when present, otherwise the provided default."""
    val = os.getenv(name)
    return val if val else default


def parse_k_list(val: str) -> list[int]:
    """Parse comma-separated k list into integers."""
    parts = [p.strip() for p in val.split(",") if p.strip()]
    return [int(p) for p in parts]


def refresh_paths() -> None:
    """Refresh file paths derived from current DATA_DIR."""
    global INDEX_PATH, META_PATH, EVAL_SET_PATH, RESULTS_JSON, METRICS_JSON, SUMMARY_CSV
    INDEX_PATH = DATA_DIR / "faiss.index"
    META_PATH = DATA_DIR / "chunk_meta.parquet"
    EVAL_SET_PATH = DATA_DIR / "eval_set_rewrites.json"
    RESULTS_JSON = DATA_DIR / "retrieval_results_rewrites.json"
    METRICS_JSON = DATA_DIR / "retrieval_metrics_rewrites.json"
    SUMMARY_CSV = DATA_DIR / "retrieval_summary_rewrites.csv"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for rewrite-based retrieval evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval with deterministic query rewrites."
    )
    parser.add_argument(
        "--data-dir",
        default=_env_or_default("DATA_DIR", str(DATA_DIR)),
        help="Directory containing faiss.index, chunk_meta.parquet, eval_set_rewrites.json.",
    )
    parser.add_argument(
        "--model",
        default=_env_or_default("EMBED_MODEL_NAME", EMBED_MODEL_NAME),
        help="Sentence-transformers model name or local path.",
    )
    parser.add_argument(
        "--k-list",
        default=_env_or_default("K_LIST", ",".join(str(k) for k in K_LIST)),
        help="Comma-separated list of k values (e.g. 1,3,5,10).",
    )
    parser.add_argument(
        "--max-k-per-variant",
        type=int,
        default=int(_env_or_default("MAX_K_PER_VARIANT", str(MAX_K_PER_VARIANT))),
        help="Top candidates retrieved per variant before merge.",
    )
    return parser.parse_args()


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def to_int_list(v) -> list[int]:
    """
    Normalise a page_list-like value into list[int].
    Handles list, tuple, numpy arrays, scalars, and stringified lists.
    """
    if v is None:
        return []

    if isinstance(v, float) and pd.isna(v):
        return []

    if isinstance(v, (list, tuple)):
        out = []
        for x in v:
            if x is None:
                continue
            if isinstance(x, float) and pd.isna(x):
                continue
            out.append(int(x))
        return out

    if hasattr(v, "tolist"):
        try:
            vv = v.tolist()
            if isinstance(vv, list):
                return [int(x) for x in vv if x is not None]
            return [int(vv)]
        except Exception:
            pass

    if isinstance(v, str):
        s = v.strip()
        if s.startswith("[") and s.endswith("]"):
            nums = re.findall(r"\d+", s)
            return [int(n) for n in nums]
        nums = re.findall(r"\d+", s)
        if len(nums) == 1:
            return [int(nums[0])]
        return []

    try:
        return [int(v)]
    except Exception:
        return []


def get_retrieved_pages(meta_row: pd.Series) -> list[int]:
    """
    Extract pages from meta row.
    Prefer page_list if present, otherwise fall back to page_start/page_end.
    """
    if "page_list" in meta_row.index:
        pl = to_int_list(meta_row["page_list"])
        if pl:
            return pl

    ps = meta_row.get("page_start", None)
    pe = meta_row.get("page_end", None)
    if ps is None or (isinstance(ps, float) and pd.isna(ps)):
        return []
    try:
        ps_i = int(ps)
        pe_i = int(pe) if pe is not None and not (isinstance(pe, float) and pd.isna(pe)) else ps_i
        if pe_i < ps_i:
            pe_i = ps_i
        return list(range(ps_i, pe_i + 1))
    except Exception:
        return []


def recall_at_k(expected_pages: set[int], retrieved_pages: list[int]) -> float:
    if not expected_pages:
        return 0.0
    return 1.0 if expected_pages.intersection(set(retrieved_pages)) else 0.0


def precision_at_k(expected_pages: set[int], retrieved_pages: list[int]) -> float:
    if not expected_pages:
        return 0.0
    if not retrieved_pages:
        return 0.0
    hits = sum(1 for p in retrieved_pages if p in expected_pages)
    return hits / len(retrieved_pages)


def mrr_for_pages(expected_pages: set[int], ranked_pages: list[int]) -> float:
    if not expected_pages:
        return 0.0
    for i, p in enumerate(ranked_pages, start=1):
        if p in expected_pages:
            return 1.0 / i
    return 0.0


def dedupe_preserve_order(xs: list[int]) -> list[int]:
    seen = set()
    out = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def normalise_query_variants(question: str, rewrites: Any) -> list[str]:
    """
    Returns a list of query variants including the original question.
    Removes empties and duplicates.
    """
    variants = [str(question).strip()]

    if isinstance(rewrites, list):
        for r in rewrites:
            rr = str(r).strip()
            if rr:
                variants.append(rr)

    # Deduplicate while preserving order
    seen = set()
    out = []
    for v in variants:
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)
    return out


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()
    global DATA_DIR, EMBED_MODEL_NAME, K_LIST, MAX_K_PER_VARIANT
    DATA_DIR = Path(args.data_dir)
    EMBED_MODEL_NAME = args.model
    K_LIST = parse_k_list(args.k_list)
    MAX_K_PER_VARIANT = int(args.max_k_per_variant)
    refresh_paths()

    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing FAISS index: {INDEX_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing chunk metadata: {META_PATH}")
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(
            f"Missing eval_set_rewrites.json: {EVAL_SET_PATH}\n"
            "Create this file with fixed questions, rewrites, and expected_pages."
        )

    meta = pd.read_parquet(META_PATH)
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading embedding model:", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    eval_items = read_json(EVAL_SET_PATH)
    if not isinstance(eval_items, list) or len(eval_items) == 0:
        raise ValueError("eval_set_rewrites.json must be a non-empty list of query objects.")

    # Validate K list against index size
    max_k_eval = min(max(K_LIST), len(meta))
    k_list = [k for k in K_LIST if 1 <= k <= max_k_eval]
    if not k_list:
        raise ValueError("K_LIST has no valid values for the current index size.")

    # Retrieval depth per variant must be >= max k being evaluated
    max_k_per_variant = max(MAX_K_PER_VARIANT, max(k_list))
    max_k_per_variant = min(max_k_per_variant, len(meta))

    run_info = {
        "run_utc": utc_now_iso(),
        "data_dir": str(DATA_DIR),
        "index_path": str(INDEX_PATH),
        "meta_path": str(META_PATH),
        "eval_set_path": str(EVAL_SET_PATH),
        "embedding_model": EMBED_MODEL_NAME,
        "k_list": k_list,
        "max_k_per_variant": int(max_k_per_variant),
        "num_queries": len(eval_items),
        "num_chunks_indexed": int(len(meta)),
        "merge_strategy": "max_score_per_chunk_across_variants",
    }

    results = []
    summary_rows = []

    # Build a flat list of all variants to embed in one batch
    variant_map = []  # list of (query_idx, variant_text)
    for qi, item in enumerate(eval_items):
        q = str(item.get("question", "")).strip()
        if not q:
            raise ValueError(f"Empty question at eval_items index {qi}")
        variants = normalise_query_variants(q, item.get("rewrites", []))
        for v in variants:
            variant_map.append((qi, v))

    variant_texts = [v for _, v in variant_map]

    # Embed all variants
    v_emb = model.encode(
        variant_texts,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    ).astype("float32")
    v_emb = l2_normalize(v_emb).astype("float32")

    # For each query, collect retrieval from each variant and merge
    offset = 0
    for qi, item in enumerate(eval_items):
        query_id = str(item.get("query_id", f"Q{qi+1:03d}"))
        question = str(item.get("question", "")).strip()
        answer_type = str(item.get("answer_type", "unknown"))

        expected = item.get("expected_pages", [])
        expected_pages = set(int(x) for x in expected) if isinstance(expected, list) else set()

        variants = normalise_query_variants(question, item.get("rewrites", []))
        num_vars = len(variants)

        # Accumulate best score per chunk across variants
        best_score_by_idx: dict[int, float] = {}
        best_variant_by_idx: dict[int, str] = {}

        for vi in range(num_vars):
            qvec = v_emb[offset + vi : offset + vi + 1]
            scores, idxs = index.search(qvec, max_k_per_variant)

            idxs_list = idxs[0].tolist()
            scores_list = scores[0].tolist()

            for ix, sc in zip(idxs_list, scores_list):
                ix_i = int(ix)
                sc_f = float(sc)
                prev = best_score_by_idx.get(ix_i, None)
                if prev is None or sc_f > prev:
                    best_score_by_idx[ix_i] = sc_f
                    best_variant_by_idx[ix_i] = variants[vi]

        offset += num_vars

        # Sort merged candidates by best score descending
        merged_pairs = sorted(best_score_by_idx.items(), key=lambda x: x[1], reverse=True)
        merged_idxs = [ix for ix, _ in merged_pairs]
        merged_scores = [sc for _, sc in merged_pairs]

        per_k = {}
        for k in k_list:
            top_idxs = merged_idxs[:k]
            top_scores = merged_scores[:k]

            retrieved_chunks = meta.iloc[top_idxs].copy()
            retrieved_chunks["score"] = top_scores
            retrieved_chunks["best_variant"] = [best_variant_by_idx.get(int(ix), question) for ix in top_idxs]

            ranked_pages = []
            for _, r in retrieved_chunks.iterrows():
                ranked_pages.extend(get_retrieved_pages(r))

            ranked_pages_unique = dedupe_preserve_order(ranked_pages)

            r_at_k = recall_at_k(expected_pages, ranked_pages_unique)
            p_at_k = precision_at_k(expected_pages, ranked_pages_unique)
            mrr = mrr_for_pages(expected_pages, ranked_pages_unique)

            per_k[str(k)] = {
                "query_variants": variants,
                "retrieved_chunk_ids": retrieved_chunks["chunk_id"].astype(str).tolist() if "chunk_id" in retrieved_chunks.columns else [],
                "retrieved_pages_ranked": ranked_pages_unique,
                "retrieved_scores": [float(s) for s in top_scores],
                "best_variant_per_chunk": retrieved_chunks["best_variant"].astype(str).tolist(),
                "recall_at_k": float(r_at_k),
                "precision_at_k": float(p_at_k),
                "mrr_at_k": float(mrr),
            }

            summary_rows.append(
                {
                    "query_id": query_id,
                    "k": k,
                    "answer_type": answer_type,
                    "num_variants": len(variants),
                    "expected_pages": sorted(list(expected_pages)),
                    "recall_at_k": r_at_k,
                    "precision_at_k": p_at_k,
                    "mrr_at_k": mrr,
                    "top_pages": per_k[str(k)]["retrieved_pages_ranked"][:10],
                }
            )

        results.append(
            {
                "query_id": query_id,
                "question": question,
                "answer_type": answer_type,
                "expected_pages": sorted(list(expected_pages)),
                "query_variants": variants,
                "per_k": per_k,
            }
        )

    # Aggregate metrics per k
    df_sum = pd.DataFrame(summary_rows)
    metrics = {"run_info": run_info, "metrics_by_k": {}}

    for k in k_list:
        dfk = df_sum[df_sum["k"] == k]
        metrics["metrics_by_k"][str(k)] = {
            "num_queries": int(len(dfk)),
            "mean_recall_at_k": float(dfk["recall_at_k"].mean()) if len(dfk) else 0.0,
            "mean_precision_at_k": float(dfk["precision_at_k"].mean()) if len(dfk) else 0.0,
            "mean_mrr_at_k": float(dfk["mrr_at_k"].mean()) if len(dfk) else 0.0,
            "hit_rate_at_k": float((dfk["recall_at_k"] > 0).mean()) if len(dfk) else 0.0,
            "mean_num_variants": float(dfk["num_variants"].mean()) if len(dfk) else 0.0,
        }

    # Save outputs
    write_json(RESULTS_JSON, {"run_info": run_info, "results": results})
    write_json(METRICS_JSON, metrics)
    df_sum.to_csv(SUMMARY_CSV, index=False)

    print("Saved:", RESULTS_JSON)
    print("Saved:", METRICS_JSON)
    print("Saved:", SUMMARY_CSV)

    # Console summary
    for k in k_list:
        m = metrics["metrics_by_k"][str(k)]
        print(
            f"k={k}  "
            f"hit_rate={m['hit_rate_at_k']:.3f}  "
            f"mean_recall={m['mean_recall_at_k']:.3f}  "
            f"mean_precision={m['mean_precision_at_k']:.3f}  "
            f"mean_mrr={m['mean_mrr_at_k']:.3f}  "
            f"mean_variants={m['mean_num_variants']:.2f}"
        )


if __name__ == "__main__":
    main()
