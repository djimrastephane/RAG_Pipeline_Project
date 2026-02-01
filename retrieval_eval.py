"""
retrieval_eval.py

Evaluate top-k retrieval using:
- faiss.index
- chunk_meta.parquet
- eval_set.json (fixed questions with expected_pages)

Outputs (written to the same DATA_DIR):
- retrieval_results.json      Per-query retrieved results for each k
- retrieval_metrics.json      Aggregate metrics for each k
- retrieval_summary.csv       Flat table for appendix and quick inspection

How to run
1) Activate your venv
2) Ensure dependencies installed:
   pip install faiss-cpu sentence-transformers pandas pyarrow numpy
3) Create eval_set.json (example below)
4) Run:
   python retrieval_eval.py

eval_set.json example
[
  {
    "query_id": "Q001",
    "question": "What is the reporting period end date?",
    "expected_pages": [1],
    "answer_type": "date"
  },
  {
    "query_id": "Q002",
    "question": "What is the total staff costs figure?",
    "expected_pages": [120, 121],
    "answer_type": "number"
  }
]
"""

from __future__ import annotations

import json
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
    "DECC Document Template - Standard Numbering"
)

INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "chunk_meta.parquet"
EVAL_SET_PATH = DATA_DIR / "eval_set.json"  # place your eval set here

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Evaluate multiple k values in one run
K_LIST = [1, 3, 5, 10]

# Save outputs
RESULTS_JSON = DATA_DIR / "retrieval_results.json"
METRICS_JSON = DATA_DIR / "retrieval_metrics.json"
SUMMARY_CSV = DATA_DIR / "retrieval_summary.csv"


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

    # fallback: use page_start..page_end if available
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
    """
    Query-level recall@k for pages:
    1 if any expected page is retrieved, else 0.
    """
    if not expected_pages:
        return 0.0
    return 1.0 if (expected_pages.intersection(set(retrieved_pages))) else 0.0


def precision_at_k(expected_pages: set[int], retrieved_pages: list[int]) -> float:
    """
    Query-level precision@k for pages:
    fraction of retrieved pages that are expected pages.
    """
    if not expected_pages:
        return 0.0
    if not retrieved_pages:
        return 0.0
    hits = sum(1 for p in retrieved_pages if p in expected_pages)
    return hits / len(retrieved_pages)


def mrr_for_pages(expected_pages: set[int], ranked_pages: list[int]) -> float:
    """
    Mean Reciprocal Rank (MRR) on page hits.
    If the first correct page appears at rank r (1-indexed), score = 1/r.
    """
    if not expected_pages:
        return 0.0
    for i, p in enumerate(ranked_pages, start=1):
        if p in expected_pages:
            return 1.0 / i
    return 0.0


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Missing FAISS index: {INDEX_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"Missing chunk metadata: {META_PATH}")
    if not EVAL_SET_PATH.exists():
        raise FileNotFoundError(
            f"Missing eval_set.json: {EVAL_SET_PATH}\n"
            "Create this file with fixed questions and expected_pages."
        )

    meta = pd.read_parquet(META_PATH)
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading embedding model:", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    eval_items = read_json(EVAL_SET_PATH)
    if not isinstance(eval_items, list) or len(eval_items) == 0:
        raise ValueError("eval_set.json must be a non-empty list of query objects.")

    # Validate K list against index size
    max_k = min(max(K_LIST), len(meta))
    k_list = [k for k in K_LIST if 1 <= k <= max_k]
    if not k_list:
        raise ValueError("K_LIST has no valid values for the current index size.")

    run_info = {
        "run_utc": utc_now_iso(),
        "data_dir": str(DATA_DIR),
        "index_path": str(INDEX_PATH),
        "meta_path": str(META_PATH),
        "eval_set_path": str(EVAL_SET_PATH),
        "embedding_model": EMBED_MODEL_NAME,
        "k_list": k_list,
        "num_queries": len(eval_items),
        "num_chunks_indexed": int(len(meta)),
    }

    results = []
    summary_rows = []

    # Pre-embed all questions (faster, reproducible)
    questions = [str(q.get("question", "")).strip() for q in eval_items]
    if any(len(q) == 0 for q in questions):
        bad = [i for i, q in enumerate(questions) if len(q) == 0]
        raise ValueError(f"Some eval items have empty 'question' fields at indices: {bad}")

    q_emb = model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
    ).astype("float32")
    q_emb = l2_normalize(q_emb).astype("float32")

    # For each query, search up to max_k and then slice for each k
    for qi, item in enumerate(eval_items):
        query_id = str(item.get("query_id", f"Q{qi+1:03d}"))
        question = questions[qi]
        expected = item.get("expected_pages", [])
        expected_pages = set(int(x) for x in expected) if isinstance(expected, list) else set()
        answer_type = str(item.get("answer_type", "unknown"))

        scores, idxs = index.search(q_emb[qi : qi + 1], max_k)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        # Build per-k retrieval info
        per_k = {}
        for k in k_list:
            top_idxs = idxs[:k]
            top_scores = scores[:k]

            retrieved_chunks = meta.iloc[top_idxs].copy()
            retrieved_chunks["score"] = top_scores

            # Flatten to a ranked page list for scoring.
            # We keep order by chunk rank, and within each chunk keep page_list order.
            ranked_pages = []
            for _, r in retrieved_chunks.iterrows():
                pages = get_retrieved_pages(r)
                ranked_pages.extend(pages)

            # Deduplicate while preserving order (to avoid overcounting)
            seen = set()
            ranked_pages_unique = []
            for p in ranked_pages:
                if p in seen:
                    continue
                seen.add(p)
                ranked_pages_unique.append(p)

            r_at_k = recall_at_k(expected_pages, ranked_pages_unique)
            p_at_k = precision_at_k(expected_pages, ranked_pages_unique)
            mrr = mrr_for_pages(expected_pages, ranked_pages_unique)

            per_k[str(k)] = {
                "retrieved_chunk_ids": retrieved_chunks["chunk_id"].astype(str).tolist() if "chunk_id" in retrieved_chunks.columns else [],
                "retrieved_pages_ranked": ranked_pages_unique,
                "retrieved_scores": [float(s) for s in top_scores],
                "recall_at_k": float(r_at_k),
                "precision_at_k": float(p_at_k),
                "mrr_at_k": float(mrr),
            }

            summary_rows.append(
                {
                    "query_id": query_id,
                    "k": k,
                    "answer_type": answer_type,
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
        }

    # Save outputs
    write_json(RESULTS_JSON, {"run_info": run_info, "results": results})
    write_json(METRICS_JSON, metrics)
    df_sum.to_csv(SUMMARY_CSV, index=False)

    print("Saved:", RESULTS_JSON)
    print("Saved:", METRICS_JSON)
    print("Saved:", SUMMARY_CSV)

    # Print a short console summary
    for k in k_list:
        m = metrics["metrics_by_k"][str(k)]
        print(
            f"k={k}  "
            f"hit_rate={m['hit_rate_at_k']:.3f}  "
            f"mean_recall={m['mean_recall_at_k']:.3f}  "
            f"mean_precision={m['mean_precision_at_k']:.3f}  "
            f"mean_mrr={m['mean_mrr_at_k']:.3f}"
        )


if __name__ == "__main__":
    main()