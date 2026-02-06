"""
retrieval_eval.py

Evaluate top-k retrieval using:
- faiss.index
- chunk_meta.parquet
- eval_set.json (fixed questions with expected_pages, optional doc_id/document_id, expected_section)

Outputs (written to the same DATA_DIR):
- retrieval_results.json      Per-query retrieved results for each k
- retrieval_metrics.json      Aggregate metrics for each k
- retrieval_summary.csv       Flat table for appendix and quick inspection

Notes
- Enforces query_id nomenclature: Q_<TOPIC>_<YEAR>_<NN>
  Allowed TOPIC values: REV, EFF, DEF, STAFF
  Examples:
    Q_REV_2023_01
    Q_EFF_2023_01
    Q_DEF_2023_01
    Q_STAFF_2023_02

- Reports both page-level metrics and chunk-level metrics.
  Page precision can drop as k increases because extra chunks add extra pages.
  Chunk metrics are often easier to interpret for retrieval ranking.

FAILURE ATTRIBUTION (RETRIEVAL STAGE ONLY)
This evaluator assigns a deterministic retrieval failure stage per query at each k:
- missing_content
- missed_top_ranked
- hit

NEW: LEAKAGE DETECTION (MULTI-DOC SAFETY)
If a query specifies an expected doc_id, this evaluator reports whether any of the
top-k retrieved chunks come from a different doc_id.

Leakage does not change recall@k. It is reported as an additional diagnostic
signal, useful for the 3-document supervisor requirement:
- Each evaluation question answerable from exactly one document.
- Detect when retrieval pulls strongly similar chunks from other documents.

Implementation details:
- leakage_count_top_k: number of retrieved chunks in top-k with doc_id != expected_doc_id
- leakage_doc_ids_top_k: unique list of non-expected doc_ids in top-k
- retrieved_doc_ids_top_k: doc_id for each retrieved chunk in top-k
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

try:
    import pyarrow.parquet as pq
except Exception as e:
    raise RuntimeError(
        "pyarrow is not installed.\n"
        "Fix:\n"
        "  pip install pyarrow\n"
    ) from e


# =============================================================================
# CONFIG
# =============================================================================
DATA_DIR = Path(
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed/"
    "Grampian-2022-2023"
)

INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "chunk_meta.parquet"
EVAL_SET_PATH = DATA_DIR / "eval_set.json"

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

K_LIST = [1, 3, 5, 10]

RESULTS_JSON = DATA_DIR / "retrieval_results.json"
METRICS_JSON = DATA_DIR / "retrieval_metrics.json"
SUMMARY_CSV = DATA_DIR / "retrieval_summary.csv"

PRINT_HIT_DEBUG = True


# =============================================================================
# QUERY ID NOMENCLATURE
# =============================================================================
QUERY_ID_PATTERN = re.compile(r"^Q_(REV|EFF|DEF|STAFF)_\d{4}_\d{2}$")


def validate_query_id(query_id: str) -> None:
    if not QUERY_ID_PATTERN.match(query_id):
        raise ValueError(
            f"Invalid query_id '{query_id}'. Expected: Q_<TOPIC>_<YEAR>_<NN> "
            f"with TOPIC in [REV, EFF, DEF, STAFF], for example Q_EFF_2023_01."
        )


def parse_query_id(query_id: str) -> dict[str, Any]:
    _, topic, year, seq = query_id.split("_")
    return {"topic": topic, "year": int(year), "sequence": int(seq)}


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


def read_parquet_safe(path: Path) -> pd.DataFrame:
    """
    Read parquet via pyarrow to avoid pandas engine issues on some Python builds.
    """
    return pq.read_table(str(path)).to_pandas()


def to_int_list(v) -> list[int]:
    """
    Normalise a page_list-like value into list[int].

    Handles:
    - list / tuple
    - numpy arrays
    - scalars
    - stringified lists
    - list of dicts like [{"element": 2}]
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
            if isinstance(x, dict) and "element" in x:
                nums = re.findall(r"\d+", str(x.get("element")))
                if nums:
                    out.append(int(nums[0]))
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
        nums = re.findall(r"\d+", s)
        return [int(n) for n in nums] if nums else []

    try:
        return [int(v)]
    except Exception:
        return []


def get_retrieved_pages(meta_row: pd.Series) -> list[int]:
    """
    Extract pages from meta row.

    Preference order:
    1) pages (plain list[int]) if present
    2) page_list if present (may be list, list-of-dicts, or stringified)
    3) page_start/page_end span
    """
    if "pages" in meta_row.index:
        pl = to_int_list(meta_row["pages"])
        if pl:
            return pl

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


def unique_preserve_order(items: list[int]) -> list[int]:
    seen = set()
    out = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def get_expected_doc_id(item: dict[str, Any]) -> str:
    """
    Support both keys to avoid breaking older eval sets.

    Preferred key is 'doc_id'. For backward compatibility, 'document_id' is also
    accepted.
    """
    v = str(item.get("doc_id", "")).strip()
    if v:
        return v
    return str(item.get("document_id", "")).strip()


def get_chunk_ids(retrieved_chunks: pd.DataFrame) -> list[str]:
    """
    Prefer chunk_id_global when present, else fall back to chunk_id.
    """
    if "chunk_id_global" in retrieved_chunks.columns:
        return retrieved_chunks["chunk_id_global"].astype(str).tolist()
    if "chunk_id" in retrieved_chunks.columns:
        return retrieved_chunks["chunk_id"].astype(str).tolist()
    return []


def get_doc_ids(retrieved_chunks: pd.DataFrame) -> list[str]:
    """
    Extract doc_id list for top-k retrieved chunks.
    """
    if "doc_id" not in retrieved_chunks.columns:
        return []
    return retrieved_chunks["doc_id"].astype(str).tolist()


def compute_leakage(expected_doc_id: str, retrieved_doc_ids: list[str]) -> dict[str, Any]:
    """
    Compute retrieval leakage statistics for a query.

    Leakage definition:
        Any retrieved chunk in top-k whose doc_id differs from the expected doc_id.

    Args:
        expected_doc_id: The query's expected document identifier.
        retrieved_doc_ids: doc_id for each retrieved chunk in rank order.

    Returns:
        dict with keys:
        - leakage_count_top_k (int)
        - leakage_doc_ids_top_k (list[str])
        - leakage_rate_top_k (float)
    """
    if not expected_doc_id or not retrieved_doc_ids:
        return {"leakage_count_top_k": 0, "leakage_doc_ids_top_k": [], "leakage_rate_top_k": 0.0}

    leakage_docs = [d for d in retrieved_doc_ids if d != expected_doc_id]
    leakage_count = len(leakage_docs)
    leakage_rate = leakage_count / max(1, len(retrieved_doc_ids))
    return {
        "leakage_count_top_k": int(leakage_count),
        "leakage_doc_ids_top_k": sorted(list(set(leakage_docs))),
        "leakage_rate_top_k": float(leakage_rate),
    }


# -------------------------
# Page-level scoring
# -------------------------
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


# -------------------------
# Chunk-level scoring (based on page overlap)
# -------------------------
def chunk_hit_flags(expected_pages: set[int], retrieved_chunks: pd.DataFrame) -> list[int]:
    flags = []
    for _, r in retrieved_chunks.iterrows():
        pages = get_retrieved_pages(r)
        flags.append(1 if expected_pages.intersection(set(pages)) else 0)
    return flags


def chunk_hit_at_k(flags: list[int]) -> float:
    return 1.0 if any(flags) else 0.0


def chunk_precision_at_k(flags: list[int]) -> float:
    if not flags:
        return 0.0
    return float(sum(flags)) / float(len(flags))


def chunk_mrr(flags: list[int]) -> float:
    for i, f in enumerate(flags, start=1):
        if f == 1:
            return 1.0 / i
    return 0.0


# =============================================================================
# FAILURE ATTRIBUTION (RETRIEVAL STAGE)
# =============================================================================
def compute_gold_presence(meta: pd.DataFrame, expected_doc_id: str, expected_pages: set[int]) -> dict[str, Any]:
    """
    Determine whether the expected (doc_id, expected_pages) content exists in the index.
    """
    if not expected_pages:
        return {"gold_exists": False, "gold_chunk_count": 0, "gold_pages_found": []}

    df = meta
    if expected_doc_id and "doc_id" in df.columns:
        df = df[df["doc_id"].astype(str) == expected_doc_id]

    if len(df) == 0:
        return {"gold_exists": False, "gold_chunk_count": 0, "gold_pages_found": []}

    pages_found: set[int] = set()
    gold_chunk_count = 0

    for _, r in df.iterrows():
        pages = get_retrieved_pages(r)
        if expected_pages.intersection(set(pages)):
            gold_chunk_count += 1
            pages_found.update(expected_pages.intersection(set(pages)))

    return {
        "gold_exists": bool(gold_chunk_count > 0),
        "gold_chunk_count": int(gold_chunk_count),
        "gold_pages_found": sorted(list(pages_found)),
    }


def attribute_retrieval_failure(page_recall: float, gold_exists: bool) -> str:
    if page_recall >= 1.0:
        return "hit"
    return "missed_top_ranked" if gold_exists else "missing_content"


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

    meta = read_parquet_safe(META_PATH)
    index = faiss.read_index(str(INDEX_PATH))

    print("Loading embedding model:", EMBED_MODEL_NAME)
    model = SentenceTransformer(EMBED_MODEL_NAME)

    eval_items = read_json(EVAL_SET_PATH)
    if not isinstance(eval_items, list) or len(eval_items) == 0:
        raise ValueError("eval_set.json must be a non-empty list of query objects.")

    for i, item in enumerate(eval_items):
        qid = str(item.get("query_id", "")).strip()
        if not qid:
            raise ValueError(f"Missing query_id for eval item at index {i}.")
        validate_query_id(qid)

    max_k = min(max(K_LIST), len(meta))
    k_list = [k for k in K_LIST if 1 <= k <= max_k]
    if not k_list:
        raise ValueError("K_LIST has no valid values for the current index size.")

    meta_doc_ids = set(meta["doc_id"].astype(str).unique()) if "doc_id" in meta.columns else set()

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
        "meta_doc_ids": sorted(list(meta_doc_ids))[:20] if meta_doc_ids else [],
        "query_id_nomenclature": "Q_<TOPIC>_<YEAR>_<NN> with TOPIC in [REV,EFF,DEF,STAFF]",
        "failure_attribution": {"stages": ["hit", "missed_top_ranked", "missing_content"], "scope": "retrieval_only"},
        "leakage_detection": {
            "enabled": True,
            "requires_expected_doc_id": True,
            "fields": ["retrieved_doc_ids_top_k", "leakage_count_top_k", "leakage_doc_ids_top_k", "leakage_rate_top_k"],
        },
    }

    results: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

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

    for qi, item in enumerate(eval_items):
        query_id = str(item.get("query_id", "")).strip()
        validate_query_id(query_id)
        qid_parts = parse_query_id(query_id)

        question = questions[qi]

        expected = item.get("expected_pages", [])
        expected_pages = set(int(x) for x in expected) if isinstance(expected, list) else set()

        answer_type = str(item.get("answer_type", "unknown"))
        expected_doc_id = get_expected_doc_id(item)
        expected_section = str(item.get("expected_section", "")).strip()

        if expected_doc_id and meta_doc_ids and expected_doc_id not in meta_doc_ids:
            raise ValueError(
                f"Query {query_id} expects doc_id={expected_doc_id}, "
                f"but DATA_DIR meta has doc_id values like: {sorted(list(meta_doc_ids))[:5]}"
            )

        gold_presence = compute_gold_presence(meta, expected_doc_id, expected_pages)

        scores, idxs = index.search(q_emb[qi : qi + 1], max_k)
        idxs = idxs[0].tolist()
        scores = scores[0].tolist()

        per_k: dict[str, Any] = {}
        for k in k_list:
            top_idxs = idxs[:k]
            top_scores = scores[:k]

            retrieved_chunks = meta.iloc[top_idxs].copy()
            retrieved_chunks["score"] = top_scores

            retrieved_chunk_ids = get_chunk_ids(retrieved_chunks)
            retrieved_doc_ids = get_doc_ids(retrieved_chunks)
            leakage = compute_leakage(expected_doc_id, retrieved_doc_ids)

            ranked_pages = []
            for _, r in retrieved_chunks.iterrows():
                ranked_pages.extend(get_retrieved_pages(r))
            ranked_pages_unique = unique_preserve_order(ranked_pages)

            page_recall = recall_at_k(expected_pages, ranked_pages_unique)
            page_precision = precision_at_k(expected_pages, ranked_pages_unique)
            page_mrr = mrr_for_pages(expected_pages, ranked_pages_unique)

            flags = chunk_hit_flags(expected_pages, retrieved_chunks)
            c_hit = chunk_hit_at_k(flags)
            c_prec = chunk_precision_at_k(flags)
            c_mrr = chunk_mrr(flags)

            failure_stage = attribute_retrieval_failure(
                page_recall=page_recall,
                gold_exists=bool(gold_presence.get("gold_exists", False)),
            )

            per_k[str(k)] = {
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "retrieved_doc_ids_top_k": retrieved_doc_ids,
                "retrieved_pages_ranked": ranked_pages_unique,
                "retrieved_scores": [float(s) for s in top_scores],
                "page_recall_at_k": float(page_recall),
                "page_precision_at_k": float(page_precision),
                "page_mrr_at_k": float(page_mrr),
                "chunk_hit_at_k": float(c_hit),
                "chunk_precision_at_k": float(c_prec),
                "chunk_mrr_at_k": float(c_mrr),
                "chunk_hit_flags": flags,
                "failure_stage": failure_stage,
                **leakage,
            }

            summary_rows.append(
                {
                    "query_id": query_id,
                    "topic": qid_parts["topic"],
                    "year": qid_parts["year"],
                    "sequence": qid_parts["sequence"],
                    "k": k,
                    "answer_type": answer_type,
                    "doc_id": expected_doc_id,
                    "expected_section": expected_section,
                    "expected_pages": sorted(list(expected_pages)),
                    "gold_exists": bool(gold_presence.get("gold_exists", False)),
                    "gold_chunk_count": int(gold_presence.get("gold_chunk_count", 0)),
                    "gold_pages_found": gold_presence.get("gold_pages_found", []),
                    "failure_stage": failure_stage,
                    "leakage_count_top_k": leakage["leakage_count_top_k"],
                    "leakage_rate_top_k": leakage["leakage_rate_top_k"],
                    "leakage_doc_ids_top_k": leakage["leakage_doc_ids_top_k"],
                    "page_recall_at_k": page_recall,
                    "page_precision_at_k": page_precision,
                    "page_mrr_at_k": page_mrr,
                    "chunk_hit_at_k": c_hit,
                    "chunk_precision_at_k": c_prec,
                    "chunk_mrr_at_k": c_mrr,
                    "top_pages": ranked_pages_unique[:10],
                    "top_chunk_ids": retrieved_chunk_ids[:5],
                    "top_doc_ids": retrieved_doc_ids[:5],
                }
            )

            if PRINT_HIT_DEBUG and k == 1 and page_recall == 1.0:
                top_pages_preview = ranked_pages_unique[:10]
                top_chunk_preview = retrieved_chunk_ids[:3]
                print(f"HIT@1 query_id={query_id} pages={top_pages_preview} chunks={top_chunk_preview}")

        results.append(
            {
                "query_id": query_id,
                "topic": qid_parts["topic"],
                "year": qid_parts["year"],
                "sequence": qid_parts["sequence"],
                "question": question,
                "answer_type": answer_type,
                "doc_id": expected_doc_id,
                "expected_section": expected_section,
                "expected_pages": sorted(list(expected_pages)),
                "gold_presence": gold_presence,
                "per_k": per_k,
            }
        )

    df_sum = pd.DataFrame(summary_rows)

    metrics: dict[str, Any] = {"run_info": run_info, "metrics_by_k": {}, "failure_counts_by_k": {}, "leakage_counts_by_k": {}}

    for k in k_list:
        dfk = df_sum[df_sum["k"] == k]

        metrics["metrics_by_k"][str(k)] = {
            "num_queries": int(len(dfk)),
            "page_hit_rate_at_k": float((dfk["page_recall_at_k"] > 0).mean()) if len(dfk) else 0.0,
            "mean_page_recall_at_k": float(dfk["page_recall_at_k"].mean()) if len(dfk) else 0.0,
            "mean_page_precision_at_k": float(dfk["page_precision_at_k"].mean()) if len(dfk) else 0.0,
            "mean_page_mrr_at_k": float(dfk["page_mrr_at_k"].mean()) if len(dfk) else 0.0,
            "chunk_hit_rate_at_k": float((dfk["chunk_hit_at_k"] > 0).mean()) if len(dfk) else 0.0,
            "mean_chunk_precision_at_k": float(dfk["chunk_precision_at_k"].mean()) if len(dfk) else 0.0,
            "mean_chunk_mrr_at_k": float(dfk["chunk_mrr_at_k"].mean()) if len(dfk) else 0.0,
        }

        metrics["failure_counts_by_k"][str(k)] = (
            dfk["failure_stage"].value_counts(dropna=False).to_dict() if len(dfk) else {}
        )

        metrics["leakage_counts_by_k"][str(k)] = {
            "num_queries": int(len(dfk)),
            "any_leakage_rate_at_k": float((dfk["leakage_count_top_k"] > 0).mean()) if len(dfk) else 0.0,
            "mean_leakage_rate_at_k": float(dfk["leakage_rate_top_k"].mean()) if len(dfk) else 0.0,
        }

    write_json(RESULTS_JSON, {"run_info": run_info, "results": results})
    write_json(METRICS_JSON, metrics)
    df_sum.to_csv(SUMMARY_CSV, index=False)

    print("Saved:", RESULTS_JSON)
    print("Saved:", METRICS_JSON)
    print("Saved:", SUMMARY_CSV)

    for k in k_list:
        m = metrics["metrics_by_k"][str(k)]
        fc = metrics["failure_counts_by_k"].get(str(k), {})
        lc = metrics["leakage_counts_by_k"].get(str(k), {})
        print(
            f"k={k}  "
            f"page_hit_rate={m['page_hit_rate_at_k']:.3f}  "
            f"page_mrr={m['mean_page_mrr_at_k']:.3f}  "
            f"page_precision={m['mean_page_precision_at_k']:.3f}  "
            f"chunk_hit_rate={m['chunk_hit_rate_at_k']:.3f}  "
            f"chunk_mrr={m['mean_chunk_mrr_at_k']:.3f}  "
            f"chunk_precision={m['mean_chunk_precision_at_k']:.3f}  "
            f"failures={fc}  "
            f"any_leakage_rate={lc.get('any_leakage_rate_at_k', 0.0):.3f}  "
            f"mean_leakage_rate={lc.get('mean_leakage_rate_at_k', 0.0):.3f}"
        )


if __name__ == "__main__":
    main()