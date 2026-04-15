"""
retrieval_eval_bm25.py

Evaluate top-k retrieval with a pure BM25 baseline over chunk text.

Outputs (written to DATA_DIR):
- retrieval_results_bm25.json
- retrieval_metrics_bm25.json
- retrieval_summary_bm25.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from runtime_env import collect_runtime_provenance, critical_environment_checks


DATA_DIR = Path("data_processed/Grampian-2024-2025")
K_LIST = [1, 3, 5, 10]

RESULTS_JSON = DATA_DIR / "retrieval_results_bm25.json"
METRICS_JSON = DATA_DIR / "retrieval_metrics_bm25.json"
SUMMARY_CSV = DATA_DIR / "retrieval_summary_bm25.csv"

QUERY_ID_PATTERN_V1 = re.compile(r"^Q_(REV|EFF|DEF|STAFF|ACC|GOV|TABLE)_\d{4}_\d{2}$")
QUERY_ID_PATTERN_V2 = re.compile(r"^Q_(\d{4})_([A-Z]+)_(\d{2}|P\d+)$")
BM25_TOKENIZER_VARIANTS = ("default", "no_hyphen")
_BM25_TOKENIZER_VARIANT = "default"


def _env_or_default(name: str, default: str) -> str:
    import os

    val = os.getenv(name)
    return val if val else default


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_k_list(val: str) -> list[int]:
    parts = [p.strip() for p in val.split(",") if p.strip()]
    out = [int(p) for p in parts]
    if not out:
        raise ValueError("k-list must contain at least one integer")
    if min(out) <= 0:
        raise ValueError("k values must be > 0")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval using BM25 over chunk text and eval_set.json."
    )
    parser.add_argument(
        "--data-dir",
        default=_env_or_default("DATA_DIR", str(DATA_DIR)),
        help="Directory containing chunk_meta.parquet, chunks.parquet, eval_set.json.",
    )
    parser.add_argument(
        "--k-list",
        default=_env_or_default("K_LIST", ",".join(str(k) for k in K_LIST)),
        help="Comma-separated list of k values (e.g. 1,3,5,10).",
    )
    parser.add_argument(
        "--k1",
        type=float,
        default=float(_env_or_default("BM25_K1", "1.5")),
        help="BM25 k1 parameter.",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=float(_env_or_default("BM25_B", "0.75")),
        help="BM25 b parameter.",
    )
    parser.add_argument(
        "--bm25-tokenizer",
        choices=BM25_TOKENIZER_VARIANTS,
        default=_env_or_default("BM25_TOKENIZER", "default"),
        help="Lexical tokenizer variant for BM25 sensitivity checks.",
    )
    return parser.parse_args()


def validate_query_id(query_id: str) -> None:
    if QUERY_ID_PATTERN_V1.match(query_id) or QUERY_ID_PATTERN_V2.match(query_id):
        return
    if not (query_id.startswith("Q_") and len(query_id) >= 6):
        raise ValueError(
            f"Invalid query_id '{query_id}'. Expected Q_<TOPIC>_<YEAR>_<NN>."
        )


def parse_query_id(query_id: str) -> dict[str, Any]:
    parts = query_id.split("_")
    if len(parts) >= 4 and parts[1].isdigit():
        year = int(parts[1])
        topic = parts[2]
        seq_raw = parts[3]
        if seq_raw.isdigit():
            seq: Any = int(seq_raw)
        else:
            seq = seq_raw
        return {"topic": topic, "year": year, "sequence": seq}
    _, topic, year, seq = parts[:4]
    if str(seq).isdigit():
        seq_val: Any = int(seq)
    else:
        seq_val = seq
    return {"topic": topic, "year": int(year), "sequence": seq_val}


def to_int_list(v: Any) -> list[int]:
    if v is None:
        return []
    if isinstance(v, float) and pd.isna(v):
        return []
    if isinstance(v, (list, tuple)):
        out: list[int] = []
        for x in v:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                continue
            if isinstance(x, dict) and "element" in x:
                nums = re.findall(r"\d+", str(x.get("element")))
                if nums:
                    out.append(int(nums[0]))
                continue
            try:
                out.append(int(x))
            except Exception:
                continue
        return out
    if hasattr(v, "tolist"):
        try:
            vv = v.tolist()
            if isinstance(vv, list):
                return [int(x) for x in vv if x is not None]
            return [int(vv)]
        except Exception:
            pass
    s = str(v).strip()
    if not s:
        return []
    nums = re.findall(r"\d+", s)
    return [int(x) for x in nums]


def unique_preserve_order(items: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def get_expected_doc_id(item: dict[str, Any]) -> str:
    for k in ("doc_id", "document_id", "expected_doc_id"):
        v = item.get(k)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def get_retrieved_pages(row: pd.Series) -> list[int]:
    pages = to_int_list(row.get("pages"))
    if pages:
        return pages
    start = row.get("page_start")
    end = row.get("page_end")
    out: list[int] = []
    try:
        if start is not None and not (isinstance(start, float) and pd.isna(start)):
            out.append(int(start))
        if end is not None and not (isinstance(end, float) and pd.isna(end)):
            end_i = int(end)
            if end_i not in out:
                out.append(end_i)
    except Exception:
        pass
    return out


def get_chunk_ids(df: pd.DataFrame) -> list[str]:
    if "chunk_id_global" in df.columns:
        return [str(x) for x in df["chunk_id_global"].tolist()]
    if "chunk_id" in df.columns:
        return [str(x) for x in df["chunk_id"].tolist()]
    return [str(i) for i in df.index.tolist()]


def get_doc_ids(df: pd.DataFrame) -> list[str]:
    if "doc_id" in df.columns:
        return [str(x) for x in df["doc_id"].tolist()]
    return ["" for _ in range(len(df))]


def recall_at_k(expected_pages: set[int], retrieved_pages: list[int]) -> float:
    if not expected_pages:
        return 0.0
    return 1.0 if expected_pages.intersection(set(retrieved_pages)) else 0.0


def precision_at_k(expected_pages: set[int], retrieved_pages: list[int]) -> float:
    if not expected_pages or not retrieved_pages:
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


def chunk_hit_flags(expected_pages: set[int], retrieved_chunks: pd.DataFrame) -> list[int]:
    flags: list[int] = []
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


def compute_leakage(expected_doc_id: str, retrieved_doc_ids: list[str]) -> dict[str, Any]:
    if not expected_doc_id or not retrieved_doc_ids:
        return {"leakage_count_top_k": 0, "leakage_doc_ids_top_k": [], "leakage_rate_top_k": 0.0}
    leakage_docs = [d for d in retrieved_doc_ids if d != expected_doc_id]
    leakage_count = len(leakage_docs)
    return {
        "leakage_count_top_k": int(leakage_count),
        "leakage_doc_ids_top_k": sorted(list(set(leakage_docs))),
        "leakage_rate_top_k": float(leakage_count / max(1, len(retrieved_doc_ids))),
    }


def set_bm25_tokenizer_variant(variant: str) -> None:
    if variant not in BM25_TOKENIZER_VARIANTS:
        raise ValueError(
            f"Unsupported BM25 tokenizer variant '{variant}'. "
            f"Expected one of: {', '.join(BM25_TOKENIZER_VARIANTS)}"
        )
    global _BM25_TOKENIZER_VARIANT
    _BM25_TOKENIZER_VARIANT = str(variant)


def tokenize(text: str, variant: str | None = None) -> list[str]:
    use_variant = str(variant or _BM25_TOKENIZER_VARIANT)
    text_norm = str(text or "").lower()
    if use_variant == "default":
        return re.findall(r"[a-z0-9][a-z0-9\-]{1,}", text_norm)
    if use_variant == "no_hyphen":
        return re.findall(r"[a-z0-9][a-z0-9]{1,}", text_norm)
    raise ValueError(
        f"Unsupported BM25 tokenizer variant '{use_variant}'. "
        f"Expected one of: {', '.join(BM25_TOKENIZER_VARIANTS)}"
    )


class BM25Index:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs_tokens = docs_tokens
        self.k1 = float(k1)
        self.b = float(b)
        self.n_docs = len(docs_tokens)
        self.doc_len = [len(toks) for toks in docs_tokens]
        self.avgdl = float(sum(self.doc_len)) / max(1.0, float(self.n_docs))
        self.term_freqs: list[Counter[str]] = [Counter(toks) for toks in docs_tokens]
        self.doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            for term in tf.keys():
                self.doc_freq[term] += 1
        self.idf: dict[str, float] = {}
        for term, df in self.doc_freq.items():
            # BM25 Robertson IDF with +1 guard.
            self.idf[term] = math.log(1.0 + ((self.n_docs - df + 0.5) / (df + 0.5)))

    def score_query(self, query_tokens: list[str]) -> list[float]:
        if self.n_docs == 0:
            return []
        q_terms = Counter(query_tokens)
        scores = [0.0] * self.n_docs
        for term in q_terms.keys():
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for i, tf in enumerate(self.term_freqs):
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                dl = self.doc_len[i]
                denom = f + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
                scores[i] += idf * (f * (self.k1 + 1.0) / max(denom, 1e-12))
        return scores


def main() -> None:
    args = parse_args()
    set_bm25_tokenizer_variant(str(args.bm25_tokenizer))
    data_dir = Path(args.data_dir).resolve()
    k_list = parse_k_list(args.k_list)
    max_k = max(k_list)

    meta_path = data_dir / "chunk_meta.parquet"
    chunks_path = data_dir / "chunks.parquet"
    eval_set_path = data_dir / "eval_set.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing file: {meta_path}")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing file: {chunks_path}")
    if not eval_set_path.exists():
        raise FileNotFoundError(f"Missing file: {eval_set_path}")

    meta = pd.read_parquet(meta_path)
    chunks = pd.read_parquet(chunks_path)
    eval_obj = json.loads(eval_set_path.read_text(encoding="utf-8"))
    if isinstance(eval_obj, list):
        eval_items = eval_obj
    elif isinstance(eval_obj, dict) and isinstance(eval_obj.get("queries"), list):
        eval_items = eval_obj.get("queries", [])
    else:
        eval_items = []
    if not eval_items:
        raise ValueError(f"eval_set.json must be a non-empty list (or {{'queries': [...]}}): {eval_set_path}")

    # Build chunk_text lookup by chunk id, then align corpus text to meta row order.
    text_by_id: dict[str, str] = {}
    if "chunk_id_global" in chunks.columns:
        for _, r in chunks.iterrows():
            cid = str(r.get("chunk_id_global") or "")
            if cid:
                text_by_id[cid] = str(r.get("chunk_text") or "")
    if "chunk_id" in chunks.columns:
        for _, r in chunks.iterrows():
            cid = str(r.get("chunk_id") or "")
            if cid and cid not in text_by_id:
                text_by_id[cid] = str(r.get("chunk_text") or "")

    corpus_texts: list[str] = []
    for _, r in meta.iterrows():
        cid = str(r.get("chunk_id_global") or r.get("chunk_id") or "")
        corpus_texts.append(text_by_id.get(cid, ""))

    bm25 = BM25Index([tokenize(t) for t in corpus_texts], k1=float(args.k1), b=float(args.b))

    summary_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    meta_doc_ids = set(str(x) for x in meta["doc_id"].dropna().unique()) if "doc_id" in meta.columns else set()

    for item in eval_items:
        query_id = str(item.get("query_id", "")).strip()
        validate_query_id(query_id)
        qid_parts = parse_query_id(query_id)
        question = str(item.get("question", "")).strip()
        if not question:
            continue

        expected_raw = item.get("expected_pages", [])
        expected_pages = set(int(x) for x in expected_raw) if isinstance(expected_raw, list) else set()
        expected_doc_id = get_expected_doc_id(item)
        expected_section = str(item.get("expected_section", "")).strip()
        answer_type = str(item.get("answer_type", "unknown"))

        if expected_doc_id and meta_doc_ids and expected_doc_id not in meta_doc_ids:
            raise ValueError(
                f"Query {query_id} expects doc_id={expected_doc_id}, "
                f"but meta has doc_id values like: {sorted(list(meta_doc_ids))[:5]}"
            )

        scores = bm25.score_query(tokenize(question))
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[: max(max_k, 1)]
        idxs = [idx for idx, _ in ranked]
        ranked_scores = [float(s) for _, s in ranked]

        per_k: dict[str, Any] = {}
        for k in k_list:
            top_idxs = idxs[:k]
            top_scores = ranked_scores[:k]
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

            failure_stage = "hit" if page_recall >= 1.0 else "missed_top_ranked"

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
                    "k": int(k),
                    "answer_type": answer_type,
                    "doc_id": expected_doc_id,
                    "expected_section": expected_section,
                    "expected_pages": sorted(list(expected_pages)),
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

        k1_data = per_k.get("1", {})
        page_hit = 1 if k1_data.get("page_recall_at_k", 0.0) > 0 else 0
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
                "page_hit": page_hit,
                "failure_type": "HIT" if page_hit else "FP2_MISSED_TOP_RANK",
                "failure_stage": "none" if page_hit else "retrieval",
                "per_k": per_k,
            }
        )

    df_sum = pd.DataFrame(summary_rows)
    metrics: dict[str, Any] = {
        "run_info": {
            "run_utc": utc_now_iso(),
            "data_dir": str(data_dir),
            "method": "bm25",
            "runtime": collect_runtime_provenance(),
            "critical_environment_checks": critical_environment_checks(),
            "bm25_k1": float(args.k1),
            "bm25_b": float(args.b),
            "bm25_tokenizer": str(args.bm25_tokenizer),
            "k_list": k_list,
            "num_queries": int(len(results)),
        },
        "metrics_by_k": {},
        "failure_counts_by_k": {},
        "leakage_counts_by_k": {},
    }

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

    metrics["answer_scoring"] = {
        "num_queries_total": int(len(results)),
        "num_queries_scored": 0,
        "answer_accuracy": None,
        "answer_status_counts": {},
    }

    results_json = data_dir / RESULTS_JSON.name
    metrics_json = data_dir / METRICS_JSON.name
    summary_csv = data_dir / SUMMARY_CSV.name

    write_json(results_json, {"run_info": metrics["run_info"], "results": results})
    write_json(metrics_json, metrics)
    df_sum.to_csv(summary_csv, index=False)

    print("Saved:", results_json)
    print("Saved:", metrics_json)
    print("Saved:", summary_csv)
    for k in k_list:
        m = metrics["metrics_by_k"][str(k)]
        print(
            f"k={k} "
            f"page_hit_rate={m['page_hit_rate_at_k']:.3f} "
            f"page_mrr={m['mean_page_mrr_at_k']:.3f} "
            f"page_precision={m['mean_page_precision_at_k']:.3f} "
            f"chunk_hit_rate={m['chunk_hit_rate_at_k']:.3f} "
            f"chunk_mrr={m['mean_chunk_mrr_at_k']:.3f} "
            f"chunk_precision={m['mean_chunk_precision_at_k']:.3f}"
        )


if __name__ == "__main__":
    main()
