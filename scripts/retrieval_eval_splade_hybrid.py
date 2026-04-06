"""
retrieval_eval_splade_hybrid.py

Evaluate top-k retrieval with a fixed hybrid method:
- Dense rank from FAISS (MiniLM embeddings)
- SPLADE sparse rank over chunk text
- Reciprocal Rank Fusion (RRF)

Outputs (written to DATA_DIR):
- retrieval_results_splade_hybrid.json
- retrieval_metrics_splade_hybrid.json
- retrieval_summary_splade_hybrid.csv
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
src_path = repo_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from scripts.retrieval_eval_bm25 import get_retrieved_pages, tokenize
from rag_pdf.question_router import route_question
from rag_pdf.retrieval.rerank import (
    RerankConfig,
    numeric_density_boost,
    query_overlap_boost,
    segment_search_hit_boost,
    table_priority_boost,
)


DATA_DIR = Path("data_processed/Grampian-2024-2025")
INDEX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "chunk_meta.parquet"
CHUNKS_PATH = DATA_DIR / "chunks.parquet"
EVAL_SET_PATH = DATA_DIR / "eval_set.json"

EMBED_MODEL_NAME = "models/all-MiniLM-L6-v2"
SPLADE_MODEL_NAME = "models/naver-splade-cocondenser-ensembledistil"
K_LIST = [1, 3, 5, 10]
MAX_K_SEARCH = int(os.getenv("MAX_K_SEARCH", "100"))

SUBSECTION_BOOST = float(os.getenv("SUBSECTION_BOOST", "0.05"))
TABLE_CHUNK_BOOST = float(os.getenv("TABLE_CHUNK_BOOST", "0.08"))
ENTITY_MATCH_BOOST = float(os.getenv("ENTITY_MATCH_BOOST", "0.04"))
NUMERIC_DENSITY_BOOST = float(os.getenv("NUMERIC_DENSITY_BOOST", "0.03"))
SEGMENT_SEARCH_HIT_BOOST = float(os.getenv("SEGMENT_SEARCH_HIT_BOOST", "0.03"))
MAX_ENTITY_MATCHES = int(os.getenv("MAX_ENTITY_MATCHES", "4"))
ENABLE_LEXICAL_RERANK = os.getenv("ENABLE_LEXICAL_RERANK", "1") != "0"
ENABLE_SUBSECTION_BOOST = os.getenv("ENABLE_SUBSECTION_BOOST", "1") != "0"

RESULTS_JSON = DATA_DIR / "retrieval_results_splade_hybrid.json"
METRICS_JSON = DATA_DIR / "retrieval_metrics_splade_hybrid.json"
SUMMARY_CSV = DATA_DIR / "retrieval_summary_splade_hybrid.csv"

QUERY_ID_PATTERN_V1 = re.compile(r"^Q_(REV|EFF|DEF|STAFF|ACC|GOV|TABLE)_\d{4}_\d{2}$")
QUERY_ID_PATTERN_V2 = re.compile(r"^Q_(\d{4})_([A-Z]+)_(\d{2}|P\d+)$")


def _env_or_default(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val else default


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def _collect_pipeline_settings(data_dir: Path) -> dict[str, Any]:
    metrics_path = data_dir / "metrics.json"
    metrics = _safe_json(metrics_path)
    params = metrics.get("params", {}) if isinstance(metrics, dict) else {}
    return {
        "source_metrics_path": str(metrics_path),
        "source_metrics_exists": metrics_path.exists(),
        "doc_id": metrics.get("doc_id"),
        "corpus_id": metrics.get("corpus_id"),
        "report_year": metrics.get("report_year"),
        "preprocess_run_utc": metrics.get("run_utc"),
        "preprocess_git_commit_short": metrics.get("git_commit_short"),
        "embedding_model_preprocess": metrics.get("embedding_model"),
        "chunk_size_tokens": params.get("chunk_size_tokens"),
        "chunk_overlap_tokens": params.get("chunk_overlap_tokens"),
        "segment_aware_chunking": params.get("segment_aware_chunking"),
        "whole_doc_markdown_mode": params.get("whole_doc_markdown_mode"),
        "markdown_header_carry_forward": params.get("markdown_header_carry_forward"),
        "markdown_table_injection": params.get("markdown_table_injection"),
        "primary_extractor": params.get("primary_extractor"),
        "min_chunk_words": params.get("min_chunk_words"),
    }


def _collect_eval_set_info(eval_set_path: Path, eval_obj: Any, query_count: int) -> dict[str, Any]:
    stat = eval_set_path.stat()
    meta: dict[str, Any] = {}
    if isinstance(eval_obj, dict) and isinstance(eval_obj.get("_meta"), dict):
        meta = eval_obj.get("_meta", {})
    return {
        "path": str(eval_set_path),
        "sha1": _file_sha1(eval_set_path),
        "size_bytes": int(stat.st_size),
        "modified_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat(),
        "query_count": int(query_count),
        "meta_dataset_name": meta.get("dataset_name"),
        "meta_doc_id": meta.get("doc_id"),
        "meta_version": meta.get("version"),
    }


def parse_k_list(val: str) -> list[int]:
    out = [int(x.strip()) for x in val.split(",") if x.strip()]
    if not out:
        raise ValueError("k-list must contain at least one integer")
    if min(out) <= 0:
        raise ValueError("k values must be > 0")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate retrieval using Dense+SPLADE RRF fusion.")
    parser.add_argument(
        "--data-dir",
        default=_env_or_default("DATA_DIR", str(DATA_DIR)),
        help="Directory containing faiss.index, chunk_meta.parquet, chunks.parquet, eval_set.json.",
    )
    parser.add_argument(
        "--model",
        default=_env_or_default("EMBED_MODEL_NAME", EMBED_MODEL_NAME),
        help="Sentence-transformers model name or local path.",
    )
    parser.add_argument(
        "--splade-model",
        default=_env_or_default("SPLADE_MODEL_NAME", SPLADE_MODEL_NAME),
        help="SPLADE masked-LM model path/name (local path recommended).",
    )
    parser.add_argument(
        "--splade-device",
        default=_env_or_default("SPLADE_DEVICE", "auto"),
        help="SPLADE device: auto/cpu/cuda/mps.",
    )
    parser.add_argument(
        "--splade-max-length",
        type=int,
        default=int(_env_or_default("SPLADE_MAX_LENGTH", "256")),
        help="Tokenizer max length for SPLADE encoding.",
    )
    parser.add_argument(
        "--splade-doc-batch-size",
        type=int,
        default=int(_env_or_default("SPLADE_DOC_BATCH_SIZE", "16")),
        help="Batch size for doc chunk SPLADE encoding.",
    )
    parser.add_argument(
        "--splade-query-batch-size",
        type=int,
        default=int(_env_or_default("SPLADE_QUERY_BATCH_SIZE", "16")),
        help="Batch size for query SPLADE encoding.",
    )
    parser.add_argument(
        "--splade-doc-top-terms",
        type=int,
        default=int(_env_or_default("SPLADE_DOC_TOP_TERMS", "128")),
        help="Keep top-N SPLADE terms per doc chunk.",
    )
    parser.add_argument(
        "--splade-query-top-terms",
        type=int,
        default=int(_env_or_default("SPLADE_QUERY_TOP_TERMS", "64")),
        help="Keep top-N SPLADE terms per query.",
    )
    parser.add_argument(
        "--splade-min-weight",
        type=float,
        default=float(_env_or_default("SPLADE_MIN_WEIGHT", "0.01")),
        help="Minimum SPLADE term weight to keep.",
    )
    parser.add_argument(
        "--splade-local-only",
        action="store_true",
        help="Require local model files only (no download).",
    )
    parser.add_argument(
        "--k-list",
        default=_env_or_default("K_LIST", ",".join(str(k) for k in K_LIST)),
        help="Comma-separated list of k values (e.g. 1,3,5,10).",
    )
    parser.add_argument(
        "--rrf-k",
        type=int,
        default=int(_env_or_default("RRF_K", "60")),
        help="RRF constant k in 1/(k+rank).",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=float(_env_or_default("RRF_DENSE_WEIGHT", "1.0")),
        help="Weight multiplier for dense rank contribution.",
    )
    parser.add_argument(
        "--splade-weight",
        type=float,
        default=float(_env_or_default("RRF_SPLADE_WEIGHT", "1.0")),
        help="Weight multiplier for SPLADE rank contribution.",
    )
    parser.add_argument(
        "--no-lexical-rerank",
        action="store_true",
        help="Disable lexical rerank boosts on top of fused ranking.",
    )
    parser.add_argument(
        "--no-subsection-boost",
        action="store_true",
        help="Disable expected_subsection boost on top of fused ranking.",
    )
    parser.add_argument(
        "--enable-cross-encoder-rerank",
        action="store_true",
        help="Enable local cross-encoder reranking on top fused candidates.",
    )
    parser.add_argument(
        "--cross-encoder-model",
        default=_env_or_default("CROSS_ENCODER_MODEL_NAME", "models/bge-reranker-v2-m3"),
        help="Cross-encoder model name/local path.",
    )
    parser.add_argument(
        "--cross-encoder-topn",
        type=int,
        default=int(_env_or_default("CROSS_ENCODER_TOPN", "50")),
        help="Top-N fused candidates to rerank with cross-encoder.",
    )
    parser.add_argument(
        "--cross-encoder-weight",
        type=float,
        default=float(_env_or_default("CROSS_ENCODER_WEIGHT", "0.2")),
        help="Additive weight for normalized cross-encoder score.",
    )
    return parser.parse_args()


def _validate_eval_items(eval_items: list[dict[str, Any]]) -> None:
    warn_qids: list[str] = []
    for item in eval_items:
        qid = str(item.get("query_id", "")).strip()
        expected_raw = item.get("expected_pages", [])
        expected_pages = expected_raw if isinstance(expected_raw, list) else []
        acceptable = item.get("acceptable_evidence", [])
        if len(expected_pages) > 1 and (not isinstance(acceptable, list) or len(acceptable) == 0):
            warn_qids.append(qid or "<missing_query_id>")

    if warn_qids:
        print(
            "[eval-schema-warning] "
            f"{len(warn_qids)} query(s) have multiple expected_pages but missing/empty "
            "acceptable_evidence."
        )
        for qid in warn_qids:
            print(f"  - {qid}")


def validate_query_id(query_id: str) -> None:
    if QUERY_ID_PATTERN_V1.match(query_id) or QUERY_ID_PATTERN_V2.match(query_id):
        return
    if not (query_id.startswith("Q_") and len(query_id) >= 6):
        raise ValueError(f"Invalid query_id '{query_id}'.")


def parse_query_id(query_id: str) -> dict[str, Any]:
    parts = query_id.split("_")
    if len(parts) >= 4 and parts[1].isdigit():
        year = int(parts[1])
        topic = parts[2]
        seq_raw = parts[3]
        seq: Any = int(seq_raw) if seq_raw.isdigit() else seq_raw
        return {"topic": topic, "year": year, "sequence": seq}
    _, topic, year, seq = parts[:4]
    seq_val: Any = int(seq) if str(seq).isdigit() else seq
    return {"topic": topic, "year": int(year), "sequence": seq_val}


def to_int_list(v: Any) -> list[int]:
    if v is None:
        return []
    if isinstance(v, float) and pd.isna(v):
        return []
    if isinstance(v, (list, tuple)):
        out: list[int] = []
        for x in v:
            if x is None:
                continue
            if isinstance(x, dict) and "element" in x:
                nums = re.findall(r"\d+", str(x.get("element")))
                if nums:
                    out.append(int(nums[0]))
            else:
                try:
                    out.append(int(x))
                except Exception:
                    continue
        return out
    s = str(v).strip()
    if not s:
        return []
    return [int(x) for x in re.findall(r"\d+", s)]


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


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def rrf_fuse_with_scores(
    dense_ranked: list[int],
    splade_ranked: list[int],
    rrf_k: int,
    dense_weight: float,
    splade_weight: float,
) -> tuple[list[int], dict[int, float]]:
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + (dense_weight / float(rrf_k + rank))
    for rank, idx in enumerate(splade_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + (splade_weight / float(rrf_k + rank))
    ranked = [idx for idx, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    return ranked, scores


def _normalize_text(v: str) -> str:
    return " ".join(str(v).lower().split())


def _normalize_unit(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi <= lo:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


def _resolve_device(name: str) -> str:
    s = (name or "auto").strip().lower()
    if s == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return s


class SpladeSparseIndex:
    def __init__(
        self,
        model_name: str,
        device: str,
        max_length: int,
        doc_batch_size: int,
        query_batch_size: int,
        doc_top_terms: int,
        query_top_terms: int,
        min_weight: float,
        local_only: bool,
    ) -> None:
        self.device = _resolve_device(device)
        self.max_length = int(max_length)
        self.doc_batch_size = int(doc_batch_size)
        self.query_batch_size = int(query_batch_size)
        self.doc_top_terms = int(doc_top_terms)
        self.query_top_terms = int(query_top_terms)
        self.min_weight = float(min_weight)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_only)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=local_only)
        self.model.to(self.device)
        self.model.eval()

        self.postings: dict[int, list[tuple[int, float]]] = {}

    def _encode_sparse(
        self,
        texts: list[str],
        batch_size: int,
        top_terms: int,
    ) -> list[dict[int, float]]:
        out: list[dict[int, float]] = []
        if not texts:
            return out

        with torch.no_grad():
            for i in range(0, len(texts), max(1, batch_size)):
                batch = texts[i : i + max(1, batch_size)]
                toks = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                toks = {k: v.to(self.device) for k, v in toks.items()}
                logits = self.model(**toks).logits

                # SPLADE pooling: max over sequence of log(1 + relu(logits)).
                activ = torch.log1p(torch.relu(logits))
                attn = toks["attention_mask"].unsqueeze(-1)
                activ = activ * attn
                pooled = torch.max(activ, dim=1).values

                norms = torch.linalg.norm(pooled, dim=1, keepdim=True) + 1e-12
                pooled = pooled / norms

                arr = pooled.detach().cpu().numpy()
                for vec in arr:
                    idx = np.where(vec >= self.min_weight)[0]
                    if idx.size == 0:
                        out.append({})
                        continue
                    vals = vec[idx]
                    if idx.size > top_terms:
                        top_sel = np.argpartition(vals, -top_terms)[-top_terms:]
                        idx = idx[top_sel]
                        vals = vals[top_sel]
                    order = np.argsort(-vals)
                    idx = idx[order]
                    vals = vals[order]
                    sparse = {int(t): float(w) for t, w in zip(idx.tolist(), vals.tolist())}
                    out.append(sparse)
        return out

    def build(self, docs: list[str]) -> None:
        doc_sparse = self._encode_sparse(
            texts=docs,
            batch_size=self.doc_batch_size,
            top_terms=self.doc_top_terms,
        )
        postings: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for d_idx, vec in enumerate(doc_sparse):
            for t_idx, w in vec.items():
                postings[int(t_idx)].append((int(d_idx), float(w)))
        self.postings = dict(postings)

    def rank(self, query: str, top_n: int) -> tuple[list[int], dict[int, float]]:
        q_sparse = self._encode_sparse(
            texts=[query],
            batch_size=self.query_batch_size,
            top_terms=self.query_top_terms,
        )
        q_vec = q_sparse[0] if q_sparse else {}
        acc: dict[int, float] = defaultdict(float)
        for term_id, qw in q_vec.items():
            plist = self.postings.get(int(term_id), [])
            for d_idx, dw in plist:
                acc[d_idx] += float(qw) * float(dw)

        ranked = sorted(acc.items(), key=lambda x: x[1], reverse=True)
        if top_n > 0:
            ranked = ranked[:top_n]
        ranked_ids = [int(i) for i, _ in ranked]
        score_map = {int(i): float(s) for i, s in ranked}
        return ranked_ids, score_map


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    index_path = data_dir / "faiss.index"
    meta_path = data_dir / "chunk_meta.parquet"
    chunks_path = data_dir / "chunks.parquet"
    eval_set_path = data_dir / "eval_set.json"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing file: {index_path}")
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
    _validate_eval_items(eval_items)

    k_list = parse_k_list(args.k_list)
    max_k = max(k_list)
    max_k_search = min(max(MAX_K_SEARCH, max_k), len(meta))
    enable_lexical_rerank = ENABLE_LEXICAL_RERANK and (not args.no_lexical_rerank)
    enable_subsection_boost = ENABLE_SUBSECTION_BOOST and (not args.no_subsection_boost)
    enable_cross_encoder_rerank = bool(args.enable_cross_encoder_rerank)

    # Dense resources
    model = SentenceTransformer(str(args.model))
    index = faiss.read_index(str(index_path))
    questions = [str(x.get("question", "")).strip() for x in eval_items]
    q_emb = model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=False,
        show_progress_bar=True,
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

    splade = SpladeSparseIndex(
        model_name=str(args.splade_model),
        device=str(args.splade_device),
        max_length=int(args.splade_max_length),
        doc_batch_size=int(args.splade_doc_batch_size),
        query_batch_size=int(args.splade_query_batch_size),
        doc_top_terms=int(args.splade_doc_top_terms),
        query_top_terms=int(args.splade_query_top_terms),
        min_weight=float(args.splade_min_weight),
        local_only=bool(args.splade_local_only),
    )
    splade.build(corpus_texts)
    cross_encoder = CrossEncoder(str(args.cross_encoder_model)) if enable_cross_encoder_rerank else None

    rerank_cfg = RerankConfig(
        table_chunk_boost=TABLE_CHUNK_BOOST,
        entity_match_boost=ENTITY_MATCH_BOOST,
        numeric_density_boost=NUMERIC_DENSITY_BOOST,
        segment_search_hit_boost=SEGMENT_SEARCH_HIT_BOOST,
        max_entity_matches=MAX_ENTITY_MATCHES,
    )

    summary_rows: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    meta_doc_ids = set(str(x) for x in meta["doc_id"].dropna().unique()) if "doc_id" in meta.columns else set()

    for qi, item in enumerate(eval_items):
        query_id = str(item.get("query_id", "")).strip()
        validate_query_id(query_id)
        qid_parts = parse_query_id(query_id)
        question = questions[qi]
        if not question:
            continue

        expected_raw = item.get("expected_pages", [])
        expected_pages = set(int(x) for x in expected_raw) if isinstance(expected_raw, list) else set()
        expected_doc_id = get_expected_doc_id(item)
        expected_section = str(item.get("expected_section", "")).strip()
        expected_subsection = str(item.get("expected_subsection", "")).strip()
        evidence_layout = str(item.get("evidence_layout", "")).strip()
        acceptable_evidence = item.get("acceptable_evidence", [])
        filter_hints = item.get("filter_hints", {})
        answer_type = str(item.get("answer_type", "unknown"))

        route = route_question(question) if enable_lexical_rerank else None

        if expected_doc_id and meta_doc_ids and expected_doc_id not in meta_doc_ids:
            raise ValueError(
                f"Query {query_id} expects doc_id={expected_doc_id}, "
                f"but meta has doc_id values like: {sorted(list(meta_doc_ids))[:5]}"
            )

        # Dense ranking
        dense_scores, dense_idxs = index.search(q_emb[qi : qi + 1], max_k_search)
        dense_ranked = dense_idxs[0].tolist()
        dense_score_map: dict[int, float] = {
            int(idx): float(score) for idx, score in zip(dense_idxs[0].tolist(), dense_scores[0].tolist())
        }

        # SPLADE sparse ranking
        splade_ranked, splade_score_map = splade.rank(query=question, top_n=max_k_search)

        fused_ranked, fused_scores = rrf_fuse_with_scores(
            dense_ranked=dense_ranked,
            splade_ranked=splade_ranked,
            rrf_k=int(args.rrf_k),
            dense_weight=float(args.dense_weight),
            splade_weight=float(args.splade_weight),
        )
        scores_map: dict[int, float] = dict(fused_scores)

        if cross_encoder is not None and fused_ranked:
            ce_topn = max(1, min(int(args.cross_encoder_topn), len(fused_ranked)))
            cand = fused_ranked[:ce_topn]
            pairs: list[tuple[str, str]] = []
            for idx in cand:
                row = meta.iloc[idx]
                cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
                pairs.append((question, text_by_id.get(cid, "")))
            ce_scores_raw = np.asarray(cross_encoder.predict(pairs), dtype=np.float32)
            ce_scores = _normalize_unit(ce_scores_raw)
            for idx, ce_s in zip(cand, ce_scores.tolist()):
                scores_map[idx] = float(scores_map.get(idx, 0.0)) + float(args.cross_encoder_weight) * float(ce_s)

        if enable_lexical_rerank:
            for idx in fused_ranked:
                row = meta.iloc[idx]
                is_table_chunk = bool(row.get("is_table", False))
                cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
                ctext = text_by_id.get(cid, "")
                score = scores_map.get(idx, 0.0)
                score += table_priority_boost(
                    is_table_chunk=is_table_chunk,
                    route_intent=route.intent if route is not None else "generic",
                    config=rerank_cfg,
                )
                score += query_overlap_boost(question=question, chunk_text=ctext, config=rerank_cfg)
                score += numeric_density_boost(question=question, chunk_text=ctext, config=rerank_cfg)
                score += segment_search_hit_boost(
                    question=question,
                    segment_has_search_hit=bool(row.get("segment_has_search_hit", False)),
                    config=rerank_cfg,
                )
                scores_map[idx] = score

        if enable_subsection_boost and expected_subsection and "subsection_title" in meta.columns:
            target = _normalize_text(expected_subsection)
            for idx in fused_ranked:
                sub = _normalize_text(str(meta.iloc[idx].get("subsection_title", "")))
                if sub == target:
                    scores_map[idx] = scores_map.get(idx, 0.0) + SUBSECTION_BOOST

        fused_ranked = sorted(fused_ranked, key=lambda i: scores_map.get(i, 0.0), reverse=True)

        dense_rank_map: dict[int, int] = {idx: r for r, idx in enumerate(dense_ranked, start=1)}
        splade_rank_map: dict[int, int] = {idx: r for r, idx in enumerate(splade_ranked, start=1)}

        per_k: dict[str, Any] = {}
        for k in k_list:
            top_idxs = fused_ranked[:k]
            retrieved_chunks = meta.iloc[top_idxs].copy()
            retrieved_chunks["score"] = [scores_map.get(i, 0.0) for i in top_idxs]

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
                    "expected_subsection": expected_subsection,
                    "expected_pages": sorted(list(expected_pages)),
                    "evidence_layout": evidence_layout,
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

        top10_ids = fused_ranked[:10]
        top10_debug = []
        for idx in top10_ids:
            top10_debug.append(
                {
                    "chunk_id": str(meta.iloc[idx].get("chunk_id_global") or meta.iloc[idx].get("chunk_id") or ""),
                    "rrf_score": float(scores_map.get(idx, 0.0)),
                    "dense_rank": int(dense_rank_map.get(idx, 0)),
                    "dense_raw_score": float(dense_score_map.get(idx, 0.0)),
                    "splade_rank": int(splade_rank_map.get(idx, 0)),
                    "splade_raw_score": float(splade_score_map.get(idx, 0.0)),
                    "pages": to_int_list(meta.iloc[idx].get("pages")),
                }
            )

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
                "expected_subsection": expected_subsection,
                "expected_pages": sorted(list(expected_pages)),
                "evidence_layout": evidence_layout,
                "acceptable_evidence": acceptable_evidence,
                "filter_hints": filter_hints,
                "page_hit": page_hit,
                "failure_type": "HIT" if page_hit else "FP2_MISSED_TOP_RANK",
                "failure_stage": "none" if page_hit else "retrieval",
                "top10_debug": top10_debug,
                "per_k": per_k,
            }
        )

    df_sum = pd.DataFrame(summary_rows)
    metrics: dict[str, Any] = {
        "run_info": {
            "run_utc": utc_now_iso(),
            "data_dir": str(data_dir),
            "method": "hybrid_rrf_dense_splade",
            "embedding_model": str(args.model),
            "splade_model": str(args.splade_model),
            "splade_device": _resolve_device(str(args.splade_device)),
            "splade_max_length": int(args.splade_max_length),
            "splade_doc_batch_size": int(args.splade_doc_batch_size),
            "splade_query_batch_size": int(args.splade_query_batch_size),
            "splade_doc_top_terms": int(args.splade_doc_top_terms),
            "splade_query_top_terms": int(args.splade_query_top_terms),
            "splade_min_weight": float(args.splade_min_weight),
            "rrf_k": int(args.rrf_k),
            "dense_weight": float(args.dense_weight),
            "splade_weight": float(args.splade_weight),
            "enable_lexical_rerank": bool(enable_lexical_rerank),
            "enable_subsection_boost": bool(enable_subsection_boost),
            "enable_cross_encoder_rerank": bool(enable_cross_encoder_rerank),
            "cross_encoder_model": (str(args.cross_encoder_model) if enable_cross_encoder_rerank else None),
            "cross_encoder_topn": int(args.cross_encoder_topn),
            "cross_encoder_weight": float(args.cross_encoder_weight),
            "table_chunk_boost": float(TABLE_CHUNK_BOOST),
            "entity_match_boost": float(ENTITY_MATCH_BOOST),
            "numeric_density_boost": float(NUMERIC_DENSITY_BOOST),
            "segment_search_hit_boost": float(SEGMENT_SEARCH_HIT_BOOST),
            "subsection_boost": float(SUBSECTION_BOOST),
            "k_list": k_list,
            "num_queries": int(len(results)),
            "num_chunks_indexed": int(len(meta)),
            "pipeline_settings": _collect_pipeline_settings(data_dir),
            "eval_set": _collect_eval_set_info(eval_set_path, eval_obj, len(eval_items)),
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
