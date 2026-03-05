from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import os
import time
import logging
import threading
from pathlib import Path
from typing import Any, Optional
import re

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag_pdf.services.local_llm_service import LocalLLMService

MAX_K_SEARCH = 100
# Keep API/UI retrieval defaults aligned with retrieval_eval_hybrid.py.
# Tuned on 5 complete-GT files (250 queries), repeated CV 5x5 on 2026-02-26.
RRF_K = 20
RRF_DENSE_WEIGHT = 0.5
RRF_BM25_WEIGHT = 2.0
ENABLE_CROSS_ENCODER_RERANK = os.getenv("ENABLE_CROSS_ENCODER_RERANK", "0").strip().lower() in {
    "1", "true", "yes", "y", "on"
}
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL_NAME", "models/bge-reranker-v2-m3")
CROSS_ENCODER_TOPN = int(os.getenv("CROSS_ENCODER_TOPN", "50"))
CROSS_ENCODER_WEIGHT = float(os.getenv("CROSS_ENCODER_WEIGHT", "0.2"))
ANSWER_GATE_ENABLED = os.getenv("ANSWER_GATE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
ANSWER_GATE_WINDOW_CHARS = int(os.getenv("ANSWER_GATE_WINDOW_CHARS", "180"))
ANSWER_GATE_MIN_SCORE = int(os.getenv("ANSWER_GATE_MIN_SCORE", "2"))
ANSWER_EVAL_STRICT_PAGE = os.getenv("ANSWER_EVAL_STRICT_PAGE", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
ANSWER_PREFER_NON_TABLE = os.getenv("ANSWER_PREFER_NON_TABLE", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
ANCHOR_DISTANCE_GATE_ENABLED = os.getenv("ANCHOR_DISTANCE_GATE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
ANCHOR_DISTANCE_MAX_CHARS = int(os.getenv("ANCHOR_DISTANCE_MAX_CHARS", "260"))
ANCHOR_DISTANCE_STRICT_NULL = os.getenv("ANCHOR_DISTANCE_STRICT_NULL", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
ENTITY_CONSISTENCY_GATE_ENABLED = os.getenv("ENTITY_CONSISTENCY_GATE_ENABLED", "1").strip().lower() in {"1", "true", "yes", "y", "on"}
ENTITY_CONSISTENCY_WINDOW_CHARS = int(os.getenv("ENTITY_CONSISTENCY_WINDOW_CHARS", "220"))
ENTITY_CONSISTENCY_MAX_ANCHOR_DISTANCE = int(os.getenv("ENTITY_CONSISTENCY_MAX_ANCHOR_DISTANCE", "340"))
RETRIEVAL_MARGIN_LOW_THRESHOLD = float(os.getenv("RETRIEVAL_MARGIN_LOW_THRESHOLD", "0.002"))
GEN_MAX_CONTEXT_CHUNKS = int(os.getenv("GEN_MAX_CONTEXT_CHUNKS", "5"))
GEN_MAX_CONTEXT_CHARS = int(os.getenv("GEN_MAX_CONTEXT_CHARS", "9000"))
GEN_MAX_CHUNK_CHARS = int(os.getenv("GEN_MAX_CHUNK_CHARS", "2200"))
GEN_TIMEOUT_SECONDS = float(os.getenv("GEN_TIMEOUT_SECONDS", "20"))

logger = logging.getLogger(__name__)

TRUST_CANONICAL_MAP: dict[str, str] = {
    "ayrshire and arran": "NHS Ayrshire & Arran",
    "borders": "NHS Borders",
    "dumfries & galloway": "NHS Dumfries & Galloway",
    "dumfries and galloway": "NHS Dumfries & Galloway",
    "fife": "NHS Fife",
    "forth valley": "NHS Forth Valley",
    "grampian": "NHS Grampian",
    "greater glasgow & clyde": "NHS Greater Glasgow & Clyde",
    "greater glasgow and clyde": "NHS Greater Glasgow & Clyde",
    "highland": "NHS Highland",
    "lanarkshire": "NHS Lanarkshire",
    "lothian": "NHS Lothian",
    "orkney": "NHS Orkney",
    "shetland": "NHS Shetland",
    "tayside": "NHS Tayside",
    "western isles": "NHS Western Isles",
}


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization for cosine similarity with IndexFlatIP."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def to_pages_list(v: Any) -> list[int]:
    """Normalize pages-like value into list[int]."""
    if v is None:
        return []
    if isinstance(v, list):
        out: list[int] = []
        for x in v:
            if isinstance(x, dict) and "element" in x:
                try:
                    out.append(int(x["element"]))
                except Exception:
                    continue
            else:
                try:
                    out.append(int(x))
                except Exception:
                    continue
        return out
    try:
        return [int(v)]
    except Exception:
        return []


def tokenize(text: str) -> list[str]:
    """Tokenize text for lightweight BM25 lexical scoring."""
    return re.findall(r"[a-z0-9][a-z0-9\-]{1,}", str(text or "").lower())


class BM25Index:
    """Simple BM25 implementation for lexical retrieval over chunk text."""

    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
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


def rrf_fuse(
    dense_ranked: list[int],
    bm25_ranked: list[int],
    rrf_k: int = RRF_K,
    dense_weight: float = RRF_DENSE_WEIGHT,
    bm25_weight: float = RRF_BM25_WEIGHT,
) -> tuple[list[int], dict[int, float]]:
    """Fuse ranked lists with Reciprocal Rank Fusion and return fused rank + scores."""
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + (dense_weight / float(rrf_k + rank))
    for rank, idx in enumerate(bm25_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + (bm25_weight / float(rrf_k + rank))
    ranked = [idx for idx, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    return ranked, scores


@dataclass
class LoadedDoc:
    """In-memory retrieval resources for one document."""

    index: faiss.Index
    meta: pd.DataFrame
    eval_items: list[dict[str, Any]]
    bm25: BM25Index
    chunk_text_by_id: dict[str, str]
    chunk_section_by_id: dict[str, str]
    chunk_subsection_by_id: dict[str, str]


@dataclass
class LoadedGlobal:
    """In-memory retrieval resources for multi-document retrieval scopes."""

    index: faiss.Index
    meta: pd.DataFrame
    chunk_text_by_id: dict[str, str]
    chunk_section_by_id: dict[str, str]
    chunk_subsection_by_id: dict[str, str]


class SearchService:
    """Run top-k vector retrieval over processed artifacts."""

    def __init__(self, repo_root: Path, model_path: Path) -> None:
        self.repo_root = repo_root
        self.model = SentenceTransformer(str(model_path))
        self.cross_encoder: Optional[CrossEncoder] = None
        self.cross_encoder_topn = max(1, int(CROSS_ENCODER_TOPN))
        self.cross_encoder_weight = float(CROSS_ENCODER_WEIGHT)
        if ENABLE_CROSS_ENCODER_RERANK:
            try:
                self.cross_encoder = CrossEncoder(str(CROSS_ENCODER_MODEL_NAME))
            except Exception as exc:
                logger.warning("Cross-encoder reranker disabled; failed to load model %s: %s", CROSS_ENCODER_MODEL_NAME, exc)
        self.local_llm = LocalLLMService()
        self._cache: dict[str, LoadedDoc] = {}
        self._global_cache: dict[str, LoadedGlobal] = {}
        self._cache_sig: dict[str, tuple[Any, ...]] = {}
        self._global_cache_sig: dict[str, tuple[Any, ...]] = {}
        self.gen_max_context_chunks = max(1, int(GEN_MAX_CONTEXT_CHUNKS))
        self.gen_max_context_chars = max(1000, int(GEN_MAX_CONTEXT_CHARS))
        self.gen_max_chunk_chars = max(200, int(GEN_MAX_CHUNK_CHARS))
        self.gen_timeout_seconds = float(GEN_TIMEOUT_SECONDS)
        self._obs_lock = threading.Lock()
        self._obs: dict[str, float] = {
            "generation_total": 0.0,
            "generation_ok": 0.0,
            "generation_skipped": 0.0,
            "generation_insufficient_evidence": 0.0,
            "generation_error": 0.0,
            "citations_parsed_total": 0.0,
            "citations_valid_total": 0.0,
            "citations_rejected_total": 0.0,
            "generation_latency_ms_sum": 0.0,
        }

    @staticmethod
    def _file_signature(path: Path) -> tuple[bool, int, int]:
        """Small signature for cache invalidation when artifacts change."""
        if not path.exists():
            return (False, 0, 0)
        try:
            stat = path.stat()
            return (True, int(stat.st_mtime_ns), int(stat.st_size))
        except Exception:
            return (True, 0, 0)

    def _doc_artifact_signature(self, data_dir: Path) -> tuple[Any, ...]:
        return (
            self._file_signature(data_dir / "faiss.index"),
            self._file_signature(data_dir / "chunk_meta.parquet"),
            self._file_signature(data_dir / "chunks.parquet"),
            self._file_signature(data_dir / "eval_set.json"),
        )

    def _global_artifact_signature(self, data_root: Path) -> tuple[Any, ...]:
        per_doc: list[tuple[str, tuple[bool, int, int], tuple[bool, int, int], tuple[bool, int, int]]] = []
        try:
            docs = sorted([d for d in data_root.iterdir() if d.is_dir()], key=lambda p: p.name)
        except Exception:
            docs = []
        for d in docs:
            per_doc.append(
                (
                    d.name,
                    self._file_signature(d / "embeddings.npy"),
                    self._file_signature(d / "chunk_meta.parquet"),
                    self._file_signature(d / "chunks.parquet"),
                )
            )
        return tuple(per_doc)

    def _build_local_generation_prompt(self, question: str, results: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        """
        Build a strict grounded prompt:
        - Use only supplied context
        - Include page/chunk citations
        """
        blocks: list[str] = []
        total_chars = 0
        used_chunks = 0
        truncated_chunks = 0
        context_truncated = False
        for r in results:
            if used_chunks >= self.gen_max_context_chunks:
                context_truncated = True
                break
            chunk_id = str(r.get("chunk_id") or "").strip()
            pages = [int(x) for x in (r.get("pages") or []) if str(x).strip().isdigit()]
            page_label = ",".join(str(p) for p in pages) if pages else "NA"
            text = str(r.get("chunk_text") or "").strip()
            if not text:
                continue
            if len(text) > self.gen_max_chunk_chars:
                text = text[: self.gen_max_chunk_chars].rstrip() + " ..."
                truncated_chunks += 1
            candidate_block = f"[chunk_id={chunk_id} pages={page_label}]\n{text}"
            if total_chars + len(candidate_block) > self.gen_max_context_chars:
                context_truncated = True
                break
            blocks.append(
                candidate_block
            )
            total_chars += len(candidate_block)
            used_chunks += 1
        context = "\n\n".join(blocks).strip()
        if not context:
            context = "[no context]"

        prompt = (
            "You are a retrieval-grounded assistant.\n"
            "Use only the provided CONTEXT.\n"
            "If the answer is not explicitly supported, reply exactly: "
            "\"Insufficient evidence in retrieved context.\"\n"
            "Keep answer concise and factual.\n"
            "Do not invent chunk_id or page values.\n"
            "Return JSON only (no markdown/code fences) with this exact shape:\n"
            "{\"answer\":\"...\",\"citations\":[{\"chunk_id\":\"...\",\"page\":21}]}\n"
            "When answer is unsupported, return:\n"
            "{\"answer\":\"Insufficient evidence in retrieved context.\",\"citations\":[]}\n\n"
            f"QUESTION:\n{str(question).strip()}\n\n"
            f"CONTEXT:\n{context}\n\n"
            "ANSWER:"
        )
        return prompt, {
            "context_chunks_used": int(used_chunks),
            "context_chars_used": int(total_chars),
            "context_chunk_char_limit": int(self.gen_max_chunk_chars),
            "context_max_chunks": int(self.gen_max_context_chunks),
            "context_max_chars": int(self.gen_max_context_chars),
            "context_truncated": bool(context_truncated),
            "chunk_text_truncations": int(truncated_chunks),
        }

    def _generate_local_answer(self, question: str, results: list[dict[str, Any]]) -> tuple[Optional[str], dict[str, Any]]:
        prompt, ctx_stats = self._build_local_generation_prompt(question=question, results=results)
        t0 = time.perf_counter()
        out = self.local_llm.generate(prompt, timeout_seconds=self.gen_timeout_seconds)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return out.answer, {
            "provider": "local_ollama",
            "status": out.status,
            "model": out.model,
            "error": out.error,
            "prompt_chars": int(out.prompt_chars),
            "latency_ms": float(round(latency_ms, 3)),
            "timeout_seconds": float(self.gen_timeout_seconds),
            **ctx_stats,
        }

    @staticmethod
    def _extract_citations_from_answer(answer: str) -> list[dict[str, Any]]:
        """Extract citations from generated text, accepting common format variants."""
        text = str(answer or "")
        if not text:
            return []
        patterns = [
            # [chunk_id=abc, page=21]
            re.compile(
                r"\[\s*chunk_id\s*[:=]\s*([^,\]\)]+)\s*[,;]\s*page(?:s)?\s*[:=]\s*(\d+)\s*\]",
                flags=re.IGNORECASE,
            ),
            # (chunk_id: abc, page: 21)
            re.compile(
                r"\(\s*chunk_id\s*[:=]\s*([^,\]\)]+)\s*[,;]\s*page(?:s)?\s*[:=]\s*(\d+)\s*\)",
                flags=re.IGNORECASE,
            ),
            # chunk_id=abc page=21 (inline free text)
            re.compile(
                r"chunk_id\s*[:=]\s*([A-Za-z0-9:_\-.]+)\s*[,; ]+\s*page(?:s)?\s*[:=]\s*(\d+)",
                flags=re.IGNORECASE,
            ),
        ]
        out: list[dict[str, Any]] = []
        seen: set[tuple[str, int]] = set()
        for pattern in patterns:
            for m in pattern.finditer(text):
                chunk_id = str(m.group(1) or "").strip()
                try:
                    page = int(m.group(2))
                except Exception:
                    continue
                key = (chunk_id, page)
                if not chunk_id or key in seen:
                    continue
                seen.add(key)
                out.append({"chunk_id": chunk_id, "page": page})
        return out

    @staticmethod
    def _parse_generation_json_payload(raw_answer: str) -> tuple[Optional[str], list[dict[str, Any]], str]:
        """
        Parse model output as JSON payload:
        {"answer":"...","citations":[{"chunk_id":"...","page":21}]}
        Returns: (answer, citations, mode)
        """
        text = str(raw_answer or "").strip()
        if not text:
            return None, [], "empty"

        payload: Optional[dict[str, Any]] = None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = None

        # Fallback: parse fenced json blocks if model still emits markdown fences.
        if payload is None:
            m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, flags=re.IGNORECASE)
            if m:
                try:
                    parsed = json.loads(m.group(1).strip())
                    if isinstance(parsed, dict):
                        payload = parsed
                except Exception:
                    payload = None

        if payload is None:
            return text, [], "text_fallback"

        answer = payload.get("answer")
        answer_text = str(answer).strip() if answer is not None else None
        raw_citations = payload.get("citations")
        citations: list[dict[str, Any]] = []
        if isinstance(raw_citations, list):
            for item in raw_citations:
                if isinstance(item, dict):
                    chunk_id = str(item.get("chunk_id") or item.get("chunk") or "").strip()
                    page_raw = item.get("page", item.get("page_no", item.get("pages")))
                    if isinstance(page_raw, list) and page_raw:
                        page_raw = page_raw[0]
                    try:
                        page = int(page_raw)
                    except Exception:
                        continue
                    if chunk_id:
                        citations.append({"chunk_id": chunk_id, "page": page})
                elif isinstance(item, str):
                    citations.extend(SearchService._extract_citations_from_answer(item))
        return answer_text, citations, "json"

    @staticmethod
    def _validate_citations(
        citations: list[dict[str, Any]],
        results: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], int]:
        """Keep citations that map to retrieved chunk/page pairs and count rejected ones."""
        allowed_pages_by_chunk: dict[str, set[int]] = {}
        for r in results:
            chunk_id = str(r.get("chunk_id") or "").strip()
            if not chunk_id:
                continue
            pages = {int(p) for p in (r.get("pages") or []) if str(p).strip().isdigit()}
            allowed_pages_by_chunk[chunk_id] = pages
        valid: list[dict[str, Any]] = []
        rejected = 0
        for c in citations:
            cid = str(c.get("chunk_id") or "").strip()
            page_raw = c.get("page")
            try:
                page = int(page_raw)
            except Exception:
                rejected += 1
                continue
            if cid in allowed_pages_by_chunk and page in allowed_pages_by_chunk[cid]:
                valid.append({"chunk_id": cid, "page": page})
            else:
                rejected += 1
        return valid, rejected

    def _update_generation_observability(
        self,
        *,
        generation_status: str,
        citations_parsed: int,
        citations_valid: int,
        citations_rejected: int,
        latency_ms: float,
    ) -> None:
        """Update in-memory counters and emit one log line for monitoring."""
        with self._obs_lock:
            self._obs["generation_total"] += 1.0
            key = f"generation_{generation_status}"
            if key in self._obs:
                self._obs[key] += 1.0
            elif generation_status not in {"ok", "skipped", "insufficient_evidence"}:
                self._obs["generation_error"] += 1.0
            self._obs["citations_parsed_total"] += float(citations_parsed)
            self._obs["citations_valid_total"] += float(citations_valid)
            self._obs["citations_rejected_total"] += float(citations_rejected)
            self._obs["generation_latency_ms_sum"] += float(latency_ms)
            totals = {
                "total": int(self._obs["generation_total"]),
                "ok": int(self._obs["generation_ok"]),
                "skipped": int(self._obs["generation_skipped"]),
                "insufficient_evidence": int(self._obs["generation_insufficient_evidence"]),
                "error": int(self._obs["generation_error"]),
            }
        logger.info(
            "generation_metrics status=%s parsed=%s valid=%s rejected=%s latency_ms=%.3f totals=%s",
            generation_status,
            citations_parsed,
            citations_valid,
            citations_rejected,
            latency_ms,
            totals,
        )

    def get_generation_observability_snapshot(self) -> dict[str, Any]:
        """Return a snapshot of generation observability counters and derived rates."""
        with self._obs_lock:
            raw = dict(self._obs)
        total = int(raw.get("generation_total", 0.0))
        parsed = int(raw.get("citations_parsed_total", 0.0))
        valid = int(raw.get("citations_valid_total", 0.0))
        rejected = int(raw.get("citations_rejected_total", 0.0))
        latency_sum = float(raw.get("generation_latency_ms_sum", 0.0))
        return {
            "generation_counts": {
                "total": total,
                "ok": int(raw.get("generation_ok", 0.0)),
                "skipped": int(raw.get("generation_skipped", 0.0)),
                "insufficient_evidence": int(raw.get("generation_insufficient_evidence", 0.0)),
                "error": int(raw.get("generation_error", 0.0)),
            },
            "citation_counts": {
                "parsed_total": parsed,
                "valid_total": valid,
                "rejected_total": rejected,
            },
            "derived": {
                "citation_valid_rate": (float(valid) / float(parsed)) if parsed > 0 else None,
                "citation_rejected_rate": (float(rejected) / float(parsed)) if parsed > 0 else None,
                "generation_avg_latency_ms": (latency_sum / float(total)) if total > 0 else None,
            },
        }

    @staticmethod
    def _trust_from_doc_id(doc_id: str) -> str:
        """
        Derive canonical trust id from document id.

        Examples:
        - 'Grampian-2023-2024' -> 'NHS Grampian'
        - 'NHS Shetland-2022-2023' -> 'NHS Shetland'
        """
        d = str(doc_id or "").strip()
        if not d:
            return ""
        base = d.split("-", 1)[0].strip()
        norm = base.lower().replace("nhs ", "").replace("_", " ")
        norm = re.sub(r"\s+", " ", norm).strip()
        return TRUST_CANONICAL_MAP.get(norm, f"NHS {base.replace('_', ' ').strip()}")

    @staticmethod
    def _read_eval_items(eval_path: Path) -> list[dict[str, Any]]:
        if not eval_path.exists():
            return []
        raw_eval = json.loads(eval_path.read_text(encoding="utf-8"))
        if isinstance(raw_eval, list):
            return [x for x in raw_eval if isinstance(x, dict)]
        if isinstance(raw_eval, dict):
            queries = raw_eval.get("queries")
            if isinstance(queries, list):
                return [x for x in queries if isinstance(x, dict)]
        return []

    @staticmethod
    def _apply_filters(meta: pd.DataFrame, filters: Optional[dict[str, Any]]) -> np.ndarray:
        """
        Build a boolean mask from metadata filters.

        Supported filter keys:
        - doc_id (str)
        - trust_id (str)
        - year (int)
        - is_table (bool)
        - section_contains (str)
        - subsection_contains (str)
        """
        if meta.empty:
            return np.zeros((0,), dtype=bool)
        mask = np.ones((len(meta),), dtype=bool)
        if not filters:
            return mask

        doc_id = str(filters.get("doc_id") or "").strip()
        if doc_id and "doc_id" in meta.columns:
            mask &= (meta["doc_id"].astype(str).values == doc_id)

        trust_id = str(filters.get("trust_id") or "").strip().lower()
        if trust_id and "trust_id" in meta.columns:
            mask &= (meta["trust_id"].astype(str).str.lower().values == trust_id)

        year_val = filters.get("year")
        if year_val is not None:
            if "year" in meta.columns:
                try:
                    y = int(year_val)
                    mask &= (pd.to_numeric(meta["year"], errors="coerce").fillna(-1).astype(int).values == y)
                except Exception:
                    pass
            elif "report_year" in meta.columns:
                try:
                    y = int(year_val)
                    mask &= (pd.to_numeric(meta["report_year"], errors="coerce").fillna(-1).astype(int).values == y)
                except Exception:
                    pass

        is_table = filters.get("is_table")
        if is_table is not None and "is_table" in meta.columns:
            want = bool(is_table)
            cur = meta["is_table"].fillna(False).astype(bool).values
            mask &= (cur == want)

        sec = str(filters.get("section_contains") or "").strip()
        if sec and "section_title" in meta.columns:
            mask &= meta["section_title"].fillna("").astype(str).str.contains(sec, case=False, regex=False).values

        sub = str(filters.get("subsection_contains") or "").strip()
        if sub and "subsection_title" in meta.columns:
            mask &= meta["subsection_title"].fillna("").astype(str).str.contains(sub, case=False, regex=False).values

        return mask

    def _load_doc(self, data_dir: Path) -> LoadedDoc:
        key = str(data_dir.resolve())
        sig = self._doc_artifact_signature(data_dir)
        if key in self._cache and self._cache_sig.get(key) == sig:
            return self._cache[key]

        index_path = data_dir / "faiss.index"
        meta_path = data_dir / "chunk_meta.parquet"
        chunks_path = data_dir / "chunks.parquet"
        eval_path = data_dir / "eval_set.json"
        if not index_path.exists() or not meta_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Missing retrieval artifacts in {data_dir}")

        index = faiss.read_index(str(index_path))
        meta = pd.read_parquet(meta_path)
        chunks = pd.read_parquet(chunks_path)
        eval_items = self._read_eval_items(eval_path)

        if "doc_id" not in meta.columns:
            meta["doc_id"] = data_dir.name
        meta["doc_id"] = meta["doc_id"].fillna(data_dir.name).astype(str)
        if "trust_id" not in meta.columns:
            meta["trust_id"] = meta["doc_id"].map(self._trust_from_doc_id)
        if "year" not in meta.columns:
            if "report_year" in meta.columns:
                meta["year"] = pd.to_numeric(meta["report_year"], errors="coerce").astype("Int64")
            else:
                meta["year"] = pd.Series([pd.NA] * len(meta), dtype="Int64")

        chunk_text_by_id: dict[str, str] = {}
        chunk_section_by_id: dict[str, str] = {}
        chunk_subsection_by_id: dict[str, str] = {}
        if "chunk_id_global" in chunks.columns:
            for _, row in chunks.iterrows():
                cid = str(row.get("chunk_id_global") or "")
                if cid:
                    chunk_text_by_id[cid] = str(row.get("chunk_text") or "")
                    chunk_section_by_id[cid] = str(row.get("section_title") or "")
                    chunk_subsection_by_id[cid] = str(row.get("subsection_title") or "")
        if "chunk_id" in chunks.columns:
            for _, row in chunks.iterrows():
                cid = str(row.get("chunk_id") or "")
                if cid and cid not in chunk_text_by_id:
                    chunk_text_by_id[cid] = str(row.get("chunk_text") or "")
                    chunk_section_by_id[cid] = str(row.get("section_title") or "")
                    chunk_subsection_by_id[cid] = str(row.get("subsection_title") or "")

        corpus_texts: list[str] = []
        for _, row in meta.iterrows():
            cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
            corpus_texts.append(chunk_text_by_id.get(cid, ""))
        bm25 = BM25Index([tokenize(t) for t in corpus_texts], k1=1.5, b=0.75)

        loaded = LoadedDoc(
            index=index,
            meta=meta,
            eval_items=eval_items,
            bm25=bm25,
            chunk_text_by_id=chunk_text_by_id,
            chunk_section_by_id=chunk_section_by_id,
            chunk_subsection_by_id=chunk_subsection_by_id,
        )
        self._cache[key] = loaded
        self._cache_sig[key] = sig
        return loaded

    def _load_global(self, data_root: Path) -> LoadedGlobal:
        """
        Load/build a global dense index over all processed documents under `data_root`.

        Requirements per doc folder:
        - embeddings.npy
        - chunk_meta.parquet
        - chunks.parquet
        """
        key = str(data_root.resolve())
        sig = self._global_artifact_signature(data_root)
        if key in self._global_cache and self._global_cache_sig.get(key) == sig:
            return self._global_cache[key]

        metas: list[pd.DataFrame] = []
        embs: list[np.ndarray] = []
        chunk_text_by_id: dict[str, str] = {}
        chunk_section_by_id: dict[str, str] = {}
        chunk_subsection_by_id: dict[str, str] = {}

        for d in sorted(data_root.iterdir()):
            if not d.is_dir():
                continue
            emb_path = d / "embeddings.npy"
            meta_path = d / "chunk_meta.parquet"
            chunks_path = d / "chunks.parquet"
            if not (emb_path.exists() and meta_path.exists() and chunks_path.exists()):
                continue

            try:
                emb = np.load(emb_path).astype("float32")
                meta = pd.read_parquet(meta_path)
                chunks = pd.read_parquet(chunks_path)
            except Exception:
                continue

            if len(meta) == 0 or len(meta) != emb.shape[0]:
                continue

            if "doc_id" not in meta.columns:
                meta["doc_id"] = d.name
            meta["doc_id"] = meta["doc_id"].fillna(d.name).astype(str)
            if "trust_id" not in meta.columns:
                meta["trust_id"] = meta["doc_id"].map(self._trust_from_doc_id)
            if "year" not in meta.columns:
                if "report_year" in meta.columns:
                    meta["year"] = pd.to_numeric(meta["report_year"], errors="coerce").astype("Int64")
                else:
                    meta["year"] = pd.Series([pd.NA] * len(meta), dtype="Int64")

            if "chunk_id_global" in chunks.columns:
                for _, row in chunks.iterrows():
                    cid = str(row.get("chunk_id_global") or "")
                    if cid:
                        chunk_text_by_id[cid] = str(row.get("chunk_text") or "")
                        chunk_section_by_id[cid] = str(row.get("section_title") or "")
                        chunk_subsection_by_id[cid] = str(row.get("subsection_title") or "")
            if "chunk_id" in chunks.columns:
                for _, row in chunks.iterrows():
                    cid = str(row.get("chunk_id") or "")
                    if cid and cid not in chunk_text_by_id:
                        chunk_text_by_id[cid] = str(row.get("chunk_text") or "")
                        chunk_section_by_id[cid] = str(row.get("section_title") or "")
                        chunk_subsection_by_id[cid] = str(row.get("subsection_title") or "")

            metas.append(meta)
            embs.append(emb)

        if not metas or not embs:
            raise FileNotFoundError(f"No global-ready artifacts found under {data_root}")

        global_meta = pd.concat(metas, ignore_index=True)
        global_emb = np.vstack(embs).astype("float32")
        global_emb = l2_normalize(global_emb).astype("float32")

        d = int(global_emb.shape[1])
        index = faiss.IndexFlatIP(d)
        index.add(global_emb)

        loaded = LoadedGlobal(
            index=index,
            meta=global_meta,
            chunk_text_by_id=chunk_text_by_id,
            chunk_section_by_id=chunk_section_by_id,
            chunk_subsection_by_id=chunk_subsection_by_id,
        )
        self._global_cache[key] = loaded
        self._global_cache_sig[key] = sig
        return loaded

    def search(
        self,
        data_dir: Path,
        question: str,
        k: int,
        query_id: Optional[str] = None,
        include_generated_answer: bool = False,
        retrieval_scope: str = "doc",
        lexical_scope: str = "doc",
        filters: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Return top-k chunks with pages/scores and expected-page highlight metadata.

        retrieval_scope:
        - doc: dense retrieval on selected document index
        - trust: dense retrieval on global index, restricted to selected doc trust
        - global: dense retrieval on global index

        lexical_scope:
        - doc/trust/global: BM25 corpus scope used for fusion
        """
        loaded_doc = self._load_doc(data_dir)
        scope = str(retrieval_scope or "doc").strip().lower()
        lex_scope = str(lexical_scope or "doc").strip().lower()
        if scope not in {"doc", "trust", "global"}:
            scope = "doc"
        if lex_scope not in {"doc", "trust", "global"}:
            lex_scope = "doc"

        selected_doc_id = str(data_dir.name)
        selected_trust_id = self._trust_from_doc_id(selected_doc_id)

        if scope == "doc":
            scope_meta = loaded_doc.meta.reset_index(drop=True)
            scope_index = loaded_doc.index
            chunk_text_by_id = loaded_doc.chunk_text_by_id
            chunk_section_by_id = loaded_doc.chunk_section_by_id
            chunk_subsection_by_id = loaded_doc.chunk_subsection_by_id
            base_mask = np.ones((len(scope_meta),), dtype=bool)
        else:
            loaded_global = self._load_global(data_dir.parent)
            scope_meta = loaded_global.meta.reset_index(drop=True)
            scope_index = loaded_global.index
            chunk_text_by_id = loaded_global.chunk_text_by_id
            chunk_section_by_id = loaded_global.chunk_section_by_id
            chunk_subsection_by_id = loaded_global.chunk_subsection_by_id
            base_mask = np.ones((len(scope_meta),), dtype=bool)
            if scope == "trust" and "trust_id" in scope_meta.columns:
                base_mask &= (
                    scope_meta["trust_id"].fillna("").astype(str).str.lower().values
                    == str(selected_trust_id).lower()
                )

        user_mask = self._apply_filters(scope_meta, filters)
        candidate_mask = base_mask & user_mask
        if not np.any(candidate_mask):
            candidate_mask = base_mask

        candidate_indices = np.where(candidate_mask)[0].tolist()
        if not candidate_indices:
            raise RuntimeError("No candidates available after metadata filtering.")

        emb = self.model.encode([question], convert_to_numpy=True, normalize_embeddings=False).astype("float32")
        emb = l2_normalize(emb).astype("float32")

        k = max(1, min(int(k), len(candidate_indices)))
        k_search = min(max(MAX_K_SEARCH, k * 20), len(scope_meta))
        dense_scores, dense_idxs = scope_index.search(emb, k_search)
        dense_ranked_all = dense_idxs[0].tolist()
        dense_score_vals_all = dense_scores[0].tolist()
        cand_set = set(candidate_indices)
        dense_ranked: list[int] = []
        dense_score_vals: list[float] = []
        for idx, score in zip(dense_ranked_all, dense_score_vals_all):
            if idx in cand_set:
                dense_ranked.append(int(idx))
                dense_score_vals.append(float(score))
            if len(dense_ranked) >= max(k_search, k):
                break
        if len(dense_ranked) < k:
            dense_scores2, dense_idxs2 = scope_index.search(emb, len(scope_meta))
            dense_ranked = []
            dense_score_vals = []
            for idx, score in zip(dense_idxs2[0].tolist(), dense_scores2[0].tolist()):
                if idx in cand_set:
                    dense_ranked.append(int(idx))
                    dense_score_vals.append(float(score))
                if len(dense_ranked) >= max(k_search, k):
                    break

        dense_rank_map: dict[int, int] = {idx: r for r, idx in enumerate(dense_ranked, start=1)}
        dense_score_map: dict[int, float] = {
            idx: float(score) for idx, score in zip(dense_ranked, dense_score_vals)
        }

        bm25_candidate_mask = candidate_mask.copy()
        if lex_scope == "doc" and "doc_id" in scope_meta.columns:
            bm25_candidate_mask &= (
                scope_meta["doc_id"].fillna("").astype(str).values == selected_doc_id
            )
        elif lex_scope == "trust" and "trust_id" in scope_meta.columns:
            bm25_candidate_mask &= (
                scope_meta["trust_id"].fillna("").astype(str).str.lower().values
                == str(selected_trust_id).lower()
            )
        bm25_candidate_indices = np.where(bm25_candidate_mask)[0].tolist()
        if not bm25_candidate_indices:
            bm25_candidate_indices = candidate_indices

        bm25_corpus: list[str] = []
        for idx in bm25_candidate_indices:
            row = scope_meta.iloc[idx]
            cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
            bm25_corpus.append(chunk_text_by_id.get(cid, ""))
        bm25 = BM25Index([tokenize(t) for t in bm25_corpus], k1=1.5, b=0.75)
        bm25_scores_local = bm25.score_query(tokenize(question))
        bm25_ranked_pairs_local = sorted(
            enumerate(bm25_scores_local),
            key=lambda x: x[1],
            reverse=True,
        )[: max(k_search, k)]
        bm25_ranked_pairs = [(bm25_candidate_indices[i], score) for i, score in bm25_ranked_pairs_local]
        bm25_ranked = [idx for idx, _ in bm25_ranked_pairs]
        bm25_rank_map: dict[int, int] = {idx: r for r, idx in enumerate(bm25_ranked, start=1)}
        bm25_score_map: dict[int, float] = {idx: float(score) for idx, score in bm25_ranked_pairs}

        fused_ranked, fused_scores = rrf_fuse(
            dense_ranked=dense_ranked,
            bm25_ranked=bm25_ranked,
            rrf_k=RRF_K,
            dense_weight=RRF_DENSE_WEIGHT,
            bm25_weight=RRF_BM25_WEIGHT,
        )
        scores_map: dict[int, float] = dict(fused_scores)
        if self.cross_encoder is not None and fused_ranked:
            ce_topn = min(len(fused_ranked), self.cross_encoder_topn)
            cand = fused_ranked[:ce_topn]
            pairs: list[tuple[str, str]] = []
            for idx in cand:
                row = scope_meta.iloc[idx]
                cid = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
                pairs.append((question, chunk_text_by_id.get(cid, "")))
            ce_scores_raw = np.asarray(self.cross_encoder.predict(pairs), dtype=np.float32)
            if ce_scores_raw.size:
                lo = float(np.min(ce_scores_raw))
                hi = float(np.max(ce_scores_raw))
                if hi > lo:
                    ce_scores = ((ce_scores_raw - lo) / (hi - lo)).astype(np.float32)
                else:
                    ce_scores = np.zeros_like(ce_scores_raw, dtype=np.float32)
                for idx, ce_s in zip(cand, ce_scores.tolist()):
                    scores_map[idx] = float(scores_map.get(idx, 0.0)) + self.cross_encoder_weight * float(ce_s)
                fused_ranked = sorted(fused_ranked, key=lambda i: scores_map.get(i, 0.0), reverse=True)
        idx_list = fused_ranked[:k]

        expected_pages: list[int] = []
        expected_answer: Optional[str] = None
        if query_id:
            for item in loaded_doc.eval_items:
                if str(item.get("query_id", "")).strip() == query_id:
                    expected_pages = [int(x) for x in item.get("expected_pages", []) if str(x).isdigit()]
                    raw_expected = item.get("expected_answer")
                    if raw_expected is not None:
                        expected_answer = str(raw_expected)
                    break

        results: list[dict[str, Any]] = []
        retrieved_pages: list[int] = []
        for rank, idx in enumerate(idx_list, start=1):
            row = scope_meta.iloc[idx]
            chunk_id = str(row.get("chunk_id_global") or row.get("chunk_id") or "")
            pages = to_pages_list(row.get("pages"))
            if not pages:
                ps = row.get("page_start")
                pe = row.get("page_end")
                if pd.notna(ps):
                    try:
                        pages.append(int(ps))
                    except Exception:
                        pass
                if pd.notna(pe):
                    try:
                        pe_i = int(pe)
                        if pe_i not in pages:
                            pages.append(pe_i)
                    except Exception:
                        pass
            retrieved_pages.extend(pages)
            snippet = chunk_text_by_id.get(chunk_id, "")
            full_chunk_text = snippet
            if len(snippet) > 280:
                snippet = snippet[:280].rstrip() + " ..."
            hit_expected = bool(expected_pages and any(p in expected_pages for p in pages))
            section_title = str(row.get("section_title") or "")
            subsection_title = str(row.get("subsection_title") or "")
            if not subsection_title:
                subsection_title = chunk_subsection_by_id.get(chunk_id, "")
            if not section_title:
                section_title = chunk_section_by_id.get(chunk_id, "")
            results.append(
                {
                    "rank": rank,
                    "chunk_id": chunk_id,
                    "pages": pages,
                    "score": float(scores_map.get(idx, 0.0)),
                    "rrf_score": float(scores_map.get(idx, 0.0)),
                    "dense_rank": int(dense_rank_map.get(idx, 0)),
                    "bm25_rank": int(bm25_rank_map.get(idx, 0)),
                    "dense_raw_score": float(dense_score_map.get(idx, 0.0)),
                    "bm25_raw_score": float(bm25_score_map.get(idx, 0.0)),
                    "section_title": section_title,
                    "subsection_title": subsection_title,
                    "snippet": snippet,
                    "chunk_text": full_chunk_text,
                    "hit_expected_page": hit_expected,
                    "is_table": bool(row.get("is_table", False)),
                }
            )

        hit_at_k = bool(expected_pages and any(p in expected_pages for p in retrieved_pages))
        predicted_answer, answer_source_chunk_id, answer_debug = self._predict_answer(
            question=question,
            results=results,
            query_id=query_id,
            expected_pages=expected_pages,
        )
        if include_generated_answer:
            raw_generated_answer, generation_debug = self._generate_local_answer(
                question=question,
                results=results,
            )
            parsed_answer, raw_citations_from_json, parse_mode = self._parse_generation_json_payload(
                str(raw_generated_answer or "")
            )
            raw_citations = list(raw_citations_from_json)
            if not raw_citations:
                raw_citations = self._extract_citations_from_answer(str(raw_generated_answer or ""))
            generated_answer = parsed_answer
            valid_citations, rejected_citations = self._validate_citations(raw_citations, results)
            raw_status = str(generation_debug.get("status") or "").strip().lower()
            text_answer = str(generated_answer or "").strip()
            if raw_status != "ok" or not text_answer:
                generation_status = raw_status or "error"
                generated_answer = None
                generated_citations: list[dict[str, Any]] = []
                generation_confidence: Optional[float] = None
            elif text_answer.lower() == "insufficient evidence in retrieved context.":
                generation_status = "insufficient_evidence"
                generated_answer = None
                generated_citations = []
                generation_confidence = None
            elif not valid_citations:
                generation_status = "insufficient_evidence"
                generated_answer = None
                generated_citations = []
                generation_confidence = None
            else:
                generation_status = "ok"
                generated_citations = valid_citations
                generation_confidence = float(len(valid_citations)) / float(max(1, len(raw_citations)))
            generation_debug["citations_parsed"] = int(len(raw_citations))
            generation_debug["citations_valid"] = int(len(valid_citations))
            generation_debug["citations_rejected"] = int(rejected_citations)
            generation_debug["parse_mode"] = parse_mode
        else:
            generated_answer = None
            generation_debug = {
                "provider": "local_ollama",
                "status": "skipped",
                "model": getattr(self.local_llm, "model", ""),
                "error": None,
                "prompt_chars": 0,
                "reason": "include_generated_answer=false",
            }
            generated_citations = []
            generation_status = "skipped"
            generation_confidence = None

        top1_score = float(results[0].get("score", 0.0)) if results else None
        top2_score = float(results[1].get("score", 0.0)) if len(results) > 1 else None
        retrieval_margin = (top1_score - top2_score) if (top1_score is not None and top2_score is not None) else None
        low_retrieval_margin = bool(
            retrieval_margin is not None and retrieval_margin < float(RETRIEVAL_MARGIN_LOW_THRESHOLD)
        )
        generation_debug["retrieval_margin"] = retrieval_margin
        generation_debug["retrieval_margin_threshold"] = float(RETRIEVAL_MARGIN_LOW_THRESHOLD)
        generation_debug["low_retrieval_margin"] = low_retrieval_margin

        self._update_generation_observability(
            generation_status=str(generation_status),
            citations_parsed=int(generation_debug.get("citations_parsed", 0) or 0),
            citations_valid=int(generation_debug.get("citations_valid", 0) or 0),
            citations_rejected=int(generation_debug.get("citations_rejected", 0) or 0),
            latency_ms=float(generation_debug.get("latency_ms", 0.0) or 0.0),
        )
        return {
            "question": question,
            "k": k,
            "retrieval_mode": "hybrid_rrf_dense_bm25",
            "retrieval_config": {
                "rrf_k": int(RRF_K),
                "dense_weight": float(RRF_DENSE_WEIGHT),
                "bm25_weight": float(RRF_BM25_WEIGHT),
                "enable_cross_encoder_rerank": bool(self.cross_encoder is not None),
                "cross_encoder_model": (str(CROSS_ENCODER_MODEL_NAME) if self.cross_encoder is not None else None),
                "cross_encoder_topn": int(self.cross_encoder_topn),
                "cross_encoder_weight": float(self.cross_encoder_weight),
                "bm25_k1": 1.5,
                "bm25_b": 0.75,
            },
            "retrieval_scope": scope,
            "lexical_scope": lex_scope,
            "filters_applied": filters or {},
            "query_id": query_id,
            "expected_pages": expected_pages,
            "expected_answer": expected_answer,
            "hit_at_k": hit_at_k,
            "predicted_answer": predicted_answer,
            "answer_source_chunk_id": answer_source_chunk_id,
            "answer_debug": answer_debug,
            "include_generated_answer": bool(include_generated_answer),
            "generated_answer": generated_answer,
            "generated_citations": generated_citations,
            "generation_status": generation_status,
            "generation_confidence": generation_confidence,
            "generation_debug": generation_debug,
            "results": results,
        }

    def _extract_entity_terms(self, q: str) -> list[str]:
        stop = {
            "what", "which", "who", "where", "when", "why", "how",
            "did", "does", "do", "was", "were", "is", "are", "the",
            "for", "from", "with", "during", "year", "reported",
            "report", "deficit", "surplus", "overspend", "shortfall",
            "overall", "amount", "total", "value",
        }
        toks = re.findall(r"[a-z][a-z0-9\-]{2,}", str(q or "").lower())
        terms = [t for t in toks if t not in stop]
        # Preserve order and keep only first few high-signal terms.
        out: list[str] = []
        seen: set[str] = set()
        for t in terms:
            if t not in seen:
                seen.add(t)
                out.append(t)
            if len(out) >= 4:
                break
        return out

    def _split_entity_like_segments(self, text: str) -> list[str]:
        """
        Split text into entity-like segments using generic IJB/board markers.
        Falls back to sentence-like windows if no explicit markers are present.
        """
        t = str(text or "")
        if not t:
            return []
        pattern = re.compile(
            r"(?i)\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\s+)?"
            r"(?:integration\s+joint\s+board\s*\(ijb\)|ijb)\b"
        )
        starts = [m.start() for m in pattern.finditer(t)]
        if not starts:
            return [w.strip() for w in re.split(r"(?<=[.!?])\s+|\n+", t) if w.strip()]
        starts = sorted(set([0] + starts))
        segs: list[str] = []
        for i, s in enumerate(starts):
            e = starts[i + 1] if i + 1 < len(starts) else len(t)
            seg = t[s:e].strip()
            if seg:
                segs.append(seg)
        return segs

    def _extract_answer_single_chunk(self, question: str, text: str) -> Optional[str]:
        q = str(question or "").lower()
        text = str(text or "").strip()
        if not text:
            return None
        is_percent_question = any(k in q for k in ("percent", "percentage", "%"))
        if is_percent_question:
            m = re.search(r"\d+(?:\.\d+)?%", text)
            if m:
                return m.group(0).strip()
            return None

        if any(k in q for k in ("deficit", "surplus", "overspend", "shortfall", "cost", "expenditure")):
            # For money-like questions, prioritize currency/amount patterns and avoid hyphen-word artifacts
            # such as "COVID-19".
            for patt in (
                r"£\s?\d[\d,]*(?:\.\d+)?\s*(?:million|billion|bn|m)?",
                r"(?<![A-Za-z0-9])(?:£\s*)?[-–]\s?(?:\d{1,3}(?:,\d{3})+|\d{4,})(?:\.\d+)?",
                r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",
            ):
                vals = re.findall(patt, text, flags=re.IGNORECASE)
                for v in vals:
                    cand = str(v).replace("–", "-").strip()
                    clean = re.sub(r"^(£|\$)\s*", "", cand)
                    clean = re.sub(r"\s*(million|billion|bn|m)\s*$", "", clean, flags=re.IGNORECASE)
                    clean = clean.replace(",", "").replace(" ", "")
                    if clean and clean not in {"0", "00", "000", "-0", "-00", "-000"} and re.search(r"\d", clean):
                        return cand

        numeric_cues = (
            "amount",
            "total",
            "value",
            "how much",
            "how many",
            "£",
            "million",
            "billion",
            "ratio",
            "rate",
            "number of",
        )
        if any(c in q for c in numeric_cues):
            m = re.search(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b", text)
            if m:
                return m.group(0).strip()
        return None

    def _extract_numeric_candidates(self, question: str, text: str) -> list[tuple[str, int, int]]:
        """
        Return numeric candidates as (value, start, end) with light question-aware patterns.
        """
        q = str(question or "").lower()
        t = str(text or "")
        out: list[tuple[str, int, int]] = []
        pats: list[str] = []
        if any(k in q for k in ("percent", "percentage", "%")):
            pats.append(r"\d+(?:\.\d+)?%")
        if any(k in q for k in ("deficit", "surplus", "overspend", "shortfall", "cost", "expenditure", "amount", "total", "value")):
            pats.extend(
                [
                    r"£\s?\d[\d,]*(?:\.\d+)?\s*(?:million|billion|bn|m)?",
                    r"(?<![A-Za-z0-9])(?:£\s*)?[-–]\s?(?:\d{1,3}(?:,\d{3})+|\d{4,})(?:\.\d+)?",
                    r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\s*(?:million|billion|bn|m)?\b",
                ]
            )
        if not pats:
            pats.append(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b")

        seen: set[tuple[int, int]] = set()
        for patt in pats:
            for m in re.finditer(patt, t, flags=re.IGNORECASE):
                span = (m.start(), m.end())
                if span in seen:
                    continue
                seen.add(span)
                cand = m.group(0).strip().replace("–", "-")
                out.append((cand, m.start(), m.end()))
        return out

    def _gate_candidate_score(
        self,
        question: str,
        full_text: str,
        candidate_span: tuple[int, int],
        entity_terms: list[str],
    ) -> tuple[int, dict[str, bool]]:
        """
        Lightweight logical gate score for a numeric candidate:
        - entity near number
        - semantic cue near number (deficit/surplus/etc)
        - unit/currency hint near number
        """
        t = str(full_text or "")
        q = str(question or "").lower()
        s, e = candidate_span
        w = max(40, int(ANSWER_GATE_WINDOW_CHARS))
        lo = max(0, s - w)
        hi = min(len(t), e + w)
        local = t[lo:hi].lower()

        entity_near = any(et in local for et in entity_terms if et and et != "ijb")
        cue_terms = ("deficit", "surplus", "overspend", "shortfall", "cash requirement", "resource limit")
        cue_near = any(ct in local for ct in cue_terms if ct in q or ct in local)
        unit_near = any(u in local for u in ("£", "million", "billion", "%", "(£000)", "£000"))
        bad_context = any(b in local for b in ("retained reserves", "earmarked reserves"))

        score = int(entity_near) + int(cue_near) + int(unit_near) - int(bad_context)
        return score, {
            "entity_near": entity_near,
            "cue_near": cue_near,
            "unit_near": unit_near,
            "bad_context": bad_context,
        }

    def _extract_entity_anchors(self, text: str) -> list[tuple[str, int]]:
        """
        Extract generic entity anchors from text, e.g. '<Name> IJB' / '<Name> Integration Joint Board'.
        """
        t = str(text or "")
        anchors: list[tuple[str, int]] = []
        patt = re.compile(
            r"\b([A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3})\s+"
            r"(?:Integration\s+Joint\s+Board(?:\s*\(IJB\))?|IJB)\b",
            flags=re.IGNORECASE,
        )
        for m in patt.finditer(t):
            label = str(m.group(1)).strip().lower()
            if label:
                anchors.append((label, int(m.start())))
        return anchors

    def _entity_consistent_pick(
        self,
        question: str,
        text: str,
        target_terms: list[str],
    ) -> tuple[Optional[str], dict[str, Any]]:
        """
        Pick numeric answer by enforcing entity-to-number consistency.

        A candidate is preferred when:
        - nearest anchor to the number matches the target entity term(s)
        - cue terms (deficit/surplus/overspend/shortfall) appear near the number
        - unit/currency hints appear near the number
        """
        t = str(text or "")
        if not t:
            return None, {"applied": False, "reason": "no_text"}
        hard_terms = [x for x in target_terms if x and x != "ijb"]
        if not hard_terms:
            return None, {"applied": False, "reason": "no_target_terms"}

        anchors = self._extract_entity_anchors(t)
        if not anchors:
            return None, {"applied": False, "reason": "no_entity_anchors"}

        candidates = self._extract_numeric_candidates(question=question, text=t)
        if not candidates:
            return None, {"applied": False, "reason": "no_numeric_candidates"}

        q = str(question or "").lower()
        cue_terms = ("deficit", "surplus", "overspend", "shortfall")
        best_row: Optional[tuple[int, int, str, dict[str, Any]]] = None
        # (score, -anchor_distance, value, debug)

        for cand, s, e in candidates:
            nearest: Optional[tuple[str, int]] = None
            nearest_target: Optional[tuple[str, int]] = None
            for label, pos in anchors:
                d = abs(int(s) - int(pos))
                if nearest is None or d < nearest[1]:
                    nearest = (label, d)
                if any(tt in label for tt in hard_terms):
                    if nearest_target is None or d < nearest_target[1]:
                        nearest_target = (label, d)

            if nearest is None:
                continue

            nearest_label, nearest_d = nearest
            target_d = int(nearest_target[1]) if nearest_target is not None else 10**9
            target_match = any(tt in nearest_label for tt in hard_terms)
            competing_entity_closer = bool(nearest_target is not None and nearest_d < target_d and not target_match)

            w = max(60, int(ENTITY_CONSISTENCY_WINDOW_CHARS))
            lo = max(0, int(s) - w)
            hi = min(len(t), int(e) + w)
            local = t[lo:hi].lower()
            cue_near = any(ct in local for ct in cue_terms if ct in q or ct in local)
            unit_near = any(u in local for u in ("£", "million", "billion", "%", "(£000)", "£000"))

            # Hard filters first.
            if nearest_d > max(80, int(ENTITY_CONSISTENCY_MAX_ANCHOR_DISTANCE)):
                continue
            if not target_match:
                continue
            if competing_entity_closer:
                continue

            score = (4 if target_match else 0) + (2 if cue_near else 0) + (1 if unit_near else 0)
            if not cue_near:
                score -= 1

            dbg = {
                "candidate": cand,
                "candidate_start": int(s),
                "nearest_label": nearest_label,
                "nearest_distance": int(nearest_d),
                "target_distance": int(target_d) if target_d < 10**9 else None,
                "target_match": bool(target_match),
                "cue_near": bool(cue_near),
                "unit_near": bool(unit_near),
                "score": int(score),
            }
            row = (int(score), -int(nearest_d), cand, dbg)
            if best_row is None or row > best_row:
                best_row = row

        if best_row is None:
            return None, {"applied": True, "reason": "no_entity_consistent_candidate"}
        return best_row[2], {"applied": True, "best": best_row[3]}

    def _anchor_distance_pick(
        self,
        question: str,
        text: str,
        target_terms: list[str],
    ) -> tuple[Optional[str], dict[str, Any]]:
        """
        Anchor-distance extraction:
        1) find target anchor in text
        2) find numeric candidates
        3) choose nearest positive candidate after anchor
        4) hard-gate by max distance and competing entity anchors
        """
        t = str(text or "")
        tl = t.lower()
        hard_terms = [x for x in target_terms if x and x != "ijb"]
        if not t or not hard_terms:
            return None, {"applied": False, "reason": "no_text_or_target_terms"}

        # Target anchor index from query terms.
        target_pos: Optional[int] = None
        target_term_used = ""
        for tt in hard_terms:
            p = tl.find(tt)
            if p >= 0 and (target_pos is None or p < target_pos):
                target_pos = p
                target_term_used = tt
        if target_pos is None:
            return None, {"applied": False, "reason": "target_not_found"}

        cands = self._extract_numeric_candidates(question=question, text=t)
        if not cands:
            return None, {"applied": True, "reason": "no_numeric_candidates", "target_pos": target_pos}

        # Keep candidates that follow target anchor.
        post = [(val, s, e, (s - target_pos)) for (val, s, e) in cands if s > target_pos]
        if not post:
            return None, {"applied": True, "reason": "no_positive_distance_candidate", "target_pos": target_pos}

        # Candidate nearest positive distance.
        post.sort(key=lambda x: x[3])
        best_val, best_s, best_e, best_d = post[0]
        debug = {
            "applied": True,
            "target_term": target_term_used,
            "target_pos": int(target_pos),
            "best_distance": int(best_d),
            "candidate_value": best_val,
        }

        # Hard gate: too far from target.
        if best_d > max(50, int(ANCHOR_DISTANCE_MAX_CHARS)):
            debug["hard_fail"] = "distance_too_large"
            return None, debug

        # Hard gate: another known entity anchor is closer to the candidate than target anchor.
        anchors = self._extract_entity_anchors(t)
        nearest_other: Optional[tuple[str, int]] = None
        for label, pos in anchors:
            if target_term_used in label:
                continue
            d = abs(best_s - pos)
            if nearest_other is None or d < nearest_other[1]:
                nearest_other = (label, d)
        if nearest_other is not None and int(nearest_other[1]) < int(best_d):
            debug["hard_fail"] = "other_entity_closer"
            debug["other_entity_label"] = nearest_other[0]
            debug["other_entity_distance"] = int(nearest_other[1])
            return None, debug

        # Soft cue check in local span from target to candidate.
        lo = max(0, target_pos)
        hi = min(len(t), best_e + 60)
        between = tl[lo:hi]
        if not any(k in between for k in ("deficit", "surplus", "overspend", "shortfall")):
            debug["hard_fail"] = "missing_cue_between_anchor_and_value"
            return None, debug

        debug["pass"] = True
        return best_val, debug

    def _predict_answer(
        self,
        question: str,
        results: list[dict[str, Any]],
        query_id: Optional[str] = None,
        expected_pages: Optional[list[int]] = None,
    ) -> tuple[Optional[str], Optional[str], dict[str, Any]]:
        """
        Heuristically predict an answer from top-k retrieved chunks.

        This is a lightweight debug extractor, not a full QA model.
        """
        if not results:
            return None, None, {"strategy": "topk_entity_aware", "reason": "no_results"}
        q = str(question or "").lower()
        entity_terms = self._extract_entity_terms(q)
        hard_entity_terms = [t for t in entity_terms if t != "ijb"]
        deficit_like = any(k in q for k in ("deficit", "surplus", "overspend", "shortfall"))

        def _entity_match(text: str) -> bool:
            if not hard_entity_terms:
                return True
            tx = str(text or "").lower()
            return all(t in tx for t in hard_entity_terms)

        base_order = list(results)
        ordering_notes: list[str] = []

        if ANSWER_EVAL_STRICT_PAGE and query_id and expected_pages:
            expected_set = set(int(x) for x in expected_pages)
            on_expected = []
            off_expected = []
            for r in base_order:
                rp = set(int(x) for x in (r.get("pages") or []) if str(x).isdigit())
                if expected_set.intersection(rp):
                    on_expected.append(r)
                else:
                    off_expected.append(r)
            if on_expected:
                base_order = on_expected + off_expected
                ordering_notes.append("prefer_expected_pages")

        if ANSWER_PREFER_NON_TABLE:
            non_table = [r for r in base_order if not bool(r.get("is_table", False))]
            table = [r for r in base_order if bool(r.get("is_table", False))]
            if non_table:
                base_order = non_table + table
                ordering_notes.append("prefer_non_table")

        entity_first = [r for r in base_order if _entity_match(r.get("chunk_text", ""))]
        entity_rest = [r for r in base_order if r not in entity_first]
        candidate_order = entity_first + entity_rest

        # Pass 1 (strict): for deficit-like questions, extract only from windows that
        # explicitly mention a deficit cue; prioritize windows with entity terms.
        if deficit_like:
            def _windows(text: str) -> list[str]:
                return [w.strip() for w in re.split(r"(?<=[.!?])\s+|\n+", str(text or "")) if w.strip()]

            def _has_deficit_cue(text: str) -> bool:
                tx = str(text or "").lower()
                return any(k in tx for k in ("deficit", "surplus", "overspend", "shortfall"))

            # 0) Entity-consistency gate: pick number whose nearest entity anchor
            # matches the query entity terms.
            if ENTITY_CONSISTENCY_GATE_ENABLED and hard_entity_terms:
                best: Optional[tuple[int, int, str, dict[str, Any], dict[str, Any]]] = None
                # (score, -rank, answer, gate_debug, selected_meta)
                for r in candidate_order:
                    text = str(r.get("chunk_text") or "")
                    ans, dbg = self._entity_consistent_pick(
                        question=question,
                        text=text,
                        target_terms=entity_terms,
                    )
                    if not ans:
                        continue
                    score = int((dbg.get("best") or {}).get("score", 0))
                    rank = int(r.get("rank") or 9999)
                    row = (
                        score,
                        -rank,
                        ans,
                        dbg,
                        {
                            "selected_rank": rank,
                            "selected_chunk_id": str(r.get("chunk_id") or ""),
                        },
                    )
                    if best is None or row > best:
                        best = row
                if best is not None:
                    _, _, ans, dbg, meta = best
                    return (
                        ans,
                        meta["selected_chunk_id"] or None,
                        {
                            "strategy": "topk_entity_consistency_gate",
                            "ordering_notes": ordering_notes,
                            "entity_terms": entity_terms,
                            "selected_rank": int(meta["selected_rank"]),
                            "selected_chunk_id": str(meta["selected_chunk_id"]),
                            "entity_consistency_gate": dbg,
                            "entity_consistency_window_chars": int(ENTITY_CONSISTENCY_WINDOW_CHARS),
                        },
                    )

            # 1) Anchor-distance gate (lightweight, query/entity-aware).
            if ANCHOR_DISTANCE_GATE_ENABLED and hard_entity_terms:
                viable: list[tuple[int, int, str, dict[str, Any], dict[str, Any]]] = []
                # (distance, rank, answer, gate_debug, meta)
                hard_fail_count = 0
                for r in candidate_order:
                    text = str(r.get("chunk_text") or "")
                    ans, dbg = self._anchor_distance_pick(
                        question=question,
                        text=text,
                        target_terms=entity_terms,
                    )
                    if ans is not None:
                        dist = int(dbg.get("best_distance", 10**9))
                        rank = int(r.get("rank") or 9999)
                        viable.append(
                            (
                                dist,
                                rank,
                                ans,
                                dbg,
                                {
                                    "selected_rank": rank,
                                    "selected_chunk_id": str(r.get("chunk_id") or ""),
                                },
                            )
                        )
                    elif dbg.get("hard_fail"):
                        hard_fail_count += 1

                if viable:
                    viable.sort(key=lambda x: (x[0], x[1]))
                    _, _, ans, dbg, meta = viable[0]
                    return (
                        ans,
                        meta["selected_chunk_id"] or None,
                        {
                            "strategy": "topk_anchor_distance_gate",
                            "ordering_notes": ordering_notes,
                            "entity_terms": entity_terms,
                            "selected_rank": int(meta["selected_rank"]),
                            "selected_chunk_id": str(meta["selected_chunk_id"]),
                            "anchor_gate": dbg,
                            "anchor_max_chars": int(ANCHOR_DISTANCE_MAX_CHARS),
                        },
                    )
                if ANCHOR_DISTANCE_STRICT_NULL and hard_fail_count > 0:
                    return (
                        None,
                        None,
                        {
                            "strategy": "topk_anchor_distance_gate_null",
                            "ordering_notes": ordering_notes,
                            "entity_terms": entity_terms,
                            "reason": "hard_gate_failed_all_candidates",
                            "anchor_max_chars": int(ANCHOR_DISTANCE_MAX_CHARS),
                        },
                    )

            # 0) Strict query-driven segment pass: find entity-like segment that best matches query terms.
            if hard_entity_terms:
                for r in candidate_order:
                    text = str(r.get("chunk_text") or "")
                    segs = self._split_entity_like_segments(text)
                    # Score segments by query term overlap; then extract from best segment first.
                    seg_scored = sorted(
                        segs,
                        key=lambda s: sum(1 for t in hard_entity_terms if t in s.lower()),
                        reverse=True,
                    )
                    for seg in seg_scored:
                        seg_l = seg.lower()
                        if sum(1 for t in hard_entity_terms if t in seg_l) == 0:
                            continue
                        cue_match = re.search(r"(deficit|surplus|overspend|shortfall)", seg_l)
                        if not cue_match:
                            continue
                        tail = seg[cue_match.start() : min(len(seg), cue_match.start() + 220)]
                        ans = self._extract_answer_single_chunk(question=question, text=tail)
                        if ans:
                            return (
                                ans,
                                str(r.get("chunk_id") or "") or None,
                                {
                                    "strategy": "topk_query_driven_segment",
                                    "ordering_notes": ordering_notes,
                                    "entity_terms": entity_terms,
                                    "selected_rank": int(r.get("rank") or 0),
                                    "selected_chunk_id": str(r.get("chunk_id") or ""),
                                },
                            )

            # 0) Entity phrase pass: prefer "<entity> ... reported ... deficit ... £X" patterns.
            if hard_entity_terms:
                for r in candidate_order:
                    text = str(r.get("chunk_text") or "")
                    for ent in hard_entity_terms:
                        patt = (
                            rf"{re.escape(ent)}"
                            rf"[\s\S]{{0,220}}?"
                            rf"(?:reported|reports|overall)?"
                            rf"[\s\S]{{0,140}}?"
                            rf"(?:deficit|surplus|overspend|shortfall)"
                            rf"[\s\S]{{0,180}}?"
                            rf"((?:£\s*)?[-–]?\s?\d[\d,]*(?:\.\d+)?\s*(?:million|billion|bn|m)?)"
                        )
                        m = re.search(patt, text, flags=re.IGNORECASE)
                        if not m:
                            continue
                        ans = self._extract_answer_single_chunk(question=question, text=m.group(0))
                        if ans:
                            return (
                                ans,
                                str(r.get("chunk_id") or "") or None,
                                {
                                    "strategy": "topk_entity_phrase_pattern",
                                    "ordering_notes": ordering_notes,
                                    "entity_terms": entity_terms,
                                    "selected_rank": int(r.get("rank") or 0),
                                    "selected_chunk_id": str(r.get("chunk_id") or ""),
                                },
                            )

            # 0) Entity-anchored spans: extract from the local context around entity mentions.
            if hard_entity_terms:
                for r in candidate_order:
                    text = str(r.get("chunk_text") or "")
                    for seg in self._split_entity_like_segments(text):
                        seg_l = seg.lower()
                        if not any(t in seg_l for t in hard_entity_terms):
                            continue
                        cue_pos = min(
                            (
                                p
                                for p in (
                                    seg_l.find("deficit"),
                                    seg_l.find("surplus"),
                                    seg_l.find("overspend"),
                                    seg_l.find("shortfall"),
                                )
                                if p >= 0
                            ),
                            default=-1,
                        )
                        if cue_pos < 0:
                            continue
                        lo = max(0, cue_pos - 45)
                        hi = min(len(seg), cue_pos + 180)
                        cue_window = seg[lo:hi]
                        ans = self._extract_answer_single_chunk(question=question, text=cue_window)
                        if ans:
                            return (
                                ans,
                                str(r.get("chunk_id") or "") or None,
                                {
                                    "strategy": "topk_entity_anchor_span",
                                    "ordering_notes": ordering_notes,
                                    "entity_terms": entity_terms,
                                    "selected_rank": int(r.get("rank") or 0),
                                    "selected_chunk_id": str(r.get("chunk_id") or ""),
                                },
                            )

            # 0b) Logical gate before generic regex selection.
            if ANSWER_GATE_ENABLED:
                best: Optional[tuple[int, int, str, dict[str, bool], dict[str, Any]]] = None
                # (gate_score, -rank, answer, gate_flags, selected_meta)
                for r in candidate_order:
                    text = str(r.get("chunk_text") or "")
                    candidates = self._extract_numeric_candidates(question=question, text=text)
                    for cand, s, e in candidates:
                        gate_score, gate_flags = self._gate_candidate_score(
                            question=question,
                            full_text=text,
                            candidate_span=(s, e),
                            entity_terms=entity_terms,
                        )
                        if any(et and et != "ijb" for et in entity_terms) and not gate_flags.get("entity_near", False):
                            continue
                        if gate_score < max(1, ANSWER_GATE_MIN_SCORE):
                            continue
                        rank = int(r.get("rank") or 9999)
                        row = (
                            gate_score,
                            -rank,
                            cand,
                            gate_flags,
                            {
                                "selected_rank": rank,
                                "selected_chunk_id": str(r.get("chunk_id") or ""),
                            },
                        )
                        if best is None or row > best:
                            best = row
                if best is not None:
                    _, _, cand, gate_flags, meta = best
                    return (
                        cand,
                        meta["selected_chunk_id"] or None,
                        {
                            "strategy": "topk_logical_gate",
                            "ordering_notes": ordering_notes,
                            "entity_terms": entity_terms,
                            "selected_rank": int(meta["selected_rank"]),
                            "selected_chunk_id": str(meta["selected_chunk_id"]),
                            "gate": gate_flags,
                            "gate_window_chars": int(ANSWER_GATE_WINDOW_CHARS),
                            "gate_min_score": int(ANSWER_GATE_MIN_SCORE),
                        },
                    )

            # 1a) entity+cue windows first
            for r in candidate_order:
                text = str(r.get("chunk_text") or "")
                for w in _windows(text):
                    if not _has_deficit_cue(w):
                        continue
                    if hard_entity_terms and not all(t in w.lower() for t in hard_entity_terms):
                        continue
                    # Anchor extraction near entity mention inside the window to avoid
                    # picking other entities' values in long flattened OCR/table text.
                    w_l = w.lower()
                    anchored = w
                    if hard_entity_terms:
                        ent_positions = [w_l.find(t) for t in hard_entity_terms if t in w_l]
                        ent_positions = [p for p in ent_positions if p >= 0]
                        if ent_positions:
                            st = min(ent_positions)
                            anchored = w[st : min(len(w), st + 320)]
                    ans = self._extract_answer_single_chunk(question=question, text=anchored)
                    if ans:
                        return (
                            ans,
                            str(r.get("chunk_id") or "") or None,
                            {
                                "strategy": "topk_entity_deficit_window",
                                "ordering_notes": ordering_notes,
                                "entity_terms": entity_terms,
                                "selected_rank": int(r.get("rank") or 0),
                                "selected_chunk_id": str(r.get("chunk_id") or ""),
                            },
                        )

            # 1b) any deficit-cue window
            for r in candidate_order:
                text = str(r.get("chunk_text") or "")
                for w in _windows(text):
                    if not _has_deficit_cue(w):
                        continue
                    ans = self._extract_answer_single_chunk(question=question, text=w)
                    if ans:
                        return (
                            ans,
                            str(r.get("chunk_id") or "") or None,
                            {
                                "strategy": "topk_deficit_window",
                                "ordering_notes": ordering_notes,
                                "entity_terms": entity_terms,
                                "selected_rank": int(r.get("rank") or 0),
                                "selected_chunk_id": str(r.get("chunk_id") or ""),
                            },
                        )

        for r in candidate_order:
            ans = self._extract_answer_single_chunk(question=question, text=str(r.get("chunk_text") or ""))
            if ans:
                return (
                    ans,
                    str(r.get("chunk_id") or "") or None,
                    {
                        "strategy": "topk_entity_aware",
                        "ordering_notes": ordering_notes,
                        "entity_terms": entity_terms,
                        "entity_filtered_first": bool(entity_first),
                        "selected_rank": int(r.get("rank") or 0),
                        "selected_chunk_id": str(r.get("chunk_id") or ""),
                    },
                )

        # Non-numeric fallback: return the sentence most similar to query terms from top chunk.
        top = results[0]
        chunk_id = str(top.get("chunk_id") or "")
        text = str(top.get("chunk_text") or "").strip()
        if not text:
            return None, chunk_id or None, {
                "strategy": "topk_entity_aware",
                "ordering_notes": ordering_notes,
                "entity_terms": entity_terms,
                "reason": "empty_top_chunk",
            }
        stop = {
            "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "in",
            "is", "it", "of", "on", "or", "that", "the", "to", "was", "what", "when",
            "where", "which", "who", "why", "with", "during", "used",
        }
        q_terms = {
            t for t in re.findall(r"[a-z0-9][a-z0-9\\-]{2,}", q)
            if t not in stop and not t.isdigit()
        }
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
        if q_terms and sentences:
            best_sent = ""
            best_score = -1
            for s in sentences:
                s_norm = s.lower()
                score = sum(1 for t in q_terms if t in s_norm)
                if score > best_score:
                    best_score = score
                    best_sent = s
            if best_sent:
                return best_sent[:240], chunk_id or None, {
                    "strategy": "fallback_sentence_overlap",
                    "ordering_notes": ordering_notes,
                    "entity_terms": entity_terms,
                    "selected_rank": int(top.get("rank") or 0),
                    "selected_chunk_id": chunk_id,
                }

        sentence = re.split(r"(?<=[.!?])\s+", text)[0].strip()
        if sentence:
            return sentence[:240], chunk_id or None, {
                "strategy": "fallback_first_sentence",
                "ordering_notes": ordering_notes,
                "entity_terms": entity_terms,
                "selected_rank": int(top.get("rank") or 0),
                "selected_chunk_id": chunk_id,
            }
        return None, chunk_id or None, {
            "strategy": "topk_entity_aware",
            "ordering_notes": ordering_notes,
            "entity_terms": entity_terms,
            "reason": "no_candidate_found",
        }
