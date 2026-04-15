from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import re
from typing import Any

import faiss
import numpy as np
import pandas as pd


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
    rrf_k: int = 20,
    dense_weight: float = 0.5,
    bm25_weight: float = 2.0,
) -> tuple[list[int], dict[int, float]]:
    """Fuse ranked lists with Reciprocal Rank Fusion and return fused rank + scores."""
    scores: dict[int, float] = {}
    for rank, idx in enumerate(dense_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + (dense_weight / float(rrf_k + rank))
    for rank, idx in enumerate(bm25_ranked, start=1):
        scores[idx] = scores.get(idx, 0.0) + (bm25_weight / float(rrf_k + rank))
    ranked = [idx for idx, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True)]
    return ranked, scores


def _minmax_normalize_scores(
    score_map: dict[int, float],
    candidates: list[int],
) -> dict[int, float]:
    vals = np.asarray([score_map.get(i, np.nan) for i in candidates], dtype=np.float32)
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


def score_fuse(
    dense_score_map: dict[int, float],
    bm25_score_map: dict[int, float],
    dense_weight: float = 0.5,
    bm25_weight: float = 2.0,
) -> tuple[list[int], dict[int, float]]:
    """Fuse min-max normalized dense/BM25 scores with weighted sum."""
    candidates = sorted(set(dense_score_map.keys()).union(set(bm25_score_map.keys())))
    if not candidates:
        return [], {}
    dn = _minmax_normalize_scores(dense_score_map, candidates)
    bn = _minmax_normalize_scores(bm25_score_map, candidates)
    scores = {
        idx: float(dense_weight) * float(dn[idx]) + float(bm25_weight) * float(bn[idx])
        for idx in candidates
    }
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

