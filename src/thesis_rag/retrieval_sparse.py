from __future__ import annotations

import math
import re
from collections import Counter

from .schemas import BM25Config, ChunkRecord, QueryRecord, RetrievalHit


def build_bm25(chunks: list[ChunkRecord], config: BM25Config):
    corpus = [_tokenize_bm25(chunk.text) for chunk in chunks]
    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(corpus, k1=config.k1, b=config.b)
    except ModuleNotFoundError:
        return _FallbackBM25(corpus, k1=config.k1, b=config.b)


def sparse_retrieve(
    bm25,
    chunk_records: list[ChunkRecord],
    queries: list[QueryRecord],
    *,
    top_k: int,
) -> list[RetrievalHit]:
    hits: list[RetrievalHit] = []
    for query in queries:
        scores = _get_bm25_scores(bm25, query.query_text)
        ranking = sorted(
            enumerate(scores),
            key=lambda item: (-float(item[1]), chunk_records[item[0]].chunk_id),
        )[:top_k]
        for rank, (index, score) in enumerate(ranking, start=1):
            chunk = chunk_records[index]
            hits.append(
                RetrievalHit(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    rank=rank,
                    score=float(score),
                    retrieval_method="bm25",
                    doc_id=chunk.doc_id,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    pages=_pages_for_chunk(chunk),
                    text=chunk.text,
                )
            )
    return hits


def sparse_retrieve_legacy_style(
    bm25,
    chunk_records: list[ChunkRecord],
    queries: list[QueryRecord],
    *,
    top_k: int,
) -> list[RetrievalHit]:
    hits: list[RetrievalHit] = []
    for query in queries:
        scores = _get_bm25_scores(bm25, query.query_text)
        ranking = [idx for idx, _score in sorted(enumerate(scores), key=lambda item: item[1], reverse=True)[:top_k]]
        for rank, index in enumerate(ranking, start=1):
            chunk = chunk_records[index]
            hits.append(
                RetrievalHit(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    rank=rank,
                    score=float(scores[index]),
                    retrieval_method="bm25",
                    doc_id=chunk.doc_id,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    pages=_pages_for_chunk(chunk),
                    text=chunk.text,
                )
            )
    return hits


def _tokenize_bm25(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9\\-]{1,}", str(text or "").lower())


def _get_bm25_scores(bm25, query_text: str):
    query_tokens = _tokenize_bm25(query_text)
    if hasattr(bm25, "get_scores"):
        return bm25.get_scores(query_tokens)
    if hasattr(bm25, "score_query"):
        return bm25.score_query(query_tokens)
    raise TypeError(f"Unsupported BM25 backend: {type(bm25)!r}")


def _pages_for_chunk(chunk: ChunkRecord) -> list[int]:
    raw_pages = chunk.pages
    if hasattr(raw_pages, "tolist"):
        raw_pages = raw_pages.tolist()
    if not raw_pages:
        return [int(chunk.page_number)]
    return [int(page) for page in raw_pages]


class _FallbackBM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float, b: float) -> None:
        self.docs_tokens = docs_tokens
        self.k1 = float(k1)
        self.b = float(b)
        self.n_docs = len(docs_tokens)
        self.doc_len = [len(tokens) for tokens in docs_tokens]
        self.avgdl = float(sum(self.doc_len)) / max(1.0, float(self.n_docs))
        self.term_freqs: list[Counter[str]] = [Counter(tokens) for tokens in docs_tokens]
        self.doc_freq: Counter[str] = Counter()
        for tf in self.term_freqs:
            for term in tf.keys():
                self.doc_freq[term] += 1
        self.idf: dict[str, float] = {
            term: math.log(1.0 + ((self.n_docs - df + 0.5) / (df + 0.5)))
            for term, df in self.doc_freq.items()
        }

    def score_query(self, query_tokens: list[str]) -> list[float]:
        if self.n_docs == 0:
            return []
        q_terms = Counter(query_tokens)
        scores = [0.0] * self.n_docs
        for term in q_terms.keys():
            if term not in self.idf:
                continue
            idf = self.idf[term]
            for index, tf in enumerate(self.term_freqs):
                freq = tf.get(term, 0)
                if freq <= 0:
                    continue
                doc_len = self.doc_len[index]
                denom = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avgdl, 1e-9)))
                scores[index] += idf * (freq * (self.k1 + 1.0) / max(denom, 1e-12))
        return scores
