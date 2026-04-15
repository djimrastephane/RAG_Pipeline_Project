from __future__ import annotations

import faiss
import numpy as np
import pandas as pd

from rag_pdf.question_router import route_question
from rag_pdf.retrieval.rerank import (
    RerankConfig,
    numeric_density_boost,
    query_overlap_boost,
    segment_search_hit_boost,
    table_priority_boost,
)

from .schemas import ChunkRecord, QueryRecord, RetrievalHit


def search_faiss_stably(index: faiss.Index, query_vectors: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Search one query at a time to avoid macOS batch-search instability in some FAISS builds."""
    if query_vectors.ndim != 2:
        raise ValueError(f"Expected 2D query matrix, got shape {query_vectors.shape}")
    scores_rows: list[np.ndarray] = []
    indices_rows: list[np.ndarray] = []
    for row in query_vectors:
        row_scores, row_indices = index.search(np.ascontiguousarray(row.reshape(1, -1).astype(np.float32)), top_k)
        scores_rows.append(row_scores[0])
        indices_rows.append(row_indices[0])
    return np.vstack(scores_rows), np.vstack(indices_rows)


def dense_retrieve(
    index: faiss.Index,
    chunk_records: list[ChunkRecord],
    query_records: list[QueryRecord],
    query_vectors: np.ndarray,
    *,
    top_k: int,
) -> list[RetrievalHit]:
    scores, indices = search_faiss_stably(index, query_vectors, top_k)
    hits: list[RetrievalHit] = []
    for query_row, query, row_scores, row_indices in zip(range(len(query_records)), query_records, scores, indices):
        for rank, (score, idx) in enumerate(zip(row_scores.tolist(), row_indices.tolist()), start=1):
            if idx < 0:
                continue
            chunk = chunk_records[idx]
            hits.append(
                RetrievalHit(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    rank=rank,
                    score=float(score),
                    retrieval_method="dense",
                    doc_id=chunk.doc_id,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    pages=_pages_for_chunk(chunk),
                    text=chunk.text,
                )
            )
    return hits


def dense_retrieve_legacy_style(
    index: faiss.Index,
    chunk_records: list[ChunkRecord],
    query_records: list[QueryRecord],
    query_vectors: np.ndarray,
    *,
    top_k: int,
    max_k_search: int = 100,
) -> list[RetrievalHit]:
    meta = pd.DataFrame([chunk.to_dict() for chunk in chunk_records])
    chunk_text_by_id = {chunk.chunk_id: chunk.text for chunk in chunk_records}
    rerank_cfg = RerankConfig(
        table_chunk_boost=0.08,
        entity_match_boost=0.04,
        numeric_density_boost=0.03,
        segment_search_hit_boost=0.03,
        max_entity_matches=4,
    )
    raw_scores, raw_indices = search_faiss_stably(index, query_vectors, min(max_k_search, len(chunk_records)))
    hits: list[RetrievalHit] = []
    for query, scores, indices in zip(query_records, raw_scores, raw_indices):
        ranked_pairs = [(float(score), int(idx)) for score, idx in zip(scores.tolist(), indices.tolist()) if idx >= 0]
        route = route_question(query.query_text)
        boosted: list[tuple[float, int]] = []
        for score, idx in ranked_pairs:
            row = meta.iloc[idx]
            ctext = chunk_text_by_id.get(str(row.get("chunk_id") or ""), "")
            score += table_priority_boost(bool(row.get("is_table", False)), route.intent, rerank_cfg)
            score += query_overlap_boost(query.query_text, ctext, rerank_cfg)
            score += numeric_density_boost(query.query_text, ctext, rerank_cfg)
            score += segment_search_hit_boost(query.query_text, bool(row.get("segment_has_search_hit", False)), rerank_cfg)
            boosted.append((score, idx))
        boosted.sort(key=lambda item: item[0], reverse=True)
        if query.expected_subsection and "subsection_title" in meta.columns:
            target = str(query.expected_subsection or "").strip().lower()
            boosted = [
                (
                    score + 0.05 if str(meta.iloc[idx].get("subsection_title", "") or "").strip().lower() == target else score,
                    idx,
                )
                for score, idx in boosted
            ]
            boosted.sort(key=lambda item: item[0], reverse=True)
        for rank, (score, idx) in enumerate(boosted[:top_k], start=1):
            chunk = chunk_records[idx]
            hits.append(
                RetrievalHit(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    rank=rank,
                    score=float(score),
                    retrieval_method="dense",
                    doc_id=chunk.doc_id,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    pages=_pages_for_chunk(chunk),
                    text=chunk.text,
                )
            )
    return hits


def _pages_for_chunk(chunk: ChunkRecord) -> list[int]:
    raw_pages = chunk.pages
    if hasattr(raw_pages, "tolist"):
        raw_pages = raw_pages.tolist()
    if not raw_pages:
        return [int(chunk.page_number)]
    return [int(page) for page in raw_pages]
