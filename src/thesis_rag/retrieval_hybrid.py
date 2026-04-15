from __future__ import annotations

import pandas as pd

from rag_pdf.retrieval.canonical_hybrid import apply_post_fusion_rerank, fuse_ranked_lists
from rag_pdf.retrieval.rerank import RerankConfig

from .retrieval_sparse import _get_bm25_scores
from .schemas import ChunkRecord, QueryRecord, RetrievalHit


def hybrid_retrieve_legacy_style(
    *,
    chunks: list[ChunkRecord],
    queries: list[QueryRecord],
    dense_scores,
    dense_indices,
    bm25,
    max_k_search: int,
    dense_weight: float,
    bm25_weight: float,
    rrf_k: int,
    enable_lexical_rerank: bool = True,
    enable_subsection_boost: bool = True,
    subsection_boost: float = 0.05,
    cross_page_out_of_section_penalty: float = 0.08,
) -> tuple[list[RetrievalHit], list[RetrievalHit], list[RetrievalHit]]:
    meta = pd.DataFrame([chunk.to_dict() for chunk in chunks])
    text_by_id = {chunk.chunk_id: chunk.text for chunk in chunks}
    rerank_cfg = RerankConfig(
        table_chunk_boost=0.08,
        entity_match_boost=0.04,
        numeric_density_boost=0.03,
        segment_search_hit_boost=0.03,
        max_entity_matches=4,
    )

    dense_hits: list[RetrievalHit] = []
    bm25_hits: list[RetrievalHit] = []
    hybrid_hits: list[RetrievalHit] = []

    for query_index, query in enumerate(queries):
        dense_ranked = dense_indices[query_index].tolist()
        dense_score_row = dense_scores[query_index].tolist()
        bm25_scores = _get_bm25_scores(bm25, query.query_text)
        bm25_ranked = [idx for idx, _score in sorted(enumerate(bm25_scores), key=lambda item: item[1], reverse=True)[:max_k_search]]

        fused_ranked, scores_map = fuse_ranked_lists(
            fusion_strategy="rrf",
            dense_ranked=dense_ranked,
            bm25_ranked=bm25_ranked,
            dense_score_map={int(idx): float(score) for idx, score in zip(dense_ranked, dense_score_row)},
            bm25_score_map={int(idx): float(score) for idx, score in enumerate(bm25_scores)},
            rrf_k=int(rrf_k),
            dense_weight=float(dense_weight),
            bm25_weight=float(bm25_weight),
        )
        fused_ranked, scores_map = apply_post_fusion_rerank(
            question=query.query_text,
            fused_ranked=fused_ranked,
            scores_map=scores_map,
            meta=meta,
            chunk_text_by_id=text_by_id,
            rerank_cfg=rerank_cfg,
            enable_lexical_rerank=enable_lexical_rerank,
            expected_section=str(query.expected_section or ""),
            expected_subsection=str(query.expected_subsection or ""),
            enable_subsection_boost=enable_subsection_boost,
            subsection_boost=subsection_boost,
            cross_page_out_of_section_penalty=cross_page_out_of_section_penalty,
        )

        for rank, idx in enumerate(dense_ranked[:max_k_search], start=1):
            chunk = chunks[idx]
            dense_hits.append(
                RetrievalHit(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    rank=rank,
                    score=float(dense_score_row[rank - 1]),
                    retrieval_method="dense",
                    doc_id=chunk.doc_id,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    pages=_pages_for_chunk(chunk),
                    text=chunk.text,
                )
            )
        for rank, idx in enumerate(bm25_ranked[:max_k_search], start=1):
            chunk = chunks[idx]
            bm25_hits.append(
                RetrievalHit(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    rank=rank,
                    score=float(bm25_scores[idx]),
                    retrieval_method="bm25",
                    doc_id=chunk.doc_id,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    pages=_pages_for_chunk(chunk),
                    text=chunk.text,
                )
            )
        for rank, idx in enumerate(fused_ranked[:max_k_search], start=1):
            chunk = chunks[idx]
            hybrid_hits.append(
                RetrievalHit(
                    query_id=query.query_id,
                    query_text=query.query_text,
                    rank=rank,
                    score=float(scores_map.get(idx, 0.0)),
                    retrieval_method="hybrid",
                    doc_id=chunk.doc_id,
                    page_number=chunk.page_number,
                    chunk_id=chunk.chunk_id,
                    pages=_pages_for_chunk(chunk),
                    text=chunk.text,
                )
            )

    return dense_hits, bm25_hits, hybrid_hits


def _pages_for_chunk(chunk: ChunkRecord) -> list[int]:
    raw_pages = chunk.pages
    if hasattr(raw_pages, "tolist"):
        raw_pages = raw_pages.tolist()
    if not raw_pages:
        return [int(chunk.page_number)]
    return [int(page) for page in raw_pages]
