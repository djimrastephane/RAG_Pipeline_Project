from __future__ import annotations

from collections import defaultdict

import pandas as pd

from .schemas import EvaluationResult, QueryDiagnostics, QueryRecord, RetrievalHit


def build_query_diagnostics(
    queries: list[QueryRecord],
    dense_hits: list[RetrievalHit],
    sparse_hits: list[RetrievalHit],
    hybrid_hits: list[RetrievalHit],
    evaluation_results: list[EvaluationResult],
) -> list[QueryDiagnostics]:
    dense_map = _group_hits(dense_hits)
    sparse_map = _group_hits(sparse_hits)
    hybrid_map = _group_hits(hybrid_hits)
    eval_map = {result.query_id: result for result in evaluation_results}
    diagnostics: list[QueryDiagnostics] = []
    for query in queries:
        dense = dense_map.get(query.query_id, [])
        sparse = sparse_map.get(query.query_id, [])
        hybrid = hybrid_map.get(query.query_id, [])
        result = eval_map[query.query_id]
        dense_top1 = dense[0].score if len(dense) >= 1 else None
        dense_top2 = dense[1].score if len(dense) >= 2 else None
        diagnostics.append(
            QueryDiagnostics(
                query_id=query.query_id,
                query_text=query.query_text,
                doc_id=query.doc_id,
                gold_pages=query.gold_pages,
                dense_top_k_pages=[hit.page_number for hit in dense],
                bm25_top_k_pages=[hit.page_number for hit in sparse],
                hybrid_top_k_pages=[hit.page_number for hit in hybrid],
                hit_at_1=result.hit_at_1,
                hit_at_3=result.hit_at_3,
                reciprocal_rank=result.reciprocal_rank,
                dense_top1_score=dense_top1,
                dense_top2_score=dense_top2,
                dense_margin=(dense_top1 - dense_top2) if dense_top1 is not None and dense_top2 is not None else None,
                hybrid_top1_item=hybrid[0].chunk_id if hybrid else None,
                evidence_layout=query.evidence_layout,
                difficulty=query.difficulty,
                failure_type=result.failure_type,
            )
        )
    return diagnostics


def save_diagnostics_csv(diagnostics: list[QueryDiagnostics], out_path) -> None:
    pd.DataFrame([row.to_dict() for row in diagnostics]).to_csv(out_path, index=False)


def _group_hits(hits: list[RetrievalHit]) -> dict[str, list[RetrievalHit]]:
    grouped: dict[str, list[RetrievalHit]] = defaultdict(list)
    for hit in hits:
        grouped[hit.query_id].append(hit)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda item: item.rank)
    return grouped
