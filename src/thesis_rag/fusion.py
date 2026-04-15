from __future__ import annotations

from collections import defaultdict

from .schemas import RetrievalHit


def reciprocal_rank_fusion(
    named_hit_lists: dict[str, list[RetrievalHit]],
    *,
    rrf_k: int,
    weights: dict[str, float] | None = None,
) -> list[RetrievalHit]:
    weights = weights or {name: 1.0 for name in named_hit_lists}
    scores: dict[tuple[str, str], float] = defaultdict(float)
    exemplar: dict[tuple[str, str], RetrievalHit] = {}
    for method_name, hits in named_hit_lists.items():
        weight = weights.get(method_name, 1.0)
        for hit in hits:
            key = (hit.query_id, hit.chunk_id or f"{hit.doc_id}:{hit.page_number}")
            scores[key] += weight / (rrf_k + hit.rank)
            exemplar.setdefault(key, hit)
    per_query: dict[str, list[tuple[RetrievalHit, float]]] = defaultdict(list)
    for key, fused_score in scores.items():
        hit = exemplar[key]
        per_query[hit.query_id].append((hit, fused_score))
    fused_hits: list[RetrievalHit] = []
    for query_id, rows in per_query.items():
        ordered = sorted(
            rows,
            key=lambda item: (-item[1], item[0].chunk_id or "", item[0].page_number),
        )
        for rank, (hit, fused_score) in enumerate(ordered, start=1):
            fused_hits.append(
                RetrievalHit(
                    query_id=hit.query_id,
                    query_text=hit.query_text,
                    rank=rank,
                    score=float(fused_score),
                    retrieval_method="rrf",
                    doc_id=hit.doc_id,
                    page_number=hit.page_number,
                    chunk_id=hit.chunk_id,
                    text=hit.text,
                )
            )
    return fused_hits
