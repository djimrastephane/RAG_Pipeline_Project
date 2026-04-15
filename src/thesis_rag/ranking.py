from __future__ import annotations

from .schemas import RetrievalHit


def chunk_hits_to_page_hits(
    hits: list[RetrievalHit],
    method_name: str,
    *,
    chunk_limit: int | None = None,
) -> list[RetrievalHit]:
    ranked_pages: list[RetrievalHit] = []
    query_order = _query_order(hits)
    for query_id in query_order:
        query_hits = [hit for hit in hits if hit.query_id == query_id]
        query_hits.sort(key=lambda hit: (hit.rank, -hit.score, hit.doc_id, hit.page_number, hit.chunk_id or ""))
        if chunk_limit is not None:
            query_hits = query_hits[:chunk_limit]
        seen_pages: set[tuple[str, int]] = set()
        page_rank = 0
        for hit in query_hits:
            pages = list(hit.pages or [hit.page_number])
            for page_number in pages:
                key = (hit.doc_id, int(page_number))
                if key in seen_pages:
                    continue
                seen_pages.add(key)
                page_rank += 1
                ranked_pages.append(
                    RetrievalHit(
                        query_id=hit.query_id,
                        query_text=hit.query_text,
                        rank=page_rank,
                        score=hit.score,
                        retrieval_method=method_name,
                        doc_id=hit.doc_id,
                        page_number=int(page_number),
                        chunk_id=hit.chunk_id,
                        pages=[int(page_number)],
                        text=hit.text,
                    )
                )
    return ranked_pages


def _query_order(hits: list[RetrievalHit]) -> list[str]:
    seen: set[str] = set()
    order: list[str] = []
    for hit in hits:
        if hit.query_id in seen:
            continue
        seen.add(hit.query_id)
        order.append(hit.query_id)
    return order
