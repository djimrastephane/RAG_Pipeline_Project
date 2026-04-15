from thesis_rag.ranking import chunk_hits_to_page_hits
from thesis_rag.schemas import RetrievalHit


def test_chunk_to_page_mapping_deduplicates_stably() -> None:
    hits = [
        RetrievalHit("q1", "query", 1, 0.9, "dense", "doc", 4, "c2"),
        RetrievalHit("q1", "query", 2, 0.95, "dense", "doc", 4, "c1"),
        RetrievalHit("q1", "query", 3, 0.7, "dense", "doc", 7, "c3"),
    ]
    page_hits = chunk_hits_to_page_hits(hits, "dense_pages")
    assert [hit.page_number for hit in page_hits] == [4, 7]
    assert page_hits[0].chunk_id == "c2"


def test_chunk_to_page_mapping_respects_chunk_limit_before_dedup() -> None:
    hits = [
        RetrievalHit("q1", "query", 1, 0.9, "dense", "doc", 4, "c1"),
        RetrievalHit("q1", "query", 2, 0.8, "dense", "doc", 4, "c2"),
        RetrievalHit("q1", "query", 3, 0.7, "dense", "doc", 7, "c3"),
    ]
    page_hits = chunk_hits_to_page_hits(hits, "dense_pages", chunk_limit=2)
    assert [hit.page_number for hit in page_hits] == [4]


def test_chunk_to_page_mapping_expands_cross_page_hits() -> None:
    hits = [
        RetrievalHit("q1", "query", 1, 0.9, "dense", "doc", 20, "x1", pages=[20, 21]),
        RetrievalHit("q1", "query", 2, 0.8, "dense", "doc", 21, "c2", pages=[21]),
        RetrievalHit("q1", "query", 3, 0.7, "dense", "doc", 22, "c3", pages=[22]),
    ]
    page_hits = chunk_hits_to_page_hits(hits, "dense_pages", chunk_limit=2)
    assert [hit.page_number for hit in page_hits] == [20, 21]
    assert all(hit.chunk_id == "x1" for hit in page_hits)


def test_cross_page_expansion_preserves_first_seen_pages_before_later_duplicates() -> None:
    hits = [
        RetrievalHit("q1", "query", 1, 0.91, "hybrid", "doc", 79, "x79", pages=[79, 80]),
        RetrievalHit("q1", "query", 2, 0.89, "hybrid", "doc", 80, "c80", pages=[80]),
        RetrievalHit("q1", "query", 3, 0.88, "hybrid", "doc", 22, "x22", pages=[22, 23]),
        RetrievalHit("q1", "query", 4, 0.87, "hybrid", "doc", 23, "c23", pages=[23]),
    ]

    page_hits = chunk_hits_to_page_hits(hits, "hybrid_pages", chunk_limit=4)

    assert [hit.page_number for hit in page_hits] == [79, 80, 22, 23]
    assert [hit.chunk_id for hit in page_hits] == ["x79", "x79", "x22", "x22"]
