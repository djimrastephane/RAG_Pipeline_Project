from thesis_rag.pipeline import _validate_queries
from thesis_rag.schemas import ChunkRecord, QueryRecord


def test_query_validation_rejects_missing_gold_pages() -> None:
    chunks = [ChunkRecord("c1", "doc", 1, 0, "text", 10, 2)]
    queries = [QueryRecord("q1", "query", "doc", [2])]
    try:
        _validate_queries(queries, chunks)
    except ValueError as exc:
        assert "missing gold pages" in str(exc)
    else:
        raise AssertionError("Expected validation error for missing gold pages")
