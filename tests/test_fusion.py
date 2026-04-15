from thesis_rag.fusion import reciprocal_rank_fusion
from thesis_rag.schemas import RetrievalHit


def test_rrf_prefers_consensus_items() -> None:
    dense = [
        RetrievalHit("q1", "query", 1, 0.9, "dense", "doc", 3, "c1"),
        RetrievalHit("q1", "query", 2, 0.8, "dense", "doc", 4, "c2"),
        RetrievalHit("q1", "query", 3, 0.7, "dense", "doc", 5, "c3"),
    ]
    sparse = [
        RetrievalHit("q1", "query", 1, 2.0, "bm25", "doc", 4, "c2"),
        RetrievalHit("q1", "query", 2, 1.7, "bm25", "doc", 5, "c3"),
        RetrievalHit("q1", "query", 3, 1.6, "bm25", "doc", 3, "c1"),
    ]
    fused = reciprocal_rank_fusion({"dense": dense, "bm25": sparse}, rrf_k=20)
    assert fused[0].chunk_id == "c2"
