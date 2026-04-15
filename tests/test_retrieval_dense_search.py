import numpy as np

from thesis_rag.retrieval_dense import search_faiss_stably


class FakeIndex:
    def search(self, query_vectors: np.ndarray, top_k: int):
        base = float(query_vectors[0, 0])
        scores = np.array([[base + rank for rank in range(top_k)]], dtype=np.float32)
        indices = np.array([[rank for rank in range(top_k)]], dtype=np.int64)
        return scores, indices


def test_search_faiss_stably_stacks_single_query_results() -> None:
    index = FakeIndex()
    query_vectors = np.array([[1.0, 0.0], [2.0, 0.0]], dtype=np.float32)

    scores, indices = search_faiss_stably(index, query_vectors, top_k=3)

    assert scores.shape == (2, 3)
    assert indices.shape == (2, 3)
    assert scores.tolist() == [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]
    assert indices.tolist() == [[0, 1, 2], [0, 1, 2]]
