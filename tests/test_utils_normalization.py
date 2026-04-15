import numpy as np

from thesis_rag.utils import l2_normalize


def test_l2_normalize_returns_unit_vectors() -> None:
    matrix = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    normalized = l2_normalize(matrix)
    norms = np.linalg.norm(normalized, axis=1)
    assert np.allclose(norms, np.array([1.0, 1.0]), atol=1e-6)
