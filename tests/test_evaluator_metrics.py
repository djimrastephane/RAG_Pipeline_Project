from thesis_rag.evaluator import hit_at_k, reciprocal_rank


def test_hit_at_k() -> None:
    assert hit_at_k([4, 7, 9], [7], 3) is True
    assert hit_at_k([4, 7, 9], [7], 1) is False


def test_mrr_reciprocal_rank() -> None:
    rr, rank = reciprocal_rank([4, 7, 9], [7])
    assert rr == 0.5
    assert rank == 2
