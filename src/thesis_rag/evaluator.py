from __future__ import annotations

from collections import defaultdict

from .schemas import EvaluationResult, QueryRecord, RetrievalHit


def hit_at_k(predicted_pages: list[int], gold_pages: list[int], k: int) -> bool:
    return any(page in set(gold_pages) for page in predicted_pages[:k])


def reciprocal_rank(predicted_pages: list[int], gold_pages: list[int]) -> tuple[float, int | None]:
    gold = set(gold_pages)
    for index, page in enumerate(predicted_pages, start=1):
        if page in gold:
            return 1.0 / index, index
    return 0.0, None


def infer_failure_type(gold_pages: list[int], predicted_pages: list[int]) -> str | None:
    if not gold_pages:
        return "missing_gold_pages"
    if not predicted_pages:
        return "no_predictions"
    if predicted_pages[0] not in set(gold_pages):
        return "missed_top_rank"
    return None


def evaluate_page_hits(queries: list[QueryRecord], page_hits: list[RetrievalHit]) -> list[EvaluationResult]:
    grouped: dict[str, list[RetrievalHit]] = defaultdict(list)
    for hit in page_hits:
        grouped[hit.query_id].append(hit)
    results: list[EvaluationResult] = []
    for query in queries:
        ordered = sorted(grouped.get(query.query_id, []), key=lambda hit: hit.rank)
        predicted_pages = [hit.page_number for hit in ordered]
        rr, rank = reciprocal_rank(predicted_pages, query.gold_pages)
        results.append(
            EvaluationResult(
                query_id=query.query_id,
                doc_id=query.doc_id,
                gold_pages=query.gold_pages,
                predicted_pages=predicted_pages,
                hit_at_1=hit_at_k(predicted_pages, query.gold_pages, 1),
                hit_at_3=hit_at_k(predicted_pages, query.gold_pages, 3),
                reciprocal_rank=rr,
                first_relevant_rank=rank,
                failure_type=infer_failure_type(query.gold_pages, predicted_pages),
            )
        )
    return results


def aggregate_metrics(results: list[EvaluationResult], ks: list[int]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in ks:
        metrics[f"hit@{k}"] = sum(1 for result in results if hit_at_k(result.predicted_pages, result.gold_pages, k)) / max(
            len(results), 1
        )
    metrics["mrr"] = sum(result.reciprocal_rank for result in results) / max(len(results), 1)
    return metrics
