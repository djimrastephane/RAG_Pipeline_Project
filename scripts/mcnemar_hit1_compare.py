from __future__ import annotations

"""
Paired McNemar significance test for retrieval Hit@1 (Hybrid vs Dense).

This script:
1) loads two retrieval result JSON files,
2) aligns rows by query_id,
3) computes paired Hit@1 correctness per query,
4) runs McNemar's test (exact for small discordant counts),
5) saves tidy JSON + CSV outputs.
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueryHit1:
    query_id: str
    hit1: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired McNemar test for Hit@1: Hybrid vs Dense retrieval.")
    p.add_argument("--hybrid", required=True, help="Path to hybrid retrieval results JSON.")
    p.add_argument("--dense", required=True, help="Path to dense retrieval results JSON.")
    p.add_argument("--cohort", required=True, help="Cohort label (e.g., Grampian-2023-2024).")
    p.add_argument("--out-dir", default="results/mcnemar_hit1", help="Output directory.")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    p.add_argument(
        "--allow-partial-overlap",
        action="store_true",
        help="Allow analysis on intersection of query IDs. Default is strict (raise on mismatch).",
    )
    return p.parse_args()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON file: {path}") from exc


def _extract_rows(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
        rows = payload["results"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(
            f"Unsupported JSON schema in {path}. Expected either a list of query records "
            f"or a dict containing a 'results' list."
        )
    if not rows:
        raise ValueError(f"No query rows found in {path}.")
    if not all(isinstance(r, dict) for r in rows):
        raise ValueError(f"Malformed query rows in {path}: every row must be an object.")
    return rows


def _normalize_page(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return str(value)
    return str(value).strip()


def _extract_gold_page(row: dict[str, Any]) -> str:
    """
    Supports:
    - gold_page
    - expected_page
    - expected_pages (list; uses first gold page)
    """
    if "gold_page" in row:
        out = _normalize_page(row.get("gold_page"))
        if out:
            return out
    if "expected_page" in row:
        out = _normalize_page(row.get("expected_page"))
        if out:
            return out
    expected_pages = row.get("expected_pages")
    if isinstance(expected_pages, list) and expected_pages:
        out = _normalize_page(expected_pages[0])
        if out:
            return out
    raise ValueError(
        f"Missing gold page for query_id={row.get('query_id')}. "
        "Expected one of: gold_page, expected_page, expected_pages."
    )


def _extract_top1_page(row: dict[str, Any]) -> str:
    """
    Supports:
    - retrieved_results: [ {page|page_id|page_number|retrieved_page}, ... ]
    - per_k['1']['retrieved_pages_ranked'][0]
    """
    rr = row.get("retrieved_results")
    if isinstance(rr, list):
        if not rr:
            raise ValueError(f"Missing top-1 retrieval: empty retrieved_results for query_id={row.get('query_id')}.")
        first = rr[0]
        if not isinstance(first, dict):
            raise ValueError(f"Malformed retrieved_results[0] for query_id={row.get('query_id')}.")
        for key in ("page", "page_id", "page_number", "retrieved_page"):
            if key in first:
                out = _normalize_page(first.get(key))
                if out:
                    return out
        raise ValueError(
            f"Missing page field in retrieved_results[0] for query_id={row.get('query_id')}. "
            "Expected page/page_id/page_number/retrieved_page."
        )

    per_k = row.get("per_k")
    if isinstance(per_k, dict):
        # Prefer explicit k=1. If unavailable, use the smallest available numeric k.
        candidate_keys: list[str] = []
        if "1" in per_k:
            candidate_keys.append("1")
        numeric_keys = sorted(
            [k for k in per_k.keys() if str(k).isdigit()],
            key=lambda z: int(str(z)),
        )
        for k in numeric_keys:
            if k not in candidate_keys:
                candidate_keys.append(k)

        for k in candidate_keys:
            block = per_k.get(k)
            if not isinstance(block, dict):
                continue
            ranked = block.get("retrieved_pages_ranked")
            if isinstance(ranked, list) and ranked:
                out = _normalize_page(ranked[0])
                if out:
                    return out
        raise ValueError(
            f"Missing top-1 retrieval in per_k[*]['retrieved_pages_ranked'] for query_id={row.get('query_id')}."
        )

    raise ValueError(
        f"Missing top-1 retrieval fields for query_id={row.get('query_id')}. "
        "Expected retrieved_results or per_k schema."
    )


def _extract_hit1_from_k1_recall(row: dict[str, Any]) -> int | None:
    """
    Fallback for pipeline outputs where top-1 page list is unavailable but per_k['1']
    contains page_recall_at_k.
    """
    per_k = row.get("per_k")
    if not isinstance(per_k, dict):
        return None
    k1 = per_k.get("1")
    if not isinstance(k1, dict):
        return None
    recall = k1.get("page_recall_at_k")
    if recall is None:
        return None
    try:
        return 1 if float(recall) > 0.0 else 0
    except (TypeError, ValueError):
        return None


def _index_hit1_by_query(path: Path) -> dict[str, QueryHit1]:
    payload = _load_json(path)
    rows = _extract_rows(payload, path)
    out: dict[str, QueryHit1] = {}

    for row in rows:
        query_id = str(row.get("query_id", "")).strip()
        if not query_id:
            raise ValueError(f"Missing query_id in {path}.")
        if query_id in out:
            raise ValueError(f"Duplicate query_id '{query_id}' in {path}.")
        try:
            gold_page = _extract_gold_page(row)
            top1_page = _extract_top1_page(row)
            hit1 = 1 if top1_page == gold_page else 0
        except ValueError:
            # If top-1 page is missing in pipeline output, use page_recall_at_k at k=1.
            # This retains paired logic while handling sparse/malformed top-1 lists.
            fallback_hit1 = _extract_hit1_from_k1_recall(row)
            if fallback_hit1 is not None:
                hit1 = fallback_hit1
            else:
                # Conservative fallback: if top-1 cannot be recovered, treat as miss.
                hit1 = 0
        out[query_id] = QueryHit1(query_id=query_id, hit1=hit1)
    return out


def _mcnemar_exact_fallback(b: int, c: int) -> tuple[float, float]:
    """
    Exact two-sided McNemar via Binomial(n=b+c, p=0.5).
    Returns: (statistic, p_value), where statistic=min(b,c).
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / float(2**n)
    p_value = min(1.0, 2.0 * tail)
    return float(k), float(p_value)


def _mcnemar_asymptotic_fallback(b: int, c: int) -> tuple[float, float]:
    """
    Continuity-corrected McNemar chi-square with df=1.
    For df=1, survival function is erfc(sqrt(x/2)).
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    statistic = ((abs(b - c) - 1.0) ** 2) / float(n)
    p_value = math.erfc(math.sqrt(statistic / 2.0))
    return float(statistic), float(p_value)


def run_mcnemar(table: list[list[int]], use_exact: bool) -> tuple[float, float, str]:
    """
    Run McNemar test.
    Preference: statsmodels; fallback implemented if unavailable.
    """
    try:
        from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

        result = sm_mcnemar(table, exact=use_exact, correction=(not use_exact))
        statistic = float(result.statistic) if result.statistic is not None else float("nan")
        p_value = float(result.pvalue)
        method = f"statsmodels_mcnemar_{'exact' if use_exact else 'asymptotic_cc'}"
        return statistic, p_value, method
    except Exception:
        b = int(table[0][1])
        c = int(table[1][0])
        if use_exact:
            statistic, p_value = _mcnemar_exact_fallback(b, c)
            method = "fallback_exact_binomial"
        else:
            statistic, p_value = _mcnemar_asymptotic_fallback(b, c)
            method = "fallback_asymptotic_cc"
        return statistic, p_value, method


def thesis_interpretation(p_value: float, alpha: float) -> str:
    if p_value < alpha:
        return (
            f"At alpha={alpha:.2f}, the paired McNemar test indicates a statistically significant "
            "difference in Hit@1 between Hybrid and Dense retrieval for this cohort."
        )
    return (
        f"At alpha={alpha:.2f}, the paired McNemar test does not indicate a statistically significant "
        "difference in Hit@1 between Hybrid and Dense retrieval for this cohort."
    )


def main() -> None:
    args = parse_args()
    hybrid_path = Path(args.hybrid).resolve()
    dense_path = Path(args.dense).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hybrid = _index_hit1_by_query(hybrid_path)
    dense = _index_hit1_by_query(dense_path)

    hybrid_ids = set(hybrid.keys())
    dense_ids = set(dense.keys())
    common_ids = sorted(hybrid_ids.intersection(dense_ids))

    if not common_ids:
        raise ValueError("No overlapping query_id values between Hybrid and Dense files.")

    missing_in_dense = sorted(hybrid_ids - dense_ids)
    missing_in_hybrid = sorted(dense_ids - hybrid_ids)
    if (missing_in_dense or missing_in_hybrid) and not args.allow_partial_overlap:
        raise ValueError(
            "Query ID mismatch between files. "
            f"Missing in dense: {len(missing_in_dense)}; missing in hybrid: {len(missing_in_hybrid)}. "
            "Use --allow-partial-overlap to analyze only the intersection."
        )

    both_correct = both_wrong = hybrid_correct_dense_wrong = hybrid_wrong_dense_correct = 0
    for qid in common_ids:
        h = hybrid[qid].hit1
        d = dense[qid].hit1
        if h == 1 and d == 1:
            both_correct += 1
        elif h == 1 and d == 0:
            hybrid_correct_dense_wrong += 1
        elif h == 0 and d == 1:
            hybrid_wrong_dense_correct += 1
        else:
            both_wrong += 1

    table = [
        [both_correct, hybrid_correct_dense_wrong],
        [hybrid_wrong_dense_correct, both_wrong],
    ]
    discordant = hybrid_correct_dense_wrong + hybrid_wrong_dense_correct
    use_exact = discordant <= 25
    statistic, p_value, method = run_mcnemar(table=table, use_exact=use_exact)
    significant = bool(p_value < float(args.alpha))

    result = {
        "cohort": args.cohort,
        "inputs": {
            "hybrid_file": str(hybrid_path),
            "dense_file": str(dense_path),
            "alpha": float(args.alpha),
            "allow_partial_overlap": bool(args.allow_partial_overlap),
        },
        "counts": {
            "n_hybrid_queries": int(len(hybrid_ids)),
            "n_dense_queries": int(len(dense_ids)),
            "n_paired_queries": int(len(common_ids)),
            "n_missing_in_dense": int(len(missing_in_dense)),
            "n_missing_in_hybrid": int(len(missing_in_hybrid)),
            "n_discordant": int(discordant),
        },
        "contingency_table": {
            "layout": "[[both_correct, hybrid_correct_dense_wrong], [hybrid_wrong_dense_correct, both_wrong]]",
            "both_correct": int(both_correct),
            "both_wrong": int(both_wrong),
            "hybrid_correct_dense_wrong": int(hybrid_correct_dense_wrong),
            "hybrid_wrong_dense_correct": int(hybrid_wrong_dense_correct),
            "table_2x2": table,
        },
        "mcnemar": {
            "method": method,
            "exact_used": bool(use_exact),
            "statistic": statistic,
            "p_value": p_value,
            "alpha": float(args.alpha),
            "significant": significant,
        },
        "interpretation": thesis_interpretation(p_value=p_value, alpha=float(args.alpha)),
    }

    json_path = out_dir / f"{args.cohort}_mcnemar_hit1.json"
    csv_path = out_dir / f"{args.cohort}_mcnemar_hit1_summary.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "cohort",
                "n_paired_queries",
                "both_correct",
                "both_wrong",
                "hybrid_correct_dense_wrong",
                "hybrid_wrong_dense_correct",
                "n_discordant",
                "exact_used",
                "statistic",
                "p_value",
                "alpha",
                "significant",
                "method",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "cohort": args.cohort,
                "n_paired_queries": len(common_ids),
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "hybrid_correct_dense_wrong": hybrid_correct_dense_wrong,
                "hybrid_wrong_dense_correct": hybrid_wrong_dense_correct,
                "n_discordant": discordant,
                "exact_used": use_exact,
                "statistic": statistic,
                "p_value": p_value,
                "alpha": float(args.alpha),
                "significant": significant,
                "method": method,
            }
        )

    print("Saved:", json_path)
    print("Saved:", csv_path)
    print("Contingency table:", table)
    print("McNemar p-value:", p_value)


if __name__ == "__main__":
    main()
