"""Retrieval quality regression gate.

Reads committed retrieval_metrics_hybrid.json files and asserts that each doc
meets minimum page-hit-rate thresholds. Exits non-zero on failure so CI can use
it as a quality gate without needing FAISS indexes at test time.

Thresholds are calibrated against the baseline run of 2026-06-06:
  - 22/26 silver docs achieve page_hit_rate@1 = 1.000
  - Worst doc (Grampian-2018-2019): 0.545 @k=1, 0.909 @k=5

Usage:
    python scripts/eval_regression_gate.py [--data-root data_processed]
    python scripts/eval_regression_gate.py --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

THRESHOLDS = {
    "min_hit_rate_at_1": 0.40,   # individual doc floor — catches total retrieval failures
    "min_hit_rate_at_5": 0.80,   # individual doc floor at k=5
    "mean_hit_rate_at_1": 0.90,  # aggregate across all docs
    "mean_hit_rate_at_5": 0.92,  # aggregate across all docs
}


def load_metrics(metrics_file: Path) -> dict:
    with metrics_file.open() as f:
        return json.load(f)


def check_doc(doc_id: str, metrics: dict, verbose: bool) -> list[str]:
    """Return list of failure messages (empty = pass)."""
    failures = []
    by_k = metrics.get("metrics_by_k", {})

    for k_str, label in [("1", "k=1"), ("5", "k=5")]:
        threshold_key = f"min_hit_rate_at_{k_str}"
        row = by_k.get(k_str, {})
        hit_rate = row.get("page_hit_rate_at_k")
        if hit_rate is None:
            if verbose:
                print(f"  {doc_id}: SKIP ({label} missing)")
            continue
        if hit_rate < THRESHOLDS[threshold_key]:
            msg = (
                f"{doc_id}: page_hit_rate@{k_str}={hit_rate:.3f} "
                f"< threshold {THRESHOLDS[threshold_key]:.3f}"
            )
            failures.append(msg)
        elif verbose:
            print(f"  {doc_id}: page_hit_rate@{k_str}={hit_rate:.3f}  OK")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data_processed")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    data_root = REPO_ROOT / args.data_root
    metrics_files = sorted(data_root.glob("*/retrieval_metrics_hybrid.json"))

    if not metrics_files:
        print(f"ERROR: no retrieval_metrics_hybrid.json files found under {data_root}")
        print("Run scripts/retrieval_eval_hybrid.py for each doc first.")
        return 1

    all_failures: list[str] = []
    hit_rates_at_1: list[float] = []
    hit_rates_at_5: list[float] = []

    for mf in metrics_files:
        doc_id = mf.parent.name
        metrics = load_metrics(mf)
        failures = check_doc(doc_id, metrics, args.verbose)
        all_failures.extend(failures)

        by_k = metrics.get("metrics_by_k", {})
        r1 = (by_k.get("1") or {}).get("page_hit_rate_at_k")
        r5 = (by_k.get("5") or {}).get("page_hit_rate_at_k")
        if r1 is not None:
            hit_rates_at_1.append(r1)
        if r5 is not None:
            hit_rates_at_5.append(r5)

    # Aggregate checks
    if hit_rates_at_1:
        mean_1 = sum(hit_rates_at_1) / len(hit_rates_at_1)
        if args.verbose:
            print(f"\nAggregate page_hit_rate@1: {mean_1:.3f} ({len(hit_rates_at_1)} docs)")
        if mean_1 < THRESHOLDS["mean_hit_rate_at_1"]:
            all_failures.append(
                f"Aggregate page_hit_rate@1={mean_1:.3f} "
                f"< threshold {THRESHOLDS['mean_hit_rate_at_1']}"
            )

    if hit_rates_at_5:
        mean_5 = sum(hit_rates_at_5) / len(hit_rates_at_5)
        if args.verbose:
            print(f"Aggregate page_hit_rate@5: {mean_5:.3f} ({len(hit_rates_at_5)} docs)")
        if mean_5 < THRESHOLDS["mean_hit_rate_at_5"]:
            all_failures.append(
                f"Aggregate page_hit_rate@5={mean_5:.3f} "
                f"< threshold {THRESHOLDS['mean_hit_rate_at_5']}"
            )

    if all_failures:
        print("\nREGRESSION GATE FAILED:")
        for f in all_failures:
            print(f"  FAIL: {f}")
        return 1

    n = len(metrics_files)
    print(f"Regression gate passed: {n} docs checked, all above thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
