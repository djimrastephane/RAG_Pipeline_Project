from __future__ import annotations

"""
Batch runner for paired McNemar Hit@1 tests across cohorts.

It wraps `scripts/mcnemar_hit1_compare.py` and aggregates outputs into:
- mcnemar_hit1_batch_summary.csv
- mcnemar_hit1_batch_summary.json
"""

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from corpus_guard import is_eval_ready_doc_dir


REPO_ROOT = Path(__file__).resolve().parents[1]
MCNEMAR_SCRIPT = REPO_ROOT / "scripts" / "mcnemar_hit1_compare.py"


@dataclass(frozen=True)
class CohortPair:
    cohort: str
    hybrid_path: Path
    dense_path: Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run paired McNemar Hit@1 test across multiple cohorts.")
    p.add_argument("--out-dir", default="results/mcnemar_hit1_batch", help="Batch output directory.")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--allow-partial-overlap", action="store_true")
    p.add_argument(
        "--pair",
        action="append",
        default=[],
        help=(
            "Explicit cohort pair mapping in the format: "
            "cohort::/abs/or/rel/hybrid.json::/abs/or/rel/dense.json . "
            "Can be provided multiple times."
        ),
    )
    p.add_argument(
        "--cohort-prefix",
        default="Grampian-",
        help="Prefix used during auto-discovery from data_processed when --pair is not provided.",
    )
    p.add_argument(
        "--allow-incomplete-corpora",
        action="store_true",
        help="Include matching cohorts even if the source folder is missing canonical evaluation artifacts.",
    )
    return p.parse_args()


def _parse_pair_arg(value: str) -> CohortPair:
    parts = value.split("::")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid --pair '{value}'. Expected format: cohort::hybrid_path::dense_path"
        )
    cohort, hybrid, dense = parts
    hp = Path(hybrid).expanduser().resolve()
    dp = Path(dense).expanduser().resolve()
    if not hp.exists():
        raise FileNotFoundError(f"Hybrid file not found for cohort {cohort}: {hp}")
    if not dp.exists():
        raise FileNotFoundError(f"Dense file not found for cohort {cohort}: {dp}")
    return CohortPair(cohort=cohort, hybrid_path=hp, dense_path=dp)


def _autodiscover_pairs(prefix: str, allow_incomplete_corpora: bool) -> list[CohortPair]:
    root = REPO_ROOT / "data_processed"
    if not root.exists():
        raise FileNotFoundError(f"Missing data_processed directory: {root}")

    pairs: list[CohortPair] = []
    for cohort_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)):
        if not allow_incomplete_corpora and not is_eval_ready_doc_dir(cohort_dir):
            continue
        hybrid = cohort_dir / "retrieval_results_hybrid.json"
        dense = cohort_dir / "retrieval_results.json"
        if hybrid.exists() and dense.exists():
            pairs.append(CohortPair(cohort=cohort_dir.name, hybrid_path=hybrid.resolve(), dense_path=dense.resolve()))
    if not pairs:
        raise ValueError(
            f"No cohort pairs found in {root} with prefix '{prefix}' where both "
            "retrieval_results_hybrid.json and retrieval_results.json are present."
        )
    return pairs


def _run_single(pair: CohortPair, out_dir: Path, alpha: float, allow_partial_overlap: bool) -> Path:
    cmd = [
        sys.executable,
        str(MCNEMAR_SCRIPT),
        "--hybrid",
        str(pair.hybrid_path),
        "--dense",
        str(pair.dense_path),
        "--cohort",
        pair.cohort,
        "--out-dir",
        str(out_dir),
        "--alpha",
        str(alpha),
    ]
    if allow_partial_overlap:
        cmd.append("--allow-partial-overlap")
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))
    return out_dir / f"{pair.cohort}_mcnemar_hit1.json"


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[CohortPair]
    if args.pair:
        pairs = [_parse_pair_arg(p) for p in args.pair]
    else:
        pairs = _autodiscover_pairs(
            prefix=str(args.cohort_prefix),
            allow_incomplete_corpora=bool(args.allow_incomplete_corpora),
        )

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for pair in pairs:
        try:
            json_path = _run_single(
                pair=pair,
                out_dir=out_dir,
                alpha=float(args.alpha),
                allow_partial_overlap=bool(args.allow_partial_overlap),
            )
            result = json.loads(json_path.read_text(encoding="utf-8"))
            m = result["mcnemar"]
            c = result["contingency_table"]
            n = result["counts"]
            rows.append(
                {
                    "cohort": pair.cohort,
                    "n_paired_queries": int(n["n_paired_queries"]),
                    "both_correct": int(c["both_correct"]),
                    "both_wrong": int(c["both_wrong"]),
                    "hybrid_correct_dense_wrong": int(c["hybrid_correct_dense_wrong"]),
                    "hybrid_wrong_dense_correct": int(c["hybrid_wrong_dense_correct"]),
                    "n_discordant": int(n["n_discordant"]),
                    "method": str(m["method"]),
                    "exact_used": bool(m["exact_used"]),
                    "statistic": float(m["statistic"]),
                    "p_value": float(m["p_value"]),
                    "alpha": float(m["alpha"]),
                    "significant": bool(m["significant"]),
                    "json_result_path": str(json_path),
                    "hybrid_file": str(pair.hybrid_path),
                    "dense_file": str(pair.dense_path),
                }
            )
        except Exception as exc:
            errors.append({"cohort": pair.cohort, "error": str(exc)})

    summary_csv = out_dir / "mcnemar_hit1_batch_summary.csv"
    summary_json = out_dir / "mcnemar_hit1_batch_summary.json"

    fieldnames = [
        "cohort",
        "n_paired_queries",
        "both_correct",
        "both_wrong",
        "hybrid_correct_dense_wrong",
        "hybrid_wrong_dense_correct",
        "n_discordant",
        "method",
        "exact_used",
        "statistic",
        "p_value",
        "alpha",
        "significant",
        "json_result_path",
        "hybrid_file",
        "dense_file",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "batch_inputs": {
            "alpha": float(args.alpha),
            "allow_partial_overlap": bool(args.allow_partial_overlap),
            "n_requested_pairs": len(pairs),
            "n_successful": len(rows),
            "n_failed": len(errors),
        },
        "results": rows,
        "errors": errors,
    }
    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("Saved:", summary_csv)
    print("Saved:", summary_json)
    if errors:
        print("Completed with errors:")
        for e in errors:
            print(f"- {e['cohort']}: {e['error']}")


if __name__ == "__main__":
    main()
