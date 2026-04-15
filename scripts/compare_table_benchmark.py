from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare benchmark per-page results between current and Docling extractors."
    )
    parser.add_argument(
        "--per-page-csv",
        type=str,
        default="",
        help="Path to table_extract_benchmark_per_page_*.csv. If omitted, latest in data_processed/benchmarks is used.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data_processed/benchmarks",
        help="Directory for comparison outputs.",
    )
    parser.add_argument(
        "--speed-weight",
        type=float,
        default=0.4,
        help="Weight for speed in combined score [0..1]. Coverage weight = 1 - speed_weight.",
    )
    return parser.parse_args()


def _resolve_input_csv(path_arg: str) -> Path:
    if path_arg:
        p = Path(path_arg).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Input CSV not found: {p}")
        return p

    bench_dir = Path("data_processed/benchmarks").resolve()
    matches = sorted(bench_dir.glob("table_extract_benchmark_per_page_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No benchmark per-page CSV found in {bench_dir}")
    return matches[-1]


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def _pick_winner(v_cur: float, v_doc: float, higher_is_better: bool) -> str:
    if v_cur == v_doc:
        return "tie"
    if higher_is_better:
        return "current" if v_cur > v_doc else "docling"
    return "current" if v_cur < v_doc else "docling"


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.speed_weight <= 1.0:
        raise ValueError("--speed-weight must be between 0 and 1.")

    in_csv = _resolve_input_csv(args.per_page_csv)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_csv)
    needed = {"pdf", "page", "extractor", "elapsed_ms", "rows", "cols", "non_empty_cells", "numeric_cells", "success"}
    missing = sorted(needed - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    df = df[df["extractor"].isin(["current", "docling"])].copy()
    if df.empty:
        raise ValueError("No current/docling rows found in input CSV.")

    page_rows: list[dict] = []
    for (pdf, page), grp in df.groupby(["pdf", "page"], dropna=False):
        cur = grp[grp["extractor"] == "current"]
        doc = grp[grp["extractor"] == "docling"]
        if cur.empty or doc.empty:
            continue
        c = cur.iloc[0]
        d = doc.iloc[0]

        c_elapsed = float(c["elapsed_ms"])
        d_elapsed = float(d["elapsed_ms"])
        c_rows = float(c["rows"])
        d_rows = float(d["rows"])
        c_numeric = float(c["numeric_cells"])
        d_numeric = float(d["numeric_cells"])
        c_non_empty = float(c["non_empty_cells"])
        d_non_empty = float(d["non_empty_cells"])
        c_success = bool(c["success"])
        d_success = bool(d["success"])

        if not c_success and not d_success:
            speed_score_cur = 0.0
            speed_score_doc = 0.0
            cov_score_cur = 0.0
            cov_score_doc = 0.0
        else:
            speed_score_cur = _safe_ratio(d_elapsed, c_elapsed + d_elapsed)
            speed_score_doc = _safe_ratio(c_elapsed, c_elapsed + d_elapsed)

            cov_cur_raw = c_rows + c_numeric + 0.2 * c_non_empty
            cov_doc_raw = d_rows + d_numeric + 0.2 * d_non_empty
            total_cov = cov_cur_raw + cov_doc_raw
            cov_score_cur = _safe_ratio(cov_cur_raw, total_cov)
            cov_score_doc = _safe_ratio(cov_doc_raw, total_cov)

        coverage_w = 1.0 - args.speed_weight
        combined_cur = args.speed_weight * speed_score_cur + coverage_w * cov_score_cur
        combined_doc = args.speed_weight * speed_score_doc + coverage_w * cov_score_doc

        page_rows.append(
            {
                "pdf": pdf,
                "page": int(page),
                "table_type": c.get("table_type", d.get("table_type")),
                "current_elapsed_ms": c_elapsed,
                "docling_elapsed_ms": d_elapsed,
                "speedup_current_vs_docling": _safe_ratio(d_elapsed, c_elapsed) if c_elapsed > 0 else 0.0,
                "current_rows": int(c_rows),
                "docling_rows": int(d_rows),
                "current_numeric_cells": int(c_numeric),
                "docling_numeric_cells": int(d_numeric),
                "current_success": c_success,
                "docling_success": d_success,
                "speed_winner": _pick_winner(c_elapsed, d_elapsed, higher_is_better=False),
                "coverage_winner": _pick_winner(c_rows + c_numeric, d_rows + d_numeric, higher_is_better=True),
                "combined_score_current": combined_cur,
                "combined_score_docling": combined_doc,
                "combined_winner": _pick_winner(combined_cur, combined_doc, higher_is_better=True),
            }
        )

    if not page_rows:
        raise RuntimeError("No comparable page pairs found (need both current and docling rows per page).")

    page_df = pd.DataFrame(page_rows).sort_values(["pdf", "page"]).reset_index(drop=True)

    summary = {
        "input_csv": str(in_csv),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pages_compared": int(len(page_df)),
        "weights": {
            "speed_weight": args.speed_weight,
            "coverage_weight": 1.0 - args.speed_weight,
        },
        "winners": {
            "speed": page_df["speed_winner"].value_counts().to_dict(),
            "coverage": page_df["coverage_winner"].value_counts().to_dict(),
            "combined": page_df["combined_winner"].value_counts().to_dict(),
        },
        "means": {
            "current_elapsed_ms": float(page_df["current_elapsed_ms"].mean()),
            "docling_elapsed_ms": float(page_df["docling_elapsed_ms"].mean()),
            "current_rows": float(page_df["current_rows"].mean()),
            "docling_rows": float(page_df["docling_rows"].mean()),
            "current_numeric_cells": float(page_df["current_numeric_cells"].mean()),
            "docling_numeric_cells": float(page_df["docling_numeric_cells"].mean()),
            "speedup_current_vs_docling": float(page_df["speedup_current_vs_docling"].mean()),
        },
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_page = out_dir / f"table_extract_comparison_per_page_{stamp}.csv"
    out_summary = out_dir / f"table_extract_comparison_summary_{stamp}.json"
    page_df.to_csv(out_page, index=False)
    out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Input:        {in_csv}")
    print(f"Per-page out: {out_page}")
    print(f"Summary out:  {out_summary}")
    print("")
    print("Winner counts:")
    print(f"- speed:    {summary['winners']['speed']}")
    print(f"- coverage: {summary['winners']['coverage']}")
    print(f"- combined: {summary['winners']['combined']}")
    print("")
    print("Means:")
    print(f"- current_elapsed_ms: {summary['means']['current_elapsed_ms']:.2f}")
    print(f"- docling_elapsed_ms: {summary['means']['docling_elapsed_ms']:.2f}")
    print(f"- current_rows: {summary['means']['current_rows']:.2f}")
    print(f"- docling_rows: {summary['means']['docling_rows']:.2f}")
    print(f"- current_numeric_cells: {summary['means']['current_numeric_cells']:.2f}")
    print(f"- docling_numeric_cells: {summary['means']['docling_numeric_cells']:.2f}")
    print(f"- speedup_current_vs_docling: {summary['means']['speedup_current_vs_docling']:.2f}x")


if __name__ == "__main__":
    main()
