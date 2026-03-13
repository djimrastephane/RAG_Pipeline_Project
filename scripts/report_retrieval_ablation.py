from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for ablation report generation."""
    parser = argparse.ArgumentParser(description="Generate markdown/CSV report from ablation summary.")
    parser.add_argument(
        "--summary-csv",
        default="data_processed/ablation/retrieval_ablation_summary.csv",
        help="Input ablation summary CSV.",
    )
    parser.add_argument(
        "--out-md",
        default="data_processed/ablation/retrieval_ablation_report.md",
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--out-csv",
        default="data_processed/ablation/retrieval_ablation_ranked.csv",
        help="Output ranked CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    """Load ablation summary and write ranked outputs for quick comparison."""
    args = parse_args()
    summary_path = Path(args.summary_csv)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary CSV not found: {summary_path}")

    df = pd.read_csv(summary_path)
    if df.empty:
        raise RuntimeError("Summary CSV is empty.")

    ranked = df.sort_values(["k", "page_hit_rate", "page_mrr", "page_precision"], ascending=[True, False, False, False])
    best = ranked.groupby("k", as_index=False).head(1).reset_index(drop=True)

    out_csv = Path(args.out_csv)
    out_md = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    ranked.to_csv(out_csv, index=False)
    report_lines = [
        "# Retrieval Ablation Report",
        "",
        "## Best Experiment Per k",
        "",
        best.to_markdown(index=False),
        "",
        "## Full Ranking",
        "",
        ranked.to_markdown(index=False),
        "",
    ]
    out_md.write_text("\n".join(report_lines), encoding="utf-8")

    print("Saved:", out_csv)
    print("Saved:", out_md)


if __name__ == "__main__":
    main()
