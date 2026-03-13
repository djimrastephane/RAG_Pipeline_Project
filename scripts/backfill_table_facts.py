from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from rag_pdf.table_canonicalize import extract_table_facts_from_markdown


def parse_args() -> argparse.Namespace:
    """Parse CLI args for table-facts backfill."""
    parser = argparse.ArgumentParser(description="Backfill table_facts.parquet from tables_structured.parquet.")
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Processed document directory containing tables_structured.parquet.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate table_facts.parquet from existing tables_structured.parquet."""
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    in_path = data_dir / "tables_structured.parquet"
    out_path = data_dir / "table_facts.parquet"

    if not in_path.exists():
        raise FileNotFoundError(f"Missing input: {in_path}")

    tables = pd.read_parquet(in_path)
    if tables.empty:
        print(f"No tables in {in_path}")
        pd.DataFrame().to_parquet(out_path, index=False)
        print(f"Wrote empty: {out_path}")
        return

    facts: list[dict] = []
    for _, row in tables.iterrows():
        table_md = str(row.get("table_markdown") or "")
        facts.extend(
            extract_table_facts_from_markdown(
                table_md,
                doc_id=str(row.get("doc_id") or ""),
                corpus_id=str(row.get("corpus_id") or ""),
                report_year=str(row.get("report_year") or "") or None,
                period_end_date=str(row.get("period_end_date") or "") or None,
                run_date_utc=str(row.get("run_date_utc") or ""),
                page=int(row.get("page") or 0),
                table_id=str(row.get("table_id") or ""),
                table_type=str(row.get("table_type") or "unknown"),
            )
        )

    facts_df = pd.DataFrame(facts)
    facts_df.to_parquet(out_path, index=False)
    print(f"Input tables: {len(tables)}")
    print(f"Output facts: {len(facts_df)}")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
