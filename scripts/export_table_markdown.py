"""
Export table markdown files from tables_structured.parquet.

Usage:
    python scripts/export_table_markdown.py --doc-id Grampian-2024-2025
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export table markdown files.")
    parser.add_argument(
        "--doc-id",
        required=True,
        help="Document id matching a folder in data_processed.",
    )
    parser.add_argument(
        "--out-root",
        default="data_processed",
        help="Root output directory (default: data_processed).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional override for markdown output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_root)
    doc_dir = out_root / args.doc_id
    tables_file = doc_dir / "tables_structured.parquet"

    if not tables_file.exists():
        print(f"Missing: {tables_file}")
        return

    tables_df = pd.read_parquet(tables_file)
    if len(tables_df) == 0:
        print("No tables found.")
        return

    out_dir = Path(args.out_dir) if args.out_dir else doc_dir / "tables_markdown"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for _, row in tables_df.iterrows():
        table_id = str(row.get("table_id") or "").strip()
        if not table_id:
            continue
        table_type = str(row.get("table_type") or "unknown").strip()
        page = row.get("page")
        summary = str(row.get("table_summary") or "").strip()
        markdown = str(row.get("table_markdown") or "").strip()
        if not markdown:
            continue

        filename = f"{table_id}_{table_type}.md"
        out_path = out_dir / filename

        header_lines = [
            f"# {table_id} ({table_type})",
            f"page: {page}",
            "",
        ]
        if summary:
            header_lines.extend([summary, ""])

        out_path.write_text("\n".join(header_lines) + markdown + "\n", encoding="utf-8")
        written += 1

    print(f"Wrote {written} markdown tables to: {out_dir}")


if __name__ == "__main__":
    main()
