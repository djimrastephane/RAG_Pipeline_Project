"""
Pipeline validation and inspection script.

Quick checks after running the preprocessing pipeline:
- Verify all output files exist
- Show table extraction results
- Display sample chunks
- Validate page citations
"""

from pathlib import Path
import pandas as pd
import json

DOC_ID = "Grampian-2022-2023"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = PROJECT_ROOT / "data_processed"


def load_and_validate():
    """Load all outputs and run validation checks."""
    doc_dir = OUT_ROOT / DOC_ID

    print(f"\n{'=' * 70}")
    print(f"PIPELINE OUTPUT VALIDATION: {DOC_ID}")
    print(f"{'=' * 70}\n")

    # Check file existence
    expected_files = [
        "pages.parquet",
        "sections.parquet",
        "chunks.parquet",
        "metrics.json",
    ]

    print("1. FILE EXISTENCE CHECK")
    print("-" * 70)
    for file_name in expected_files:
        file_path = doc_dir / file_name
        status = "✓" if file_path.exists() else "✗ MISSING"
        print(f"  {status} {file_name}")

    # Check optional table file
    tables_file = doc_dir / "tables_structured.parquet"
    if tables_file.exists():
        print(f"  ✓ tables_structured.parquet")

    print()

    # Load metrics
    metrics_file = doc_dir / "metrics.json"
    if not metrics_file.exists():
        print("Cannot proceed - metrics.json missing")
        return

    with open(metrics_file) as f:
        metrics = json.load(f)

    print("2. PIPELINE STATISTICS")
    print("-" * 70)
    counts = metrics.get("counts", {})
    for key, val in counts.items():
        print(f"  {key:<25} {val:>6}")
    print()

    # Table type breakdown
    if "table_types_detected" in metrics and metrics["table_types_detected"]:
        print("3. TABLE TYPES DETECTED")
        print("-" * 70)
        for table_type, count in metrics["table_types_detected"].items():
            print(f"  {table_type:<30} {count:>3} tables")
        print()

    # Load chunks for validation
    chunks_file = doc_dir / "chunks.parquet"
    if not chunks_file.exists():
        return

    chunks_df = pd.read_parquet(chunks_file)

    print("4. CHUNK VALIDATION")
    print("-" * 70)

    # Check page spans
    bad_spans = chunks_df[chunks_df["page_start"] != chunks_df["page_end"]]
    if len(bad_spans) > 0:
        print(f"  ✗ WARNING: {len(bad_spans)} chunks span multiple pages")
    else:
        print(f"  ✓ All chunks are page-bounded")

    # Check chunk IDs are unique
    dupes = chunks_df["chunk_id_global"].duplicated().sum()
    if dupes > 0:
        print(f"  ✗ WARNING: {dupes} duplicate chunk_id_global values")
    else:
        print(f"  ✓ All chunk_id_global values are unique")

    # Check pages field
    if "pages" in chunks_df.columns:
        missing_pages = chunks_df["pages"].isna().sum()
        if missing_pages > 0:
            print(f"  ✗ WARNING: {missing_pages} chunks missing 'pages' field")
        else:
            print(f"  ✓ All chunks have 'pages' field populated")

    print()

    # Show chunk distribution
    print("5. CHUNK DISTRIBUTION")
    print("-" * 70)
    if "is_table" in chunks_df.columns:
        text_chunks = (~chunks_df["is_table"]).sum()
        table_chunks = chunks_df["is_table"].sum()
        print(f"  Text chunks:  {text_chunks:>6}")
        print(f"  Table chunks: {table_chunks:>6}")
    else:
        print(f"  Total chunks: {len(chunks_df):>6}")

    print()

    # Sample chunks
    print("6. SAMPLE CHUNKS")
    print("-" * 70)

    # Show first text chunk
    text_samples = chunks_df[~chunks_df.get("is_table", False)]
    if len(text_samples) > 0:
        sample = text_samples.iloc[0]
        print(f"  [TEXT CHUNK]")
        print(f"  ID: {sample['chunk_id_global']}")
        print(f"  Page: {sample['page_start']}")
        print(f"  Section: {sample['section_title']}")
        print(f"  Tokens: {sample['chunk_tokens']}")
        print(f"  Preview: {sample['chunk_text'][:200]}...")
        print()

    # Show first table chunk
    table_samples = chunks_df[chunks_df.get("is_table", False)]
    if len(table_samples) > 0:
        sample = table_samples.iloc[0]
        print(f"  [TABLE CHUNK]")
        print(f"  ID: {sample['chunk_id_global']}")
        print(f"  Page: {sample['page_start']}")
        print(f"  Table Type: {sample.get('table_type', 'unknown')}")
        print(f"  Table Ref: {sample.get('table_ref', 'N/A')}")
        print(f"  Preview: {sample['chunk_text'][:200]}...")
        print()

    # Check structured tables
    if tables_file.exists():
        tables_df = pd.read_parquet(tables_file)

        print("7. STRUCTURED TABLES")
        print("-" * 70)
        print(f"  Total tables extracted: {len(tables_df)}")

        if len(tables_df) > 0:
            print(f"\n  Table inventory:")
            for _, row in tables_df.iterrows():
                print(f"    Page {row['page']:>3}: {row['table_type']:<20} ({row['rows']}x{row['cols']})")
        print()

    print(f"{'=' * 70}")
    print("VALIDATION COMPLETE")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    load_and_validate()
