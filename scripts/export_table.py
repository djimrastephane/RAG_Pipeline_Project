"""
Export individual table CSVs from structured tables parquet.

Usage:
    python export_table_csvs.py

Reads tables_structured.parquet and for each table, fetches the raw data
and exports to tables_raw/<table_id>.csv
"""

from pathlib import Path
import pandas as pd
import pdfplumber
import camelot

# CONFIG
DOC_ID = "Grampian-2022-2023"
OUT_ROOT = Path("/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed")
PDF_PATH = Path("/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/Data/Annual Accounts NHS Grampian/Preliminary_Test/Grampian-2022-2023.pdf")


def extract_table_to_csv(
        pdf_path: Path,
        page_no: int,
        output_path: Path
) -> bool:
    """
    Extract table from PDF page and save as CSV.

    Args:
        pdf_path: Source PDF file
        page_no: Page number (1-indexed)
        output_path: Output CSV path

    Returns:
        True if successful, False otherwise
    """
    # Try Camelot first
    try:
        tables = camelot.read_pdf(
            str(pdf_path),
            pages=str(page_no),
            flavor='lattice',
        )
        if len(tables) > 0:
            df = tables[0].df
            df.to_csv(output_path, index=False)
            return True
    except Exception:
        pass

    # Fallback to pdfplumber
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            page = pdf.pages[page_no - 1]
            table = page.extract_table()
            if table:
                df = pd.DataFrame(table)
                df.to_csv(output_path, index=False, header=False)
                return True
    except Exception:
        pass

    return False


def main():
    """Export all tables to individual CSV files."""
    doc_dir = OUT_ROOT / DOC_ID
    tables_file = doc_dir / "tables_structured.parquet"

    if not tables_file.exists():
        print(f"No tables file found: {tables_file}")
        return

    tables_df = pd.read_parquet(tables_file)

    if len(tables_df) == 0:
        print("No tables to export")
        return

    # Create output directory
    raw_dir = doc_dir / "tables_raw"
    raw_dir.mkdir(exist_ok=True)

    print(f"Exporting {len(tables_df)} tables...")

    success_count = 0
    for _, row in tables_df.iterrows():
        table_id = row["table_id"]
        page_no = int(row["page"])
        table_type = row["table_type"]

        # Generate filename
        filename = f"{table_id}_{table_type}.csv"
        output_path = raw_dir / filename

        # Extract and save
        if extract_table_to_csv(PDF_PATH, page_no, output_path):
            print(f"  ✓ {filename}")
            success_count += 1
        else:
            print(f"  ✗ {filename} (extraction failed)")

    print(f"\nExported {success_count}/{len(tables_df)} tables to: {raw_dir}")


if __name__ == "__main__":
    main()