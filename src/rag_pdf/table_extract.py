from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import camelot  # type: ignore
except Exception:
    camelot = None
    print("WARNING: camelot-py not installed. Table extraction will use pdfplumber only.")

from rag_pdf.chunking import count_tokens
from rag_pdf.config import DEFAULT_CONFIG
from rag_pdf.schemas import build_page_list_struct, make_chunk_id_global

CAMELOT_LATTICE_ACCURACY_THRESHOLD = DEFAULT_CONFIG.CAMELOT_LATTICE_ACCURACY_THRESHOLD
TABLE_SUMMARY_MAX_ROWS = DEFAULT_CONFIG.TABLE_SUMMARY_MAX_ROWS


def clean_table_dataframe(df: pd.DataFrame, table_type: str | None) -> pd.DataFrame:
    """
    Clean extracted table dataframe.

    Operations:
    - Strip whitespace from all cells
    - Remove empty rows
    - Convert numeric-looking strings to numbers
    - Set first row as column headers if appropriate

    Args:
        df: Raw extracted dataframe
        table_type: Detected table type (for type-specific cleaning)

    Returns:
        Cleaned dataframe
    """
    if df is None or len(df) == 0:
        return df

    # Strip whitespace
    df = df.map(lambda x: str(x).strip() if pd.notna(x) else "")

    # Remove fully empty rows
    df = df.loc[~(df == "").all(axis=1)]

    # Try to detect and set header row
    # If first row has mostly text and subsequent rows have numbers, promote it
    if len(df) > 1:
        first_row = df.iloc[0]
        has_text = sum(
            1 for v in first_row
            if str(v).strip() and not str(v).replace(",", "").replace(".", "").replace("-", "").isdigit()
        )

        if has_text / len(first_row) > 0.5:  # More than half are text labels
            df.columns = [str(v) for v in first_row]
            df = df.iloc[1:].reset_index(drop=True)

    return df


def extract_table_camelot(pdf_path: Path, page_no: int) -> pd.DataFrame | None:
    """
    Extract table using Camelot (lattice then stream methods).

    Args:
        pdf_path: Path to PDF file
        page_no: Page number (1-indexed)

    Returns:
        Extracted dataframe or None if extraction fails
    """
    if camelot is None:
        return None

    try:
        # Try lattice method first (works for bordered tables)
        tables = camelot.read_pdf(
            str(pdf_path),
            pages=str(page_no),
            flavor='lattice',
            strip_text='\n'
        )

        if len(tables) > 0 and tables[0].accuracy >= CAMELOT_LATTICE_ACCURACY_THRESHOLD:
            return tables[0].df
    except Exception:
        pass

    try:
        # Try stream method (works for borderless tables)
        tables = camelot.read_pdf(
            str(pdf_path),
            pages=str(page_no),
            flavor='stream'
        )

        if len(tables) > 0:
            return tables[0].df
    except Exception:
        pass

    return None


def extract_table_pdfplumber(pdf_plumber, page_no: int) -> pd.DataFrame | None:
    """
    Extract table using pdfplumber (fallback method).

    Args:
        pdf_plumber: Open pdfplumber PDF object
        page_no: Page number (1-indexed)

    Returns:
        Extracted dataframe or None if extraction fails
    """
    try:
        page_idx = page_no - 1
        page = pdf_plumber.pages[page_idx]
        table = page.extract_table()

        if table and len(table) > 1:
            df = pd.DataFrame(table[1:], columns=table[0])
            return df
    except Exception:
        pass

    return None


def extract_table_cells(
    pdf_path: Path,
    pdf_plumber,
    page_no: int,
    table_type: str | None,
) -> pd.DataFrame | None:
    """
    Multi-method table extraction with validation.

    Extraction cascade:
    1. Camelot lattice (bordered tables) - highest accuracy
    2. Camelot stream (borderless tables)
    3. pdfplumber (fallback)

    Args:
        pdf_path: Path to PDF file
        pdf_plumber: Open pdfplumber PDF object
        page_no: Page number (1-indexed)
        table_type: Detected table type (for type-specific cleaning)

    Returns:
        Cleaned dataframe or None if all methods fail
    """
    # Try Camelot methods
    df = extract_table_camelot(pdf_path, page_no)
    if df is not None and len(df) > 0:
        return clean_table_dataframe(df, table_type)

    # Fallback to pdfplumber
    df = extract_table_pdfplumber(pdf_plumber, page_no)
    if df is not None and len(df) > 0:
        return clean_table_dataframe(df, table_type)

    return None


def generate_table_summary(
    df: pd.DataFrame,
    table_type: str | None,
    page_no: int,
) -> str:
    """
    Convert structured table to searchable text description.

    Creates a natural language summary that can be indexed for RAG search
    while preserving key information for answer generation.

    Args:
        df: Parsed table dataframe
        table_type: Classified table type
        page_no: Source page number

    Returns:
        Text summary string

    Example output:
        "Financial table on page 111 (Cash Flow - Non-cash adjustments).
         Contains 8 line items across 6 columns.
         Key items: Depreciation | 25,586 | 25,586 | 24,825;
         Impairments | 6,985 | 6,985 | 1,343..."
    """
    rows, cols = df.shape

    summary_parts = [
        f"Financial table on page {page_no}",
    ]

    if table_type:
        type_label = table_type.replace("_", " ").title()
        summary_parts.append(f"({type_label})")

    summary_parts.append(f"Contains {rows} line items across {cols} columns.")

    # Extract key rows (first few non-empty rows)
    key_items = []
    for i in range(min(TABLE_SUMMARY_MAX_ROWS, len(df))):
        row_values = [str(v).strip() for v in df.iloc[i] if str(v).strip()]
        if len(row_values) >= 2:  # At least a label and one value
            row_text = " | ".join(row_values[:4])  # Limit to first 4 columns
            key_items.append(row_text)

    if key_items:
        summary_parts.append("Key items: " + "; ".join(key_items) + ".")

    return " ".join(summary_parts)


def process_table_pages(
    table_pages: list[dict],
    pdf_path: Path,
    pdf_plumber,
    doc_id: str,
    corpus_id: str,
    report_year: str | None,
    period_end_date: str | None,
    report_year_source: str | None,
    run_date_utc: str,
    enc,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract tables and create dual representation.

    For each table page:
    1. Extract structured cells → tables_structured.parquet
    2. Generate text summary → chunks.parquet (for RAG search)

    Args:
        table_pages: List of classified table pages
        pdf_path: Path to source PDF
        pdf_plumber: Open pdfplumber PDF object
        doc_id: Document identifier
        corpus_id: Corpus identifier
        report_year: Report year range
        period_end_date: Period end date (ISO format)
        report_year_source: Source of report year ("pdf_cover" or "filename")
        run_date_utc: Processing timestamp
        enc: Tiktoken encoder

    Returns:
        (text_chunks_df, structured_tables_df)
        - text_chunks_df: Table summaries for RAG (same schema as text chunks)
        - structured_tables_df: Parsed cells with metadata
    """
    text_chunks = []
    structured_tables = []

    for tpage in table_pages:
        page_no = tpage["page"]
        table_type = tpage["table_type"]

        # Extract structured table
        raw_table = extract_table_cells(
            pdf_path,
            pdf_plumber,
            page_no,
            table_type
        )

        if raw_table is not None and len(raw_table) > 0:
            # Store structured version
            table_id = f"table_p{page_no:04d}"

            structured_tables.append({
                "doc_id": doc_id,
                "corpus_id": corpus_id,
                "report_year": report_year,
                "period_end_date": period_end_date,
                "run_date_utc": run_date_utc,
                "page": page_no,
                "table_id": table_id,
                "table_type": table_type or "unknown",
                "rows": len(raw_table),
                "cols": len(raw_table.columns),
                "extraction_method": "camelot" if camelot else "pdfplumber",
            })

            # Create searchable text summary
            summary = generate_table_summary(raw_table, table_type, page_no)

            chunk_id_local = f"table_p{page_no:04d}"
            pages = [page_no]
            page_list_struct = build_page_list_struct(pages)

            text_chunks.append({
                "doc_id": doc_id,
                "corpus_id": corpus_id,
                "report_year": report_year,
                "report_year_source": report_year_source,
                "period_end_date": period_end_date,
                "run_date_utc": run_date_utc,
                "chunk_id": chunk_id_local,
                "chunk_id_global": make_chunk_id_global(doc_id, chunk_id_local),
                "part": "Unknown",  # Tables don't have part classification
                "section_title": "Financial Tables",
                "page_start": page_no,
                "page_end": page_no,
                "pages": pages,
                "page_list": page_list_struct,
                "chunk_text": summary,
                "chunk_tokens": count_tokens(summary, enc),
                "word_count": len(summary.split()),
                "is_table_like": True,
                "many_numbers": True,
                "is_table": True,  # NEW FLAG
                "table_type": table_type,  # NEW
                "table_ref": table_id,  # NEW: Link to structured data
            })

    return pd.DataFrame(text_chunks), pd.DataFrame(structured_tables)
