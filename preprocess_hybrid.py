from __future__ import annotations

"""
Hybrid PDF preprocessing for RAG - Text + Tables

GOAL
Convert a digital PDF into:
1) Page-level cleaned text with metadata and page citations
2) Semantic sections for context
3) Text chunks for retrieval
4) DUAL TABLE REPRESENTATION:
   - Text summaries (searchable via RAG)
   - Structured CSV data (for precise queries)

ARCHITECTURE
- Text pages → standard RAG chunking pipeline
- Table pages → dual output:
  * Searchable summary chunks (merged into chunks.parquet)
  * Structured cell data (tables_structured.parquet + CSV files)

WHY HYBRID APPROACH
Financial Q&A systems need both:
- Semantic search: "What were the performance highlights?"
- Precise lookups: "What was depreciation in 2023?"

Tables get indexed as text for discovery, but preserve structure for accuracy.

OUTPUTS
OUT_ROOT/<DOC_ID>/
  chunks.parquet           Unified text + table summary chunks (for RAG)
  tables_structured.parquet Parsed table cells with metadata
  tables_raw/              Individual table CSVs
  pages.parquet            Page-level text + table flags
  sections.parquet         Inferred sections
  metrics.json             Pipeline stats

DEPENDENCIES
pip install pymupdf pdfplumber pandas pyarrow camelot-py[cv] tiktoken

For Camelot (table extraction):
- Linux/Mac: ghostscript installed
- All platforms: pip install camelot-py[cv]
"""

import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Optional

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError(
        "Failed to import PyMuPDF.\n"
        "Fix: pip uninstall -y fitz frontend && pip install -U pymupdf\n"
    ) from e

import pdfplumber
import pandas as pd

try:
    import camelot  # type: ignore
except Exception:
    camelot = None
    print("WARNING: camelot-py not installed. Table extraction will use pdfplumber only.")

from rag_pdf.chunking import chunk_text_by_tokens, count_tokens, get_encoder
from rag_pdf.extract_page import extract_page_struct_hybrid
from rag_pdf.metrics import StepTimer, safe_json_dump
from rag_pdf.schemas import build_page_list_struct, make_chunk_id_global
from rag_pdf.text_normalize import (
    extract_report_metadata_from_pdf,
    extract_report_year_from_filename,
    normalize_line,
    normalize_page_text,
    now_utc_iso,
)


# =============================================================================
# CONFIG
# =============================================================================
PDF_PATH = Path(
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/Data/Annual Accounts NHS Grampian/Preliminary_Test/Grampian-2022-2023.pdf"
)
DOC_ID = PDF_PATH.stem

OUT_ROOT = Path("/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed")

# Optional: stable identifier for multi-document experiments
CORPUS_ID: Optional[str] = None

# Chunking settings
CHUNK_SIZE_TOKENS = 320
CHUNK_OVERLAP_TOKENS = 90

# Boilerplate removal (coordinate strips)
TOP_STRIP_FRAC = 0.08
BOTTOM_STRIP_FRAC = 0.08
LEFT_STRIP_FRAC = 0.08
RIGHT_STRIP_FRAC = 0.08

# Extra repetition-based removal
HEADER_FOOTER_REPEAT_FRAC = 0.40
TOP_LINE_K = 5
BOT_LINE_K = 5

# Heading detection
HEADING_MAX_CHARS = 110
HEADING_MIN_CHARS = 4
HEADING_FONT_BOOST_FRAC = 0.85

# Filters
MIN_CHUNK_WORDS = 20

# Hybrid loader settings
PRIMARY_EXTRACTOR = "pymupdf"
FALLBACK_MIN_CHARS = 80
FALLBACK_ON_BAD_TEXT = True
FALLBACK_ON_EXCEPTION = True

# Table detection thresholds
TABLE_DIGIT_RATIO = 0.15  # Raised: NHS tables have high numeric content
TABLE_SPACE_RATIO = 0.3  # Lowered: Tables may be collapsed after cleanup
TABLE_MIN_LINES = 1  # Accept single-line (post-cleanup artifacts)

# Table extraction settings
CAMELOT_LATTICE_ACCURACY_THRESHOLD = 70
TABLE_SUMMARY_MAX_ROWS = 5

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def is_part_label(line: str) -> str | None:
    """
    Detect report part labels (e.g., 'Part A', 'Part B').

    Returns:
        Part label if detected, else None
    """
    if re.search(r"\bPart\s+A\b", line, flags=re.IGNORECASE):
        return "Part A"
    if re.search(r"\bPart\s+B\b", line, flags=re.IGNORECASE):
        return "Part B"
    return None


def looks_like_numbered_heading(line: str) -> bool:
    """Check if line matches numbered heading pattern (e.g., '1.2.3 Heading Text')."""
    return bool(re.match(r"^\s*\d+(\.\d+)*\s+.+", line))


def looks_like_heading_text_only(line: str) -> bool:
    """
    Heuristic check for heading-like text.

    Criteria:
    - Reasonable length (4-110 chars)
    - No sentence-ending punctuation
    - No bullet points
    - Mostly capitalized words
    """
    line = line.strip()
    if len(line) < HEADING_MIN_CHARS or len(line) > HEADING_MAX_CHARS:
        return False
    if line.endswith((".", ":", ";")):
        return False
    if re.search(r"[•\u2022]", line):
        return False
    if re.search(r"\bpage\s+\d+\b", line, flags=re.IGNORECASE):
        return False
    if line.lower() in {"contents", "table of contents"}:
        return False

    words = [w for w in line.split() if w]
    if not words:
        return False

    cap_count = sum(w[:1].isupper() for w in words)
    few_punct = not re.search(r"[.,;]", line)
    return few_punct and (cap_count >= max(2, len(words) // 2) or looks_like_numbered_heading(line))


def is_table_like(text: str) -> bool:
    """
    Heuristic check if text content represents a table.

    Criteria:
    - At least 4 lines
    - High digit ratio (>12%)
    - Many double-spaces (column separation)
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < TABLE_MIN_LINES:
        return False
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text))
    many_spaces = sum(l.count("  ") for l in lines) / max(1, len(lines))
    return digit_ratio > TABLE_DIGIT_RATIO and many_spaces > TABLE_SPACE_RATIO


def is_table_like_from_raw_lines(lines: list[str]) -> bool:
    """
    Check if raw lines (before cleanup) look like a table.

    More lenient than is_table_like() because it checks the structure
    before boilerplate removal may have collapsed the layout.

    Criteria:
    - High digit content (>15%)
    - Contains common table keywords
    - Multiple lines with consistent spacing patterns
    """
    if not lines or len(lines) < 2:
        return False

    text = "\n".join(lines)

    # Check digit ratio
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text))
    if digit_ratio < 0.15:
        return False

    # Check for financial table keywords
    text_lower = text.lower()
    table_keywords = [
        "note", "£", "£000", "£'000", "2022/23", "2021/22",
        "total", "balance", "expenditure", "income", "assets",
        "liabilities", "depreciation", "impairment",
    ]
    keyword_hits = sum(1 for kw in table_keywords if kw in text_lower)

    # Strong signal: multiple keywords + high digits
    if keyword_hits >= 2 and digit_ratio > 0.15:
        return True

    # Check for tabular spacing (multiple aligned columns)
    lines_with_content = [l for l in lines if l.strip()]
    if len(lines_with_content) >= 3:
        # Look for consistent spacing patterns (tabs or multiple spaces)
        multi_space_lines = sum(1 for l in lines_with_content if "  " in l or "\t" in l)
        if multi_space_lines / len(lines_with_content) > 0.5:
            return True

    return False


def contains_many_numbers(text: str) -> bool:
    """Check if text has high numeric content (>10% digits)."""
    digits = sum(ch.isdigit() for ch in text)
    return digits / max(1, len(text)) > 0.10


def is_section_anchor_line(line: str) -> bool:
    """
    Detect semantic section anchor lines (report bands).

    Intended positives:
    - PERFORMANCE REPORT
    - ACCOUNTABILITY REPORT
    - CORPORATE GOVERNANCE REPORT

    Intended negatives:
    - NHS GRAMPIAN (organization name)
    - ANNUAL REPORT AND ACCOUNTS (global boilerplate)

    This is conservative to keep true section markers while removing
    global document boilerplate.
    """
    if not isinstance(line, str):
        return False

    s = re.sub(r"\s+", " ", line).strip()
    if not s:
        return False

    # Reasonable length for banner label
    if len(s) < 8 or len(s) > 48:
        return False

    # Must contain anchor keywords
    anchor_kw = {"REPORT", "ANALYSIS", "STATEMENT", "GOVERNANCE"}
    words = re.findall(r"[A-Za-z]+", s.upper())
    if not words:
        return False
    if not any(w in anchor_kw for w in words):
        return False

    # Exclude global boilerplate tokens
    hard_exclude = {
        "ANNUAL", "ACCOUNTS", "YEAR", "ENDED",
        "NHS", "BOARD", "SCOTLAND", "GRAMPIAN",
    }
    if any(w in hard_exclude for w in words):
        return False

    # Exclude date/year patterns
    if re.search(r"\b(?:19|20)\d{2}\b", s):
        return False
    if re.search(r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)\b", s.upper()):
        return False

    # Require uppercase style (typical for banner bars)
    alpha = [c for c in s if c.isalpha()]
    if not alpha:
        return False
    upper_ratio = sum(c.isupper() for c in alpha) / len(alpha)
    if upper_ratio < 0.80:
        return False

    # Avoid pure separator lines
    if re.fullmatch(r"[-–—_= ]+", s):
        return False

    return True


# =============================================================================
# TABLE CLASSIFICATION
# =============================================================================
def detect_table_type(text: str) -> str | None:
    """
    Classify financial table type from text content.

    Uses keyword matching on normalized text.

    Recognized types:
    - cash_flow: Cash flow statements and non-cash adjustments
    - balance_sheet: Statement of financial position
    - income_statement: Income/expenditure statements (SOCNE)
    - staff_costs: Employee benefits and staff costs
    - property: Property, plant, and equipment (PPE)
    - provisions: Provisions and liabilities
    - unknown: Unrecognized table type

    Args:
        text: Page text content

    Returns:
        Table type string or None if not a table
    """
    text_norm = normalize_line(text.lower())

    patterns = {
        "cash_flow": [
            "cash flow",
            "non-cash transaction",
            "note 2a",
            "note 2b",
            "reconciliation of net cash",
        ],
        "balance_sheet": [
            "balance sheet",
            "statement of financial position",
            "net assets",
            "total assets",
        ],
        "income_statement": [
            "statement of comprehensive net expenditure",
            "socne",
            "income and expenditure",
            "operating costs",
        ],
        "staff_costs": [
            "staff costs",
            "employee benefit",
            "remuneration",
            "pension costs",
        ],
        "property": [
            "property, plant and equipment",
            "ppe",
            "intangible assets",
            "additions to assets",
        ],
        "provisions": [
            "provisions",
            "contingent liabilities",
            "clinical negligence",
        ],
        "financial_instruments": [
            "financial instruments",
            "financial assets",
            "financial liabilities",
        ],
    }

    # Score each table type by keyword matches
    scores = {}
    for table_type, keywords in patterns.items():
        score = sum(1 for kw in keywords if kw in text_norm)
        if score > 0:
            scores[table_type] = score

    if not scores:
        return "unknown"

    # Return highest scoring type
    return max(scores.items(), key=lambda x: x[1])[0]


def classify_page_content(text: str) -> dict:
    """
    Enhanced page classification with table subtyping.

    Returns:
        {
            "is_table": bool,
            "table_type": str or None,
            "is_text": bool,
            "has_numbers": bool,
            "confidence": str  # "high", "medium", "low"
        }
    """
    is_tbl = is_table_like(text)
    table_type = detect_table_type(text) if is_tbl else None

    # Confidence scoring
    confidence = "low"
    if is_tbl and table_type and table_type != "unknown":
        confidence = "high"
    elif is_tbl:
        confidence = "medium"

    return {
        "is_table": is_tbl,
        "table_type": table_type if table_type != "unknown" else None,
        "is_text": not is_tbl,
        "has_numbers": contains_many_numbers(text),
        "confidence": confidence,
    }


# =============================================================================
# TABLE EXTRACTION
# =============================================================================
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
        has_text = sum(1 for v in first_row
                       if str(v).strip() and not str(v).replace(",", "").replace(".", "").replace("-", "").isdigit())

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
        table_type: str | None
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
        page_no: int
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
        enc
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


# =============================================================================
# PDF EXTRACTION AND CLEANUP
# =============================================================================
def strip_by_coordinates(
        lines_all: list[dict],
        *,
        page_height: float,
        page_width: float,
        rotation: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Strip boilerplate using page coordinates (orientation-aware).

    Portrait pages: Strip top and bottom
    Rotated/landscape pages: Strip left and right

    Returns:
        (kept_lines, removed_primary, removed_secondary)
    """
    rot = rotation % 360
    is_rotated = rot in (90, 270)
    is_landscape = page_width / max(page_height, 1.0) > 1.2
    use_side_strips = is_rotated or is_landscape

    kept: list[str] = []
    removed_a: list[str] = []
    removed_b: list[str] = []

    if use_side_strips:
        left_x = page_width * LEFT_STRIP_FRAC
        right_x = page_width * (1.0 - RIGHT_STRIP_FRAC)

        for ln in lines_all:
            x_mid = (ln["x0"] + ln["x1"]) / 2.0
            txt = ln["text"]
            if x_mid <= left_x:
                removed_a.append(txt)
                continue
            if x_mid >= right_x:
                removed_b.append(txt)
                continue
            kept.append(txt)

        return kept, removed_a, removed_b

    top_y = page_height * TOP_STRIP_FRAC
    bot_y = page_height * (1.0 - BOTTOM_STRIP_FRAC)

    for ln in lines_all:
        y_mid = (ln["y0"] + ln["y1"]) / 2.0
        txt = ln["text"]
        if y_mid <= top_y:
            removed_a.append(txt)
            continue
        if y_mid >= bot_y:
            removed_b.append(txt)
            continue
        kept.append(txt)

    return kept, removed_a, removed_b


def remove_repeated_header_footer_lines(
        pages_text_lines: dict[int, list[str]]
) -> tuple[dict[int, list[str]], set[str], set[str]]:
    """
    Remove repeated header/footer lines across pages.

    Lines appearing in top/bottom K positions on ≥40% of pages
    are considered boilerplate and removed.
    """
    top_lines: list[str] = []
    bot_lines: list[str] = []

    for _, ls in pages_text_lines.items():
        norm = [normalize_line(x) for x in ls if normalize_line(x)]
        top_lines.extend(norm[:TOP_LINE_K])
        bot_lines.extend(norm[-BOT_LINE_K:])

    top_counts = Counter(top_lines)
    bot_counts = Counter(bot_lines)
    threshold = int(HEADER_FOOTER_REPEAT_FRAC * len(pages_text_lines))

    common_header = {l for l, c in top_counts.items() if c >= threshold}
    common_footer = {l for l, c in bot_counts.items() if c >= threshold}

    cleaned: dict[int, list[str]] = {}
    for pno, ls in pages_text_lines.items():
        norm = [normalize_line(x) for x in ls if normalize_line(x)]
        out: list[str] = []
        for i, l in enumerate(norm):
            if i < TOP_LINE_K and l in common_header:
                continue
            if i >= len(norm) - BOT_LINE_K and l in common_footer:
                continue
            out.append(l)
        cleaned[pno] = out

    return cleaned, common_header, common_footer


def select_heading_candidates(lines_all: list[dict], page_p95_size: float) -> list[str]:
    """
    Select potential heading lines using font size and text heuristics.

    Only checks first 30 lines of page (headings appear near top).
    """
    size_thr = page_p95_size * HEADING_FONT_BOOST_FRAC if page_p95_size else 0.0
    cands: list[str] = []

    for ln in lines_all[:30]:
        txt = ln["text"]
        if not txt:
            continue
        if is_part_label(txt):
            continue
        if looks_like_heading_text_only(txt) and float(ln.get("max_size", 0.0)) >= size_thr:
            cands.append(txt)

    return cands


# =============================================================================
# SECTION BUILDING
# =============================================================================
def build_sections_from_pages(pages_df: pd.DataFrame) -> pd.DataFrame:
    """
    Infer document sections from page-level headings.

    Sections are bounded by heading detections. Each section spans
    from its heading page to the page before the next heading.
    """
    sections = []
    current_part = None
    current_section = "Unknown"
    current_pages: list[int] = []
    current_texts: list[str] = []

    def flush():
        if not current_pages:
            return
        sections.append({
            "doc_id": pages_df["doc_id"].iloc[0],
            "report_year": pages_df["report_year"].iloc[0],
            "period_end_date": pages_df["period_end_date"].iloc[0],
            "report_year_source": pages_df["report_year_source"].iloc[0],
            "run_date_utc": pages_df["run_date_utc"].iloc[0],
            "part": current_part or "Unknown",
            "section_title": current_section or "Unknown",
            "page_start": int(min(current_pages)),
            "page_end": int(max(current_pages)),
            "section_text": "\n".join(current_texts).strip(),
        })
        current_pages.clear()
        current_texts.clear()

    for _, row in pages_df.iterrows():
        page_no = int(row["page"])
        text = str(row["clean_text"] or "")
        lines = [normalize_line(x) for x in text.splitlines() if normalize_line(x)]

        # Check for part labels
        for l in lines[:25]:
            p = is_part_label(l)
            if p:
                current_part = p
                break

        # Check for headings
        heading_candidates = row.get("heading_candidates", [])
        heading_found = None

        if isinstance(heading_candidates, list) and heading_candidates:
            heading_found = heading_candidates[0]
        else:
            for l in lines[:25]:
                if looks_like_heading_text_only(l) and not is_part_label(l):
                    heading_found = l
                    break

        if heading_found and current_texts:
            flush()
            current_section = heading_found

        current_pages.append(page_no)
        current_texts.append(text)

    flush()
    df = pd.DataFrame(sections)
    if len(df) > 0:
        df["word_count"] = df["section_text"].str.split().str.len()
    else:
        df["word_count"] = []
    return df


def find_section_for_page(sections_df: pd.DataFrame, page_no: int) -> tuple[str, str]:
    """
    Find which section a page belongs to.

    Returns:
        (part, section_title)
    """
    if len(sections_df) == 0:
        return "Unknown", "Unknown"

    m = sections_df[(sections_df["page_start"] <= page_no) & (sections_df["page_end"] >= page_no)]
    if len(m) == 0:
        return "Unknown", "Unknown"
    r = m.iloc[-1]
    return str(r["part"]), str(r["section_title"])


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    """
    Execute hybrid text + table preprocessing pipeline.

    Pipeline stages:
    0. Extract metadata from cover pages
    1. Page extraction with hybrid loader (PyMuPDF + pdfplumber fallback)
    2. Coordinate-based boilerplate stripping (orientation-aware)
    3. Repetition-based header/footer removal
    4. Page classification (text vs. table)
    5. Section inference from headings
    6. Fork processing:
       - Text pages → standard chunking
       - Table pages → dual representation (summary + structured)
    7. Write outputs (parquet + CSV)
    """
    timer = StepTimer()

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    run_date_utc = now_utc_iso()
    enc = get_encoder()
    corpus_id = CORPUS_ID or DOC_ID

    print(f"\n{'=' * 60}")
    print(f"Processing: {DOC_ID}")
    print(f"{'=' * 60}\n")

    doc = fitz.open(PDF_PATH)
    timer.mark("Open PDF (PyMuPDF)")

    with pdfplumber.open(str(PDF_PATH)) as pdf_plumber:
        timer.mark("Open PDF (PDFPlumber)")

        # Extract cover metadata
        pdf_meta = extract_report_metadata_from_pdf(doc, max_pages=2)
        report_year_from_pdf = pdf_meta.get("report_year_from_pdf")
        report_year_from_filename = extract_report_year_from_filename(DOC_ID)

        report_year = report_year_from_pdf or report_year_from_filename
        report_year_source = "pdf_cover" if report_year_from_pdf else "filename"
        period_end_date = pdf_meta.get("period_end_date")

        print(f"Report Year: {report_year} (source: {report_year_source})")
        print(f"Period End: {period_end_date or 'Not detected'}\n")

        timer.mark("Step 0: cover metadata extraction")

        # Extract all pages
        pages_text_lines = {}
        page_heading_candidates = {}
        page_extractor_used = {}
        page_extractor_notes = {}

        qa_removed_top = defaultdict(list)
        qa_removed_bottom = defaultdict(list)

        print("Extracting pages...")
        for i in range(doc.page_count):
            if (i + 1) % 20 == 0:
                print(f"  Page {i + 1}/{doc.page_count}")

            page_no = i + 1

            s, used, note = extract_page_struct_hybrid(doc, pdf_plumber, i)
            page_extractor_used[page_no] = used
            page_extractor_notes[page_no] = note

            # Check if raw lines look like a table (before cleanup)
            raw_lines = [ln["text"] for ln in s.get("lines_all", [])]
            is_raw_table = is_table_like_from_raw_lines(raw_lines)

            kept, rem_a, rem_b = strip_by_coordinates(
                s["lines_all"],
                page_height=s["page_height"],
                page_width=s["page_width"],
                rotation=s["rotation"],
            )

            pages_text_lines[page_no] = kept
            page_heading_candidates[page_no] = select_heading_candidates(
                s["lines_all"], s["p95_font"]
            )

            # Store raw table flag for later use
            pages_text_lines[page_no] = (kept, is_raw_table)

            qa_removed_top[page_no] = rem_a
            qa_removed_bottom[page_no] = rem_b

        timer.mark("Step 1: page extraction + coord strip")

        # Remove repeated headers/footers
        pages_text_only = {pno: lines if isinstance(lines, list) else lines[0]
                           for pno, lines in pages_text_lines.items()}
        pages_text_lines2, common_header, common_footer = remove_repeated_header_footer_lines(
            pages_text_only
        )
        timer.mark("Step 2: repeated header/footer strip")

        # Build pages dataframe with classification
        print("\nClassifying pages...")
        pages_records = []
        text_pages = []
        table_pages = []

        for i in range(doc.page_count):
            page_no = i + 1
            raw = "\n".join(pages_text_lines2.get(page_no, [])).strip()
            clean_text = normalize_page_text(raw)

            # Get raw table flag from earlier detection
            page_data = pages_text_lines.get(page_no)
            is_raw_table = False
            if isinstance(page_data, tuple):
                is_raw_table = page_data[1]

            # Classify page content (combine raw check + post-cleanup check)
            classification = classify_page_content(clean_text)

            # Override if raw structure indicated table
            if is_raw_table and not classification["is_table"]:
                classification["is_table"] = True
                classification["confidence"] = "medium"
                if not classification["table_type"]:
                    classification["table_type"] = detect_table_type(clean_text)

            pages_records.append({
                "doc_id": DOC_ID,
                "corpus_id": corpus_id,
                "report_year": report_year,
                "report_year_source": report_year_source,
                "period_end_date": period_end_date,
                "run_date_utc": run_date_utc,
                "page": page_no,
                "clean_text": clean_text,
                "heading_candidates": page_heading_candidates.get(page_no, []),
                "extractor": page_extractor_used.get(page_no, "unknown"),
                "extractor_notes": page_extractor_notes.get(page_no, ""),
                "is_table": classification["is_table"],
                "table_type": classification["table_type"],
                "classification_confidence": classification["confidence"],
            })

            # Split into text vs. table pages
            if classification["is_table"]:
                table_pages.append({
                    "page": page_no,
                    "text": clean_text,
                    "table_type": classification["table_type"],
                })
            else:
                text_pages.append({
                    "page": page_no,
                    "text": clean_text,
                })

        pages_df = pd.DataFrame(pages_records)

        print(f"  Text pages: {len(text_pages)}")
        print(f"  Table pages: {len(table_pages)}")

        timer.mark("Step 3: pages dataframe + classification")

        # Build sections (from all pages for context)
        sections_df = build_sections_from_pages(pages_df)
        timer.mark("Step 4: section inference")

        # Process TEXT pages → standard chunking
        print("\nChunking text pages...")
        text_chunks = []

        for tpage in text_pages:
            page_no = tpage["page"]
            text = tpage["text"]
            if not text:
                continue

            part, section = find_section_for_page(sections_df, page_no)
            page_chunks = chunk_text_by_tokens(
                text,
                CHUNK_SIZE_TOKENS,
                CHUNK_OVERLAP_TOKENS,
                enc,
            )

            for j, ctext in enumerate(page_chunks):
                wc = len(ctext.split())
                if wc < MIN_CHUNK_WORDS:
                    continue

                chunk_id_local = f"p{page_no:04d}_{j:03d}"
                pages = [page_no]
                page_list_struct = build_page_list_struct(pages)

                text_chunks.append({
                    "doc_id": DOC_ID,
                    "corpus_id": corpus_id,
                    "report_year": report_year,
                    "report_year_source": report_year_source,
                    "period_end_date": period_end_date,
                    "run_date_utc": run_date_utc,
                    "chunk_id": chunk_id_local,
                    "chunk_id_global": make_chunk_id_global(DOC_ID, chunk_id_local),
                    "part": part,
                    "section_title": section,
                    "page_start": page_no,
                    "page_end": page_no,
                    "pages": pages,
                    "page_list": page_list_struct,
                    "chunk_text": ctext,
                    "chunk_tokens": count_tokens(ctext, enc),
                    "word_count": wc,
                    "is_table_like": False,
                    "many_numbers": contains_many_numbers(ctext),
                    "is_table": False,
                    "table_type": None,
                    "table_ref": None,
                })

        text_chunks_df = pd.DataFrame(text_chunks)
        print(f"  Created {len(text_chunks_df)} text chunks")

        timer.mark("Step 5: text chunking")

        # Process TABLE pages → dual representation
        print("\nExtracting tables...")
        table_chunks_df, structured_tables_df = process_table_pages(
            table_pages,
            PDF_PATH,
            pdf_plumber,
            DOC_ID,
            corpus_id,
            report_year,
            period_end_date,
            report_year_source,
            run_date_utc,
            enc,
        )

        print(f"  Extracted {len(structured_tables_df)} tables")
        print(f"  Created {len(table_chunks_df)} table summary chunks")

        timer.mark("Step 6: table extraction + summarization")

        # Merge text and table chunks
        all_chunks_df = pd.concat([text_chunks_df, table_chunks_df], ignore_index=True)
        all_chunks_df = all_chunks_df.sort_values(["page_start", "chunk_id"]).reset_index(drop=True)

        print(f"\nTotal chunks: {len(all_chunks_df)} ({len(text_chunks_df)} text + {len(table_chunks_df)} table)")

        # Validate page-bounded chunks
        if len(all_chunks_df) > 0:
            bad_span = all_chunks_df[all_chunks_df["page_start"] != all_chunks_df["page_end"]]
            if len(bad_span) > 0:
                raise ValueError(
                    f"Found {len(bad_span)} chunks spanning multiple pages. "
                    "Pipeline requires page-bounded chunks for accurate citations."
                )

        timer.mark("Step 7: chunk merging + validation")

        # Write outputs
        out_dir = OUT_ROOT / DOC_ID
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nWriting outputs to: {out_dir}")

        pages_df.to_parquet(out_dir / "pages.parquet", index=False)
        sections_df.to_parquet(out_dir / "sections.parquet", index=False)
        all_chunks_df.to_parquet(out_dir / "chunks.parquet", index=False)

        # Write structured tables
        if len(structured_tables_df) > 0:
            structured_tables_df.to_parquet(out_dir / "tables_structured.parquet", index=False)

        timer.mark("Step 8: parquet writes")

        # Generate metrics
        metrics = {
            "schema_version": "3.0_hybrid",
            "doc_id": DOC_ID,
            "corpus_id": corpus_id,
            "report_year": report_year,
            "period_end_date": period_end_date,
            "counts": {
                "pages_total": len(pages_df),
                "pages_text": len(text_pages),
                "pages_table": len(table_pages),
                "sections": len(sections_df),
                "chunks_total": len(all_chunks_df),
                "chunks_text": len(text_chunks_df),
                "chunks_table": len(table_chunks_df),
                "tables_extracted": len(structured_tables_df),
            },
            "params": {
                "chunk_size_tokens": CHUNK_SIZE_TOKENS,
                "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS,
                "top_strip_frac": TOP_STRIP_FRAC,
                "bottom_strip_frac": BOTTOM_STRIP_FRAC,
                "left_strip_frac": LEFT_STRIP_FRAC,
                "right_strip_frac": RIGHT_STRIP_FRAC,
                "header_footer_repeat_frac": HEADER_FOOTER_REPEAT_FRAC,
                "min_chunk_words": MIN_CHUNK_WORDS,
                "primary_extractor": PRIMARY_EXTRACTOR,
            },
            "table_types_detected": (
                structured_tables_df["table_type"].value_counts().to_dict()
                if len(structured_tables_df) > 0
                else {}
            ),
        }

        safe_json_dump(metrics, out_dir / "metrics.json")

        print(f"\n{'=' * 60}")
        print("PROCESSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"\nOutputs written to: {out_dir}")
        print(f"  - pages.parquet: {len(pages_df)} pages")
        print(f"  - sections.parquet: {len(sections_df)} sections")
        print(f"  - chunks.parquet: {len(all_chunks_df)} chunks (text + table summaries)")
        if len(structured_tables_df) > 0:
            print(f"  - tables_structured.parquet: {len(structured_tables_df)} tables")
        print(f"  - metrics.json: Pipeline statistics")

        timer.mark("Step 9: metrics + completion")

    doc.close()
    timer.mark("Close documents")
    timer.report()


if __name__ == "__main__":
    main()
