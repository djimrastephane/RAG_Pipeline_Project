from __future__ import annotations

"""
PDF preprocessing for RAG (Retrieval-Augmented Generation)

GOAL
- Convert a digital PDF into:
  1) page-level cleaned text (page numbers preserved as metadata)
  2) approximate sections (for context tagging and analysis)
  3) page-accurate chunks (for retrieval, citations, and evaluation)

WHY THIS SCRIPT USES A "HYBRID LOADER" (PyMuPDF with PDFPlumber fallback)
This pipeline relies on page layout signals for two key tasks:
1) Header/footer removal using vertical coordinates (top and bottom strips of each page)
2) Heading detection using font size signals (page p95 font size, plus per-line max font size)

PyMuPDF (fitz) provides:
- Fast extraction
- Layout structure and coordinates
- Font sizes per span (useful for heading detection)

However, some PDFs contain pages where PyMuPDF text extraction can be weak:
- Very short text returned for a page that visibly contains content
- Text with many replacement characters (�) indicating decoding issues
- Pages dominated by low-alphabetic content after extraction, sometimes caused by odd encodings
- Occasional exceptions on certain pages

PDFPlumber can recover better text on these problematic pages because it can reconstruct text
from lower-level PDF objects differently.

So we use:
- PRIMARY: PyMuPDF for speed, coordinates, and fonts
- FALLBACK: PDFPlumber per-page when PyMuPDF output is suspicious or when PyMuPDF fails

Important note:
- The fallback is page-scoped, not document-scoped.
- We keep the output schema identical, and we record the extractor used per page.
- This supports QA and reproducibility. You can show which pages required fallback.

HOW TO RUN
1) Create/activate your Python environment.
2) Install dependencies:
   pip install pymupdf pdfplumber pandas pyarrow
   Optional (recommended for exact token chunking):
   pip install tiktoken

3) Set PDF_PATH at the top of this file to point to your PDF.

4) Run from Terminal:
   python preprocess_pdf_rag.py

EXPECTED OUTPUTS
OUT_ROOT/<DOC_ID>/
  pages.parquet        Cleaned text per page + metadata + extractor used
  sections.parquet     Section spans inferred from headings + metadata
  chunks.parquet       Page-accurate chunks + flags + token/word counts
  metrics.json         Run summary (counts, params, summaries)
  qa_report.json       Header/footer removal evidence and samples
  sample_chunks.md     Human-readable sample chunks for quick inspection
"""

import re
import json
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict
from typing import Any

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError(
        "Failed to import PyMuPDF.\n\n"
        "Common cause: you installed the wrong 'fitz' package.\n"
        "Fix:\n"
        "  pip uninstall -y fitz frontend\n"
        "  pip install -U pymupdf\n"
        "Then re-run.\n"
    ) from e

import pdfplumber
import pandas as pd

# Optional dependency for real token counts
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None

# Code chunk to check time taken to complete the preprocessing portion
# with aim to optimize it
import time
from collections import OrderedDict


class StepTimer:
    """
    Lightweight step-level timer for profiling pipeline stages.

    Usage:
        timer = StepTimer()
        timer.mark("step name")
        ...
        timer.report()

    Output:
        - step duration
        - cumulative runtime
    """

    def __init__(self):
        self.start = time.perf_counter()
        self.last = self.start
        self.steps = OrderedDict()

    def mark(self, label: str) -> None:
        now = time.perf_counter()
        self.steps[label] = {
            "step_seconds": now - self.last,
            "total_seconds": now - self.start,
        }
        self.last = now

    def report(self) -> None:
        print("\n=== PIPELINE TIMING REPORT ===")
        for k, v in self.steps.items():
            print(
                f"{k:<45} "
                f"step={v['step_seconds']:>7.3f}s  "
                f"total={v['total_seconds']:>7.3f}s"
            )

# =============================================================================
# CONFIG
# =============================================================================
PDF_PATH = Path(
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/Data/C236-DO-U-RY-00001_02_I_As built report.pdf"
)
DOC_ID = PDF_PATH.stem

OUT_ROOT = Path(
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed"
)

# Chunking settings (tokens if tiktoken is available)
CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 150

# Header/footer removal by page coordinate strips (fractions of page height)
TOP_STRIP_FRAC = 0.08
BOTTOM_STRIP_FRAC = 0.08

# Extra repetition-based removal after coordinate stripping
HEADER_FOOTER_REPEAT_FRAC = 0.40
TOP_LINE_K = 5
BOT_LINE_K = 5

# Heading detection tuning
HEADING_MAX_CHARS = 110
HEADING_MIN_CHARS = 4
HEADING_FONT_BOOST_FRAC = 0.85  # heading line max size must be >= (page p95 font * this factor)

# Filters
MIN_CHUNK_WORDS = 20  # keep low to avoid dropping short but relevant pages

# Hybrid loader settings
PRIMARY_EXTRACTOR = "pymupdf"  # "pymupdf" or "pdfplumber"

# Fallback triggers
FALLBACK_MIN_CHARS = 80
FALLBACK_ON_BAD_TEXT = True
FALLBACK_ON_EXCEPTION = True


# =============================================================================
# TEXT NORMALIZATION CONSTANTS (PDF ARTIFACTS)
# =============================================================================
LIGATURES = {
    "\ufb00": "ff",   # ﬀ
    "\ufb01": "fi",   # ﬁ
    "\ufb02": "fl",   # ﬂ
    "\ufb03": "ffi",  # ﬃ
    "\ufb04": "ffl",  # ﬄ
}

ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff"]  # invisibles seen in extracted PDF text


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def now_utc_iso() -> str:
    """
    Returns the current UTC time as an ISO-8601 string.
    Used to tag outputs so we can compare runs over time.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_line(s: str) -> str:
    """
    Normalises whitespace and trims a string.
    Used to make comparisons stable (headers, headings, etc.).
    """
    return re.sub(r"\s+", " ", s).strip()


def normalize_page_text(text: str) -> str:
    """
    Final page-level text normalization after header/footer removal.

    Why needed:
    - PDF extraction can preserve typographic ligatures (ﬁ, ﬂ, ﬀ, …)
    - Zero-width characters can silently split tokens
    - Soft hyphens can appear inside words and break matching
    - Some PDFs contain extraction artifacts (for example: '￾') that split words

    This function is applied:
    - after coordinate stripping and repetition stripping
    - before section building, chunking, and embedding

    Args:
        text (str): Page text assembled from kept lines.

    Returns:
        str: Normalized page text suitable for chunking and embedding.
    """
    s = text.replace("\r", "\n")

    # soft hyphen
    s = s.replace("\u00ad", "")

    # common extraction artifact observed in some PDFs (words like hun￾dreds)
    s = s.replace("￾", "")

    # zero-width characters
    for ch in ZERO_WIDTH:
        s = s.replace(ch, "")

    # non-breaking space
    s = s.replace("\xa0", " ")

    # ligatures
    for k, v in LIGATURES.items():
        s = s.replace(k, v)

    # normalize bullets (optional but helps chunk coherence)
    s = s.replace("•", "- ").replace("▶", "- ")

    # collapse all whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_json_dump(obj: Any, path: Path) -> None:
    """
    Writes a JSON file safely with UTF-8 encoding.
    Ensures the parent directory exists.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def describe_series(s: pd.Series) -> dict:
    """
    Compute descriptive statistics for a pandas Series and return them
    as JSON-serialisable Python types.

    Purpose:
        This function is used to summarise numeric outputs of the preprocessing
        pipeline (e.g. chunk token counts, word counts, section lengths) in a
        compact and auditable form. The resulting statistics are written to
        metrics.json to support reproducibility, sanity checking, and
        cross-document comparison.

    Why this is needed:
        pandas.Series.describe() returns NumPy scalar types (e.g. np.float64),
        which are not reliably serialisable to JSON. This function converts all
        such values into native Python types (float or int), ensuring safe and
        consistent JSON output.

    Args:
        s (pd.Series):
            A pandas Series containing numeric values, such as token counts or
            word counts.

    Returns:
        dict:
            A dictionary containing standard descriptive statistics with
            JSON-friendly values.

    Example:
        >>> import pandas as pd
        >>> s = pd.Series([100, 200, 300, 400])
        >>> describe_series(s)
        {
            'count': 4.0,
            'mean': 250.0,
            'std': 129.09944487358055,
            'min': 100.0,
            '25%': 175.0,
            '50%': 250.0,
            '75%': 325.0,
            'max': 400.0
        }

    Notes:
        - These summaries are intended for logging and inspection, not for
          statistical inference.
        - Storing summaries instead of full distributions keeps metrics files
          lightweight while preserving useful diagnostic information.
    """
    d = s.describe()
    return {k: (float(v) if hasattr(v, "item") else v) for k, v in d.to_dict().items()}

def _join_lines_text(lines_all: list[dict]) -> str:
    """
    Join structured lines into a single page text for quality checks.
    Used only to decide whether to fall back to the other extractor.
    """
    return "\n".join(
        [normalize_line(l.get("text", "")) for l in lines_all if normalize_line(l.get("text", ""))]
    ).strip()


def _is_bad_page_text(text: str, min_chars: int) -> tuple[bool, str]:
    """
    Identify whether extracted page text is likely to be unreliable and should
    trigger a fallback extraction method.

    Purpose:
        PDF text extraction quality can vary across pages due to encoding issues,
        layout complexity, or parser limitations. This function applies simple,
        deterministic heuristics to detect pages where the extracted text is
        likely incomplete or corrupted.

        The result is used exclusively to decide whether to re-extract the page
        using an alternative parser. It is not used for scoring, ranking, or
        evaluation of the RAG system.

    Detection criteria:
        A page is flagged as problematic if any of the following hold:
        - The extracted text is empty after normalisation
        - The text length is below a minimum character threshold
        - Unicode replacement characters (�) are present, indicating decoding errors
        - The proportion of alphabetic characters is unusually low, suggesting
          layout noise or failed reconstruction

    Args:
        text (str):
            Raw or lightly cleaned text extracted from a single PDF page.
        min_chars (int):
            Minimum acceptable character count for a page to be considered valid.

    Returns:
        tuple[bool, str]:
            A tuple of (is_bad, reason), where:
            - is_bad (bool): True if the page should trigger fallback extraction
            - reason (str): A short, human-readable label describing the failure mode

    Example:
        >>> _is_bad_page_text("", 80)
        (True, 'empty')

        >>> _is_bad_page_text("Table 1  2023  45  67", 80)
        (True, 'too_short<80')

        >>> _is_bad_page_text("This is a valid paragraph of extracted text.", 80)
        (False, 'ok')

    Notes:
        - All checks are intentionally simple and deterministic to ensure
          reproducibility.
        - Thresholds are conservative and tuned to avoid false positives on
          short but meaningful pages (e.g. section headers).
        - The returned reason is logged for QA and audit purposes.
    """
    t = normalize_line(text)
    if not t:
        return True, "empty"
    if len(t) < min_chars:
        return True, f"too_short<{min_chars}"
    if "\ufffd" in t:
        return True, "replacement_char"
    alpha = sum(c.isalpha() for c in t)
    if alpha / max(len(t), 1) < 0.15:
        return True, "low_alpha_ratio"
    return False, "ok"

# =============================================================================
# REPORT METADATA EXTRACTION (FROM COVER AND FILENAME)
# =============================================================================
def extract_report_year_from_filename(name: str) -> str | None:
    """
    Extract a reporting year or year-range from a document filename.

    Purpose:
        Public-sector PDF filenames often encode the reporting period
        (e.g. "Annual_Report_2023_24.pdf"). This function attempts to infer
        the report year from the filename as a lightweight metadata signal.

        The extracted value is used as a fallback when the reporting period
        cannot be reliably identified from the document content itself
        (e.g. cover pages).

    Behaviour:
        - If two or more four-digit years are present, the first two are
          interpreted as a year range.
        - If a single four-digit year is present, it is returned as-is.
        - If no valid year is found, None is returned.

    Args:
        name (str):
            The filename or filename stem (without directory path).

    Returns:
        str | None:
            A year string (e.g. "2023") or year-range string (e.g. "2023-2024"),
            or None if no valid year can be inferred.

    Examples:
        >>> extract_report_year_from_filename("nhs_report_2023_24.pdf")
        '2023-2024'

        >>> extract_report_year_from_filename("financial_statement_2022.pdf")
        '2022'

        >>> extract_report_year_from_filename("policy_document_final.pdf")
        None

    Notes:
        - This function does not validate that the extracted year corresponds
          to the actual reporting period within the document.
        - Filename-based extraction is intentionally treated as a secondary
          metadata source and is overridden when document-derived metadata
          is available.
    """
    yrs = re.findall(r"(?:19|20)\d{2}", name)
    if len(yrs) >= 2:
        return f"{yrs[0]}-{yrs[1]}"
    if len(yrs) == 1:
        return yrs[0]
    return None


def extract_year_range_from_text(text: str) -> str | None:
    """
    Extract a reporting year or year-range from document text.

    Purpose:
        Public-sector reports often state the reporting period on the cover
        page or in introductory text using a variety of formats. This function
        attempts to identify and normalise such year references from free text.

        The extracted value is used as the primary source of reporting-period
        metadata when available, taking precedence over filename-based inference.

    Supported patterns:
        The function recognises common year and year-range formats, including:
        - Short ranges: 2023-24, 2023–24, 2023/24
        - Full ranges: 2023-2024
        - Natural language: 2023 to 2024

        Dash variants (e.g. en-dash, em-dash) are normalised prior to matching.

    Normalisation:
        - Short-form ranges are returned in the form "YYYY-YY"
          (e.g. "2023-24").
        - Full explicit ranges are returned as "YYYY-YYYY"
          (e.g. "2023-2024").

    Args:
        text (str):
            Raw text extracted from the document, typically from cover or
            introductory pages.

    Returns:
        str | None:
            A normalised year-range string if a valid pattern is detected,
            otherwise None.

    Examples:
        >>> extract_year_range_from_text("Annual Report and Accounts 2023–24")
        '2023-24'

        >>> extract_year_range_from_text("For the period 2022 to 2023")
        '2022-2023'

        >>> extract_year_range_from_text("Policy response document")
        None

    Notes:
        - This function does not verify that the extracted period corresponds
          to the true accounting or reporting period of the document.
        - If multiple year ranges are present, the first matching pattern
          is returned.
        - If no valid year range is found, downstream logic may fall back to
          filename-based metadata extraction.
    """
    t = normalize_line(text).replace("–", "-").replace("—", "-")

    m = re.search(r"\b((?:19|20)\d{2})\s*[-/]\s*(\d{2})\b", t)
    if m:
        y1 = int(m.group(1))
        y2_2 = int(m.group(2))
        y2 = (y1 // 100) * 100 + y2_2
        return f"{y1}-{str(y2)[-2:]}"

    m = re.search(r"\b((?:19|20)\d{2})\s*-\s*((?:19|20)\d{2})\b", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    m = re.search(r"\b((?:19|20)\d{2})\s*(?:to|TO)\s*((?:19|20)\d{2})\b", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    return None

def extract_period_end_date(text: str) -> str | None:
    """
    Extract the reporting period end date from document text.

    Purpose:
        Many public-sector reports explicitly state the end of the reporting
        period on the cover page or in introductory sections using a fixed
        natural-language pattern (e.g. "For the period ended 31 March 2024").
        This function attempts to identify such statements and extract a
        machine-readable end date.

        The extracted date is used as supplementary document metadata and
        supports temporal filtering, analysis, and auditability.

    Behaviour:
        - Searches for case-insensitive occurrences of the phrase
          "period ended <day> <month> <year>".
        - Converts the extracted date into ISO 8601 format (YYYY-MM-DD).
        - Returns None if no valid pattern is detected or if parsing fails.

    Args:
        text (str):
            Raw text extracted from the document, typically from the cover
            page or early pages.

    Returns:
        str | None:
            An ISO-formatted date string (YYYY-MM-DD) if a valid reporting
            period end date is found, otherwise None.

    Examples:
        >>> extract_period_end_date("For the period ended 31 March 2024")
        '2024-03-31'

        >>> extract_period_end_date("Annual Report and Accounts 2023–24")
        None

        >>> extract_period_end_date("Policy consultation document")
        None

    Notes:
        - This function assumes English month names and does not support
          locale-specific formats.
        - Only explicit "period ended" statements are considered; inferred
          or approximate dates are intentionally ignored.
        - Failure to extract a date does not invalidate the document and
          does not prevent downstream processing.
    """
    t = normalize_line(text)
    m = re.search(
        r"\bperiod\s+ended\s+(\d{1,2})\s+([A-Za-z]+)\s+((?:19|20)\d{2})\b",
        t,
        flags=re.IGNORECASE,
    )
    if not m:
        return None

    day = int(m.group(1))
    month = m.group(2).lower()
    year = int(m.group(3))

    month_map = {
        "january": 1,
        "february": 2,
        "march": 3,
        "april": 4,
        "may": 5,
        "june": 6,
        "july": 7,
        "august": 8,
        "september": 9,
        "october": 10,
        "november": 11,
        "december": 12,
    }
    if month not in month_map:
        return None

    try:
        dt = datetime(year, month_map[month], day)
        return dt.date().isoformat()
    except Exception:
        return None


def extract_report_metadata_from_pdf(doc: fitz.Document, max_pages: int = 2) -> dict:
    """
    Extract high-level reporting metadata from the opening pages of a PDF document.

    Purpose:
        Public-sector reports typically state key metadata, such as the reporting
        year and period end date, on the cover page or within the first few pages.
        This function aggregates text from the initial pages of the document and
        applies lightweight pattern-based extraction to identify this metadata.

        The extracted values serve as authoritative document-level metadata when
        available and take precedence over filename-based inference.

    Behaviour:
        - Reads text from the first `max_pages` pages of the PDF.
        - Attempts to extract:
            • a reporting year or year-range using textual patterns
            • a reporting period end date using explicit natural-language cues
        - Returns None values when metadata cannot be reliably identified.

    Args:
        doc (fitz.Document):
            An open PyMuPDF document object.
        max_pages (int, optional):
            Number of initial pages to scan for metadata.
            Defaults to 2, which typically covers the cover and title pages.

    Returns:
        dict:
            A dictionary with the following keys:
            - "report_year_from_pdf": str | None
              Normalised year or year-range extracted from document text.
            - "period_end_date": str | None
              ISO-formatted reporting period end date (YYYY-MM-DD), if present.

    Examples:
        >>> extract_report_metadata_from_pdf(doc)
        {
            "report_year_from_pdf": "2023-24",
            "period_end_date": "2024-03-31"
        }

        >>> extract_report_metadata_from_pdf(doc)
        {
            "report_year_from_pdf": None,
            "period_end_date": None
        }

    Notes:
        - Only the opening pages are inspected to avoid false positives from
          references to historical periods later in the document.
        - Metadata extraction is intentionally conservative; absence of extracted
          values does not indicate an error and does not halt downstream processing.
        - This function assumes the document is already opened using PyMuPDF and
          does not manage file I/O itself.
    """
    text_parts = []
    for i in range(min(max_pages, doc.page_count)):
        p = doc.load_page(i)
        text_parts.append(p.get_text("text") or "")

    raw = "\n".join(text_parts)
    year_range = extract_year_range_from_text(raw)
    period_end = extract_period_end_date(raw)

    return {"report_year_from_pdf": year_range, "period_end_date": period_end}

# =============================================================================
# TEXT CLEANING AND HEADING SUPPORT
# =============================================================================
def dehyphenate_lines(lines: list[str]) -> list[str]:
    """
    Repair word hyphenation caused by PDF line breaks.

    Purpose:
        Text extracted from PDFs often contains artificial hyphenation where
        words are split across line boundaries (e.g. "develop-" followed by
        "ment"). This function attempts to reconstruct such words to improve
        tokenisation, embedding quality, and retrieval consistency.

    Behaviour:
        - If a line ends with a hyphen ("-") and the following line begins with
          a lowercase letter, the two lines are merged with the hyphen removed.
        - In all other cases, lines are preserved unchanged.

    Args:
        lines (list[str]):
            A sequence of text lines extracted from a PDF page.

    Returns:
        list[str]:
            A new list of lines with eligible hyphenated line breaks repaired.

    Examples:
        >>> dehyphenate_lines(["develop-", "ment of the system"])
        ['development of the system']

        >>> dehyphenate_lines(["Well-being", "is important"])
        ['Well-being', 'is important']

    Notes:
        - The lowercase check is intentional and conservative, reducing the
          risk of incorrectly merging genuine hyphenated terms or headings.
        - This function operates purely at the line level and does not attempt
          dictionary-based or language-model-driven word reconstruction.
        - Applying this step early in preprocessing improves downstream
          embedding similarity and retrieval stability.
    """
    out: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i]
        if cur.endswith("-") and i + 1 < len(lines):
            nxt = lines[i + 1]
            nxt_stripped = nxt.lstrip()
            if nxt_stripped[:1].islower():
                merged = cur[:-1] + nxt_stripped
                out.append(merged)
                i += 2
                continue
        out.append(cur)
        i += 1
    return out


def is_part_label(line: str) -> str | None:
    """
    Identify high-level document part labels within a line of text.

    Purpose:
        Some public-sector reports are explicitly structured into major parts
        (e.g. "Part A", "Part B") that group related sections. This function
        detects such part labels to support coarse-grained document structure
        tagging during section construction.

        The extracted part labels are used for contextual metadata only and
        are not relied upon for citation or retrieval boundaries.

    Behaviour:
        - Performs a case-insensitive search for predefined part labels.
        - Returns a normalised part identifier if a match is found.
        - Returns None if the line does not correspond to a recognised part label.

    Args:
        line (str):
            A single line of text extracted from the document.

    Returns:
        str | None:
            A normalised part label (e.g. "Part A", "Part B") if detected,
            otherwise None.

    Examples:
        >>> is_part_label("Part A: Governance and Accountability")
        'Part A'

        >>> is_part_label("PART B Financial Statements")
        'Part B'

        >>> is_part_label("1 Introduction")
        None

    Notes:
        - Detection is intentionally limited to a small, explicit set of
          part labels to avoid false positives.
        - Additional part patterns can be added if required for other
          document types, but such extensions should be validated carefully.
        - Absence of a part label does not imply missing structure and does
          not affect downstream chunking or retrieval.
    """
    if re.search(r"\bPart\s+A\b", line, flags=re.IGNORECASE):
        return "Part A"
    if re.search(r"\bPart\s+B\b", line, flags=re.IGNORECASE):
        return "Part B"
    return None


def looks_like_numbered_heading(line: str) -> bool:
    """
    Determine whether a line resembles a numbered section heading.

    Purpose:
        Many structured reports use hierarchical numbering to denote section
        and subsection titles (e.g. "1 Introduction", "2.3 Methods"). This
        function provides a lightweight heuristic to identify such headings
        based on their numeric prefix.

        The result is used as a supporting signal during section boundary
        detection and document structure reconstruction.

    Behaviour:
        - Matches lines that begin with one or more integers separated by dots,
          followed by at least one whitespace character and additional text.
        - Returns True if the pattern matches, otherwise False.

    Args:
        line (str):
            A single line of text extracted from the document.

    Returns:
        bool:
            True if the line matches a numbered heading pattern, otherwise False.

    Examples:
        >>> looks_like_numbered_heading("1 Introduction")
        True

        >>> looks_like_numbered_heading("2.3.4 Subsection Title")
        True

        >>> looks_like_numbered_heading("Introduction")
        False

        >>> looks_like_numbered_heading("Table 1 Results")
        False

    Notes:
        - This function does not verify semantic correctness of the heading
          content; it only checks structural formatting.
        - Numbered lists or references that coincidentally match the pattern
          may produce false positives, which are mitigated by combining this
          signal with additional heuristics elsewhere in the pipeline.
    """
    return bool(re.match(r"^\s*\d+(\.\d+)*\s+.+", line))


def looks_like_heading_text_only(line: str) -> bool:
    """
    Identify potential section headings using text-only heuristics.

    Purpose:
        In some pages, font size or layout cues may be unavailable or unreliable
        (e.g. when using a fallback PDF extractor). This function provides a
        conservative, text-based heuristic to identify lines that are likely to
        represent section headings based solely on their textual properties.

        The result is used as a fallback signal during section boundary detection
        and is combined with other heuristics to reduce false positives.

    Criteria:
        A line is considered a heading candidate if it:
        - has a reasonable character length (neither very short nor paragraph-like),
        - does not end with sentence punctuation,
        - does not contain bullet symbols or page numbering artifacts,
        - is not a table-of-contents marker,
        - contains a high proportion of capitalised words, or
        - matches a numbered heading pattern.

    Args:
        line (str):
            A single line of text extracted from the document.

    Returns:
        bool:
            True if the line satisfies the text-based heading criteria,
            otherwise False.

    Examples:
        >>> looks_like_heading_text_only("Introduction")
        True

        >>> looks_like_heading_text_only("2.1 Data Collection")
        True

        >>> looks_like_heading_text_only("This section describes the method.")
        False

        >>> looks_like_heading_text_only("Page 12")
        False

    Notes:
        - This heuristic is intentionally conservative and may miss some valid
          headings rather than incorrectly classifying body text.
        - Capitalisation is used as a weak structural signal and is not assumed
          to reflect semantic importance.
        - Text-only heading detection is treated as a fallback and is weighted
          less strongly than font-based cues when both are available.
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
    Flag text chunks that likely originate from flattened tables.

    Purpose:
        When extracting text from PDFs, tables are often linearised into
        space-aligned text that differs structurally from narrative prose.
        Such chunks can degrade embedding quality and retrieval precision.
        This function provides a lightweight heuristic to identify table-like
        chunks for tagging, analysis, or optional downstream handling.

    Behaviour:
        A chunk is flagged as table-like if it satisfies both conditions:
        - contains a relatively high proportion of numeric characters,
        - exhibits frequent multi-space alignment patterns across lines.

        Very short chunks are ignored to reduce false positives.

    Args:
        text (str):
            A chunk of cleaned text produced during preprocessing.

    Returns:
        bool:
            True if the chunk exhibits table-like characteristics,
            otherwise False.

    Examples:
        >>> is_table_like("Year  Revenue  Cost\\n2022  1200  800\\n2023  1350  900")
        True

        >>> is_table_like("This section describes the financial performance.")
        False

    Notes:
        - This heuristic is intentionally simple and does not attempt to parse
          tables or recover their original structure.
        - The output is used as an annotation signal rather than a hard filter;
          table-like chunks are retained to preserve recall.
        - Thresholds were chosen empirically to balance false positives and
          false negatives across heterogeneous public-sector documents.
    """
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return False
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text))
    many_spaces = sum(l.count("  ") for l in lines) / max(1, len(lines))
    return digit_ratio > 0.12 and many_spaces > 1.5


def contains_many_numbers(text: str) -> bool:
    """
    Flag text chunks that contain a high proportion of numeric characters.

    Purpose:
        Public-sector documents often include sections dominated by financial
        figures, KPIs, or statistical summaries. Such numeric-heavy text can
        behave differently during embedding and retrieval compared to narrative
        prose. This function provides a simple signal to identify these chunks.

    Behaviour:
        - Computes the ratio of digit characters to total characters.
        - Flags the text as numeric-heavy if this ratio exceeds a fixed threshold.

    Args:
        text (str):
            A chunk of cleaned text produced during preprocessing.

    Returns:
        bool:
            True if the digit-to-character ratio exceeds the threshold,
            otherwise False.

    Examples:
        >>> contains_many_numbers("Revenue increased to 1250 in 2023 from 980 in 2022")
        True

        >>> contains_many_numbers("This section outlines the governance structure")
        False

    Notes:
        - This heuristic is intentionally coarse and does not distinguish between
          different types of numeric content.
        - The output is used as an annotation signal for analysis and evaluation,
          not as a hard exclusion criterion.
        - Numeric-heavy chunks are retained to preserve recall and citation
          completeness.
    """
    digits = sum(ch.isdigit() for ch in text)
    return digits / max(1, len(text)) > 0.10


# =============================================================================
# TOKENISATION AND CHUNKING
# =============================================================================
def get_encoder():
    """
    Retrieve a token encoder for accurate text length estimation.

    Purpose:
        Chunking and evaluation in the RAG pipeline benefit from token-level
        length estimates that approximate those used by modern large language
        models. This function attempts to load a `tiktoken` encoder to enable
        precise token counting when available.

        If the dependency is not installed or cannot be loaded, the pipeline
        falls back to a word-based approximation to remain robust and portable.

    Behaviour:
        - Returns a `tiktoken` encoder initialised with the "cl100k_base" encoding
          if available.
        - Returns None if `tiktoken` is not installed or initialisation fails.

    Returns:
        tiktoken.Encoding | None:
            A token encoder instance if available, otherwise None.

    Examples:
        >>> enc = get_encoder()
        >>> enc is None or hasattr(enc, "encode")
        True

    Notes:
        - The "cl100k_base" encoding is commonly used by modern transformer-based
          language models and provides a reasonable approximation for token-based
          chunking.
        - Absence of a tokenizer does not prevent pipeline execution; downstream
          logic explicitly handles the None case using conservative estimates.
        - Token counts are used for chunk sizing and analysis, not for enforcing
          strict model input limits.
    """
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str, enc) -> int:
    """
    Estimate the number of tokens in a text segment.

    Purpose:
        Token counts are used to control chunk sizes and to report statistics
        about the processed corpus. When an exact tokenizer is available,
        this function provides accurate token counts aligned with modern
        language model encodings. When unavailable, it falls back to a
        conservative approximation based on word count.

    Behaviour:
        - If a tokenizer encoder is provided, returns the exact token count.
        - If no encoder is available, estimates token count using a fixed
          word-to-token conversion ratio.

    Args:
        text (str):
            The text segment to be analysed.
        enc:
            A token encoder instance (e.g. from `tiktoken`), or None.

    Returns:
        int:
            The number of tokens (exact or estimated).

    Examples:
        >>> count_tokens("This is a short sentence.", None)
        6

        >>> enc = get_encoder()
        >>> isinstance(count_tokens("This is a short sentence.", enc), int)
        True

    Notes:
        - The fallback ratio (approximately 0.75 words per token) is a
          commonly used heuristic for English text and is intentionally
          conservative.
        - Token counts are used for chunk sizing and descriptive metrics,
          not for enforcing strict model input limits.
        - Minor inaccuracies in the fallback estimate do not affect the
          correctness of the retrieval pipeline.
    """
    if enc is None:
        return max(1, int(len(text.split()) / 0.75))
    return len(enc.encode(text))


def chunk_text_by_tokens(text: str, chunk_tokens: int, overlap_tokens: int, enc) -> list[str]:
    """
    Split text into overlapping chunks for retrieval and citation.

    Purpose:
        Retrieval-Augmented Generation relies on dividing documents into
        semantically coherent chunks that are small enough to embed and retrieve
        accurately, while still preserving sufficient local context. This
        function implements deterministic, overlap-aware chunking to balance
        recall and precision during retrieval.

    Behaviour:
        - If a token encoder is available, chunk boundaries are defined in terms
          of exact token counts.
        - If no encoder is available, chunk sizes are approximated using a
          conservative word-based conversion.
        - Adjacent chunks overlap by a fixed amount to reduce boundary effects
          and prevent loss of context at chunk edges.
        - Empty or whitespace-only input returns an empty list.

    Args:
        text (str):
            The cleaned text to be split into chunks.
        chunk_tokens (int):
            Target chunk size, expressed in tokens.
        overlap_tokens (int):
            Number of tokens to overlap between consecutive chunks.
        enc:
            A token encoder instance (e.g. from `tiktoken`), or None.

    Returns:
        list[str]:
            A list of text chunks, each representing a contiguous span of the
            original text.

    Examples:
        >>> chunk_text_by_tokens("This is a simple example.", 5, 2, None)
        ['This is a simple example.']

        >>> enc = get_encoder()
        >>> isinstance(chunk_text_by_tokens("Longer text here...", 800, 150, enc), list)
        True

    Notes:
        - Overlap is intentionally used to preserve contextual continuity across
          chunk boundaries, which improves retrieval robustness.
        - The word-based fallback uses a conservative words-to-tokens ratio and
          minimum chunk sizes to avoid producing excessively small fragments.
        - Chunking is page-scoped in this pipeline, ensuring page-accurate
          provenance for citation and evaluation.
    """
    text = text.strip()
    if not text:
        return []

    if enc is None:
        words = text.split()
        words_per_chunk = max(50, int(chunk_tokens * 0.75))
        words_overlap = max(10, int(overlap_tokens * 0.75))
        chunks = []
        start = 0
        while start < len(words):
            end = min(len(words), start + words_per_chunk)
            chunk = " ".join(words[start:end]).strip()
            if chunk:
                chunks.append(chunk)
            if end == len(words):
                break
            start = max(0, end - words_overlap)
        return chunks

    toks = enc.encode(text)
    chunks = []
    start = 0
    while start < len(toks):
        end = min(len(toks), start + chunk_tokens)
        chunk = enc.decode(toks[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(toks):
            break
        start = max(0, end - overlap_tokens)
    return chunks


# =============================================================================
# PDF EXTRACTION AND CLEANUP
# =============================================================================
def extract_page_struct_pymupdf(page: fitz.Page) -> dict:
    """
    Extract layout-aware, line-level structure from a PDF page using PyMuPDF.

    Purpose:
        Accurate preprocessing of long-form PDFs requires more than raw text.
        This function extracts line text together with layout and typography
        signals that support two downstream tasks:
        - header and footer removal using vertical page coordinates
        - heading detection using font size cues

        PyMuPDF is used as the primary extractor because it exposes structured
        page content (blocks, lines, spans) including bounding boxes and font
        sizes, which are not consistently available in simpler text extractors.

    Outputs:
        The function returns a dictionary with:
        - "lines_all": a list of line records, each containing:
            • "text": cleaned line text (whitespace-normalised)
            • "y0", "y1": vertical bounding box coordinates (top and bottom)
            • "max_size": maximum font size observed in the line
        - "page_height": page height in points (used to compute relative header/footer strips)
        - "p95_font": 95th percentile of font sizes on the page (used as a robust
          reference for heading detection)

    Args:
        page (fitz.Page):
            A single PyMuPDF page object.

    Returns:
        dict:
            A dictionary containing line-level text and layout metadata.

    Example:
        >>> out = extract_page_struct_pymupdf(page)
        >>> out.keys()
        dict_keys(['lines_all', 'page_height', 'p95_font'])

        >>> out['lines_all'][0].keys()
        dict_keys(['text', 'y0', 'y1', 'max_size'])

    Notes:
        - Line text is assembled from span-level text and then normalised to
          reduce extraction noise.
        - A conservative dehyphenation pass is applied at the line level to
          repair words split by PDF line breaks.
        - The page-level font percentile (p95) is used instead of the maximum
          to reduce sensitivity to rare oversized elements (e.g., decorative titles).
        - This function does not remove headers/footers or detect headings; it
          only provides the structured signals required by later steps.
    """
    d = page.get_text("dict")
    page_height = float(page.rect.height)

    lines_all: list[dict] = []
    sizes: list[float] = []

    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            parts = []
            max_size = 0.0
            y0s = []
            y1s = []

            for span in line.get("spans", []):
                t = span.get("text", "")
                if t:
                    parts.append(t)
                sz = float(span.get("size", 0.0) or 0.0)
                if sz > max_size:
                    max_size = sz
                if sz > 0:
                    sizes.append(sz)

            bbox = line.get("bbox", None)
            if bbox and len(bbox) == 4:
                y0s.append(float(bbox[1]))
                y1s.append(float(bbox[3]))

            text = normalize_line(" ".join(parts))
            if not text:
                continue

            lines_all.append(
                {
                    "text": text,
                    "y0": min(y0s) if y0s else 0.0,
                    "y1": max(y1s) if y1s else 0.0,
                    "max_size": max_size,
                }
            )

    lines_all.sort(key=lambda x: (x["y0"], x["y1"]))

    texts = [l["text"] for l in lines_all]
    texts = dehyphenate_lines(texts)
    if len(texts) == len(lines_all):
        for i in range(len(lines_all)):
            lines_all[i]["text"] = texts[i]

    if sizes:
        s_sorted = sorted(sizes)
        p95 = s_sorted[int(0.95 * (len(s_sorted) - 1))]
    else:
        p95 = 0.0

    return {"lines_all": lines_all, "page_height": page_height, "p95_font": float(p95)}

def extract_page_struct_pdfplumber(pl_page) -> dict:
    """
    Extract line-level structure from a PDF page using PDFPlumber.

    Purpose:
        This function provides a fallback extractor for pages where PyMuPDF
        returns unreliable text or raises an exception. PDFPlumber can recover
        readable text on certain PDFs because it reconstructs text from PDF
        objects using a different extraction strategy.

        The output schema is designed to match the structure produced by
        `extract_page_struct_pymupdf` to keep downstream processing uniform.

    Method:
        - Extract word-level items with bounding box coordinates using
          `extract_words`.
        - Sort words by vertical position and then horizontal position.
        - Group words into approximate lines using a small y-coordinate tolerance.
        - Construct a list of line records with text and vertical bounds.

    Args:
        pl_page:
            A single PDFPlumber page object.

    Returns:
        dict:
            A dictionary containing:
            - "lines_all": list of line records, each with:
                • "text": whitespace-normalised line text
                • "y0", "y1": vertical bounds derived from word boxes
                • "max_size": set to 0.0 (font sizes not reliably available)
            - "page_height": page height in points
            - "p95_font": set to 0.0 (font sizes not available)

    Example:
        >>> out = extract_page_struct_pdfplumber(pl_page)
        >>> out.keys()
        dict_keys(['lines_all', 'page_height', 'p95_font'])

    Notes:
        - PDFPlumber does not provide consistent font size information across
          documents, therefore typographic heading detection cannot be applied
          reliably on fallback pages. Downstream section detection uses a text-only
          heading heuristic when p95_font is 0.0.
        - The y-tolerance used for grouping words into lines is intentionally
          small to preserve layout ordering; it may not perfectly reconstruct
          complex multi-column layouts.
        - This function is used only as a per-page fallback to improve robustness,
          not as the default extraction strategy.
    """
    page_height = float(pl_page.height)

    words = pl_page.extract_words(
        use_text_flow=True,
        keep_blank_chars=False,
    ) or []

    if not words:
        return {"lines_all": [], "page_height": page_height, "p95_font": 0.0}

    y_tol = 3.0
    words_sorted = sorted(words, key=lambda w: (float(w.get("top", 0.0)), float(w.get("x0", 0.0))))

    grouped: list[list[dict]] = []
    cur: list[dict] = []
    cur_y: float | None = None

    for w in words_sorted:
        y = float(w.get("top", 0.0))
        if cur_y is None:
            cur_y = y
            cur = [w]
            continue
        if abs(y - cur_y) <= y_tol:
            cur.append(w)
        else:
            grouped.append(cur)
            cur_y = y
            cur = [w]

    if cur:
        grouped.append(cur)

    lines_all: list[dict] = []

    for ws in grouped:
        ws = sorted(ws, key=lambda w: float(w.get("x0", 0.0)))
        txt = normalize_line(
            " ".join([str(w.get("text", "")).strip() for w in ws if str(w.get("text", "")).strip()])
        )
        if not txt:
            continue

        y0 = min(float(w.get("top", 0.0)) for w in ws)
        y1 = max(float(w.get("bottom", 0.0)) for w in ws)

        lines_all.append({"text": txt, "y0": y0, "y1": y1, "max_size": 0.0})

    lines_all.sort(key=lambda x: (x["y0"], x["y1"]))

    texts = [l["text"] for l in lines_all]
    texts = dehyphenate_lines(texts)
    if len(texts) == len(lines_all):
        for i in range(len(lines_all)):
            lines_all[i]["text"] = texts[i]

    return {"lines_all": lines_all, "page_height": page_height, "p95_font": 0.0}


def extract_page_struct_hybrid(doc: fitz.Document, pdf_plumber, page_index: int) -> tuple[dict, str, str]:
    """
    Extract page structure using a hybrid two-extractor strategy with per-page fallback.

    Purpose:
        PDF text extraction quality can vary by page due to layout complexity,
        encoding differences, or parser limitations. This function improves
        robustness by using a primary extractor by default and selectively
        falling back to an alternative extractor only when the primary output
        is likely unreliable or when the primary extractor raises an exception.

        The hybrid design is page-scoped (not document-scoped) to minimise
        unnecessary extractor switching while preserving coverage on problematic
        pages.

    Behaviour:
        - Execute the configured primary extractor ("pymupdf" or "pdfplumber").
        - Optionally evaluate the extracted text using deterministic quality
          heuristics (e.g. minimum length, replacement characters, low alphabetic ratio).
        - If the page is flagged as low-quality, attempt extraction with the
          fallback extractor and select the better result.
        - If the primary extractor fails with an exception and exception fallback
          is enabled, use the fallback extractor.

    Args:
        doc (fitz.Document):
            Open PyMuPDF document used for PyMuPDF extraction.
        pdf_plumber:
            Open PDFPlumber document used for PDFPlumber extraction.
        page_index (int):
            Zero-based page index to extract.

    Returns:
        tuple[dict, str, str]:
            A tuple of (struct, extractor_used, notes), where:
            - struct (dict): Page structure with keys:
                • "lines_all": list of line records (text + coordinates + font cues when available)
                • "page_height": page height in points
                • "p95_font": 95th percentile font size (0.0 on fallback pages without font sizes)
            - extractor_used (str): "pymupdf" or "pdfplumber"
            - notes (str): A short audit label describing the decision, for example:
                • "ok"
                • "fallback_used:<reason>"
                • "fallback_not_better:<reason>"
                • "primary_failed:<ExceptionType>"
                • "fallback_failed:<ExceptionType>:<reason>"

    Examples:
        >>> struct, used, notes = extract_page_struct_hybrid(doc, pdf_plumber, 0)
        >>> used in {"pymupdf", "pdfplumber"}
        True

    Notes:
        - The returned schema is consistent regardless of which extractor was used,
          enabling downstream processing to remain uniform.
        - Fallback decisions are logged via the returned notes string for QA and
          reproducibility.
        - This function is a robustness mechanism; it is not part of the evaluation
          metrics and does not change retrieval scoring directly.
    """
    primary = PRIMARY_EXTRACTOR.strip().lower()
    if primary not in {"pymupdf", "pdfplumber"}:
        raise ValueError("PRIMARY_EXTRACTOR must be 'pymupdf' or 'pdfplumber'")

    def run_pymupdf() -> dict:
        page = doc.load_page(page_index)
        return extract_page_struct_pymupdf(page)

    def run_pdfplumber() -> dict:
        pl_page = pdf_plumber.pages[page_index]
        return extract_page_struct_pdfplumber(pl_page)

    def run_primary() -> tuple[dict, str]:
        if primary == "pymupdf":
            return run_pymupdf(), "pymupdf"
        return run_pdfplumber(), "pdfplumber"

    def run_fallback() -> tuple[dict, str]:
        if primary == "pymupdf":
            return run_pdfplumber(), "pdfplumber"
        return run_pymupdf(), "pymupdf"

    try:
        s, used = run_primary()

        if FALLBACK_ON_BAD_TEXT:
            text = _join_lines_text(s.get("lines_all", []))
            bad, reason = _is_bad_page_text(text, FALLBACK_MIN_CHARS)
            if bad:
                try:
                    s2, used2 = run_fallback()
                    text2 = _join_lines_text(s2.get("lines_all", []))
                    bad2, reason2 = _is_bad_page_text(text2, FALLBACK_MIN_CHARS)

                    if (not bad2) and bad:
                        return s2, used2, f"fallback_used:{reason}"
                    if len(text2) > len(text):
                        return s2, used2, f"fallback_used:{reason};fallback_quality:{reason2}"
                    return s, used, f"fallback_not_better:{reason};fallback_quality:{reason2}"

                except Exception as e2:
                    return s, used, f"fallback_failed:{type(e2).__name__}:{reason}"

        return s, used, "ok"

    except Exception as e1:
        if not FALLBACK_ON_EXCEPTION:
            raise
        s2, used2 = run_fallback()
        return s2, used2, f"primary_failed:{type(e1).__name__}"


def strip_by_coordinates(lines_all: list[dict], page_height: float) -> tuple[list[str], list[str], list[str]]:
    """
    Remove probable header and footer lines using page-relative coordinates.

    Purpose:
        Headers and footers in PDF documents are often repeated across pages
        and introduce noise into retrieval if not removed. This function
        performs an initial, layout-based filtering step by discarding lines
        that appear within fixed top and bottom regions of the page.

        The approach is document-agnostic and does not rely on hard-coded
        header or footer text.

    Behaviour:
        - Computes vertical cutoff thresholds as fixed fractions of page height.
        - Classifies each line based on the vertical midpoint of its bounding box:
            • lines near the top are treated as header candidates,
            • lines near the bottom are treated as footer candidates,
            • all other lines are retained as body text.
        - Returns both retained and removed lines for QA and inspection.

    Args:
        lines_all (list[dict]):
            Line-level records produced by a page extractor, each containing
            text and vertical bounding box coordinates.
        page_height (float):
            Height of the page in points.

    Returns:
        tuple[list[str], list[str], list[str]]:
            A tuple containing:
            - kept: lines retained as body text
            - removed_top: lines removed from the top strip (header candidates)
            - removed_bottom: lines removed from the bottom strip (footer candidates)

    Examples:
        >>> kept, top, bottom = strip_by_coordinates(lines_all, page_height)
        >>> isinstance(kept, list) and isinstance(top, list) and isinstance(bottom, list)
        True

    Notes:
        - Thresholds are expressed as proportions of page height, making the
          method robust to different page sizes and layouts.
        - This step is intentionally permissive; it may leave some header or
          footer lines in place, which are handled by a second-stage,
          repetition-based removal process.
        - Removed lines are retained separately to support transparency,
          debugging, and qualitative QA reporting.
    """
    top_y = page_height * TOP_STRIP_FRAC
    bot_y = page_height * (1.0 - BOTTOM_STRIP_FRAC)

    kept: list[str] = []
    removed_top: list[str] = []
    removed_bottom: list[str] = []

    for ln in lines_all:
        y_mid = (ln["y0"] + ln["y1"]) / 2.0
        txt = ln["text"]
        if y_mid <= top_y:
            removed_top.append(txt)
            continue
        if y_mid >= bot_y:
            removed_bottom.append(txt)
            continue
        kept.append(txt)

    return kept, removed_top, removed_bottom


def remove_repeated_header_footer_lines(
    pages_text_lines: dict[int, list[str]]
) -> tuple[dict[int, list[str]], set[str], set[str]]:
    """
    Remove repeated header and footer lines using cross-page repetition analysis.

    Purpose:
        Some headers and footers are not fully removed by coordinate-based
        filtering because they fall slightly outside the top or bottom strips
        or vary in vertical position. This function performs a second-stage,
        document-level cleanup by identifying lines that repeat across many
        pages in consistent positions.

        The method is adaptive and does not rely on hard-coded header or footer
        text.

    Behaviour:
        - Collects the top and bottom K lines from each page after initial
          coordinate-based stripping.
        - Normalises whitespace to enable stable comparison.
        - Counts line occurrences across all pages.
        - Flags lines that appear on a large fraction of pages as common
          headers or footers.
        - Removes only those repeated lines when they occur in the expected
          top or bottom positions on each page.

    Args:
        pages_text_lines (dict[int, list[str]]):
            Mapping from page number to a list of text lines retained after
            coordinate-based filtering.

    Returns:
        tuple:
            A tuple containing:
            - cleaned (dict[int, list[str]]):
                Page-level text lines with repeated headers and footers removed.
            - common_header (set[str]):
                Normalised header lines identified as repeating across pages.
            - common_footer (set[str]):
                Normalised footer lines identified as repeating across pages.

    Examples:
        >>> cleaned, headers, footers = remove_repeated_header_footer_lines(pages_text_lines)
        >>> isinstance(headers, set) and isinstance(footers, set)
        True

    Notes:
        - A line is considered a header or footer only if it appears on at least
          a fixed fraction of pages, reducing sensitivity to page-specific noise.
        - Removal is position-aware: repeated lines are removed only when they
          appear within the top or bottom K lines of a page.
        - This step complements coordinate-based stripping and together forms a
          two-stage, document-agnostic header/footer removal strategy.
        - Identified header and footer lines are returned to support QA,
          transparency, and qualitative inspection.
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
    Identify candidate section headings using font size cues.

    Purpose:
        Many structured PDF documents distinguish section headings through
        typographic emphasis, typically larger font sizes relative to body
        text. This function exploits that signal to identify likely headings
        on a page while remaining robust to outliers.

        The output is used as an input to section boundary detection rather
        than as a definitive classification.

    Behaviour:
        - Computes a font size threshold as a fraction of the page-level
          95th percentile font size.
        - Scans only the first portion of the page (early lines) where
          headings are most likely to appear.
        - Selects lines that:
            • satisfy text-only heading heuristics, and
            • have a maximum font size greater than or equal to the threshold.
        - Explicitly excludes high-level part labels handled elsewhere.

    Args:
        lines_all (list[dict]):
            Line-level records for a page, each containing text, font size,
            and bounding box metadata.
        page_p95_size (float):
            The 95th percentile font size observed on the page. A value of
            0.0 indicates that reliable font size information is unavailable
            (e.g. on fallback-extracted pages).

    Returns:
        list[str]:
            A list of line texts identified as candidate headings.

    Examples:
        >>> select_heading_candidates(lines_all, page_p95_size=12.0)
        ['1 Introduction', '2 Methodology']

        >>> select_heading_candidates(lines_all, page_p95_size=0.0)
        []

    Notes:
        - The 95th percentile font size is used instead of the maximum to
          reduce sensitivity to rare, oversized elements such as decorative
          titles or page numbers.
        - On pages where font size information is unavailable, this function
          typically returns no candidates, and downstream logic relies on
          text-only heading heuristics.
        - This function is intentionally conservative; missing a heading is
          preferred over incorrectly classifying body text.
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
    Construct approximate document sections by scanning pages for heading starts.

    Purpose:
        This function infers coarse-grained document structure by grouping
        consecutive pages into sections based on detected headings and part
        labels. The resulting sections provide contextual metadata that can be
        attached to chunks for analysis, inspection, and qualitative evaluation.

        Section boundaries are approximate and are not used as the authoritative
        source for citation or retrieval provenance.

    Behaviour:
        - Iterates through pages in document order.
        - Detects high-level part labels (e.g. "Part A", "Part B") when present.
        - Identifies section starts using:
            • font-based heading candidates when available, or
            • conservative text-only heading heuristics as a fallback.
        - Aggregates page text until a new heading is detected, then finalises
          the current section and starts a new one.
        - Assigns each section a page span and concatenated section text.

    Args:
        pages_df (pd.DataFrame):
            Page-level dataset produced during preprocessing. Expected to
            include cleaned page text, detected heading candidates, and
            document-level metadata.

    Returns:
        pd.DataFrame:
            A DataFrame where each row represents an inferred section, with
            fields including:
            - doc_id, report_year, period_end_date
            - part, section_title
            - page_start, page_end
            - section_text
            - word_count

    Examples:
        >>> sections_df = build_sections_from_pages(pages_df)
        >>> set(["section_title", "page_start", "page_end"]).issubset(sections_df.columns)
        True

    Notes:
        - Section boundaries are heuristic and may not perfectly align with the
          document’s logical structure.
        - Sections are used for context tagging and descriptive analysis only.
          All citations remain page-accurate via chunk-level page_start and
          page_end fields.
        - This design choice avoids propagating section-detection errors into
          retrieval evaluation and citation accuracy metrics.
    """
    sections = []
    current_part = None
    current_section = "Unknown"
    current_pages: list[int] = []
    current_texts: list[str] = []

    def flush():
        if not current_pages:
            return
        sections.append(
            {
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
            }
        )
        current_pages.clear()
        current_texts.clear()

    for _, row in pages_df.iterrows():
        page_no = int(row["page"])
        text = str(row["clean_text"] or "")
        lines = [normalize_line(x) for x in text.splitlines() if normalize_line(x)]

        for l in lines[:25]:
            p = is_part_label(l)
            if p:
                current_part = p
                break

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
    Resolve the inferred section context for a given page.

    Purpose:
        During chunk construction, each chunk is associated with a single page.
        This function maps a page number to its inferred document context
        (part and section title) based on precomputed section spans.

        The result is used for contextual tagging and analysis only; it does not
        affect citation accuracy or retrieval scoring.

    Behaviour:
        - Identifies the section whose page span contains the given page number.
        - If multiple sections overlap (rare but possible), the most recent
          matching section is selected.
        - Returns placeholder values when no section match is found.

    Args:
        sections_df (pd.DataFrame):
            DataFrame of inferred document sections, including page_start and
            page_end boundaries.
        page_no (int):
            One-based page number to resolve.

    Returns:
        tuple[str, str]:
            A tuple of (part, section_title). If no matching section is found,
            both values default to "Unknown".

    Examples:
        >>> find_section_for_page(sections_df, page_no=5)
        ('Part A', '1 Introduction')

        >>> find_section_for_page(sections_df, page_no=1)
        ('Unknown', 'Unknown')

    Notes:
        - Section boundaries are heuristic and may not perfectly align with the
          logical document structure.
        - Missing or ambiguous section assignments do not affect downstream
          retrieval or evaluation, as page-level provenance remains authoritative.
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
    Executes the full document preprocessing pipeline for a single digital PDF
    as preparation for retrieval-augmented generation (RAG).

    The pipeline converts an unstructured PDF into three aligned, queryable
    datasets with page-accurate provenance:

    1. Page-level text:
       - Cleaned, normalised text per page
       - Page numbers preserved
       - Header and footer artefacts removed
       - Extractor and fallback decisions recorded for QA

    2. Section-level spans:
       - Approximate document sections inferred from headings
       - Used for contextual tagging and analysis only
       - Not used as a citation source

    3. Chunk-level units:
       - Overlapping, token-bounded text chunks
       - Each chunk linked to exact source pages
       - Suitable for embedding, retrieval, and evaluation

    Extraction strategy:
    - PyMuPDF is used as the primary extractor to preserve layout coordinates
      and font size information required for header removal and heading detection.
    - PDFPlumber is applied as a per-page fallback when deterministic heuristics
      indicate weak or failed extraction.
    - All fallback decisions are logged for transparency and reproducibility.

    Cleaning and normalisation:
    - Coordinate-based and repetition-based header/footer removal
    - Dehyphenation across line breaks
    - Unicode normalisation (ligatures, zero-width characters, artefacts)
    - Final text normalisation applied once before chunking

    Chunking:
    - Token-aware chunking using tiktoken when available
    - Conservative word-based fallback when tokenisation is unavailable
    - Overlap preserved to maintain semantic continuity across chunks

    Quality assurance and logging:
    - Extractor usage and fallback pages recorded per page
    - Header/footer removal samples stored for inspection
    - Per-run metrics and parameters written to metrics.json
    - Human-readable chunk samples written for rapid validation

    Outputs (written to OUT_ROOT / DOC_ID):
    - pages.parquet        Page-level cleaned text and metadata
    - sections.parquet     Section spans and inferred structure
    - chunks.parquet       Page-accurate text chunks for retrieval
    - qa_report.json       Extraction and cleaning diagnostics
    - metrics.json         Run parameters and summary statistics
    - sample_chunks.md     Human-readable chunk inspection file

    Scope and limitations:
    - Designed for digital PDFs with embedded text
    - Scanned documents and OCR are explicitly out of scope
    - Cross-document retrieval and document-level summarisation
      are not performed at this stage

    This function is intentionally deterministic and side-effect free
    beyond writing versioned artefacts to disk, enabling reproducible
    evaluation across documents and experimental runs.
    """

    timer = StepTimer()   # ← START TIMING HERE

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    run_date_utc = now_utc_iso()
    enc = get_encoder()

    # ------------------------------------------------------------
    # Open documents
    # ------------------------------------------------------------
    doc = fitz.open(PDF_PATH)
    timer.mark("Open PDF (PyMuPDF)")

    with pdfplumber.open(str(PDF_PATH)) as pdf_plumber:
        timer.mark("Open PDF (PDFPlumber)")

        # ------------------------------------------------------------
        # Step 0: metadata extraction
        # ------------------------------------------------------------
        pdf_meta = extract_report_metadata_from_pdf(doc, max_pages=2)
        report_year_from_pdf = pdf_meta.get("report_year_from_pdf")
        report_year_from_filename = extract_report_year_from_filename(DOC_ID)

        report_year = report_year_from_pdf or report_year_from_filename
        report_year_source = "pdf_cover" if report_year_from_pdf else "filename"
        period_end_date = pdf_meta.get("period_end_date")

        timer.mark("Step 0: cover metadata extraction")

        # ------------------------------------------------------------
        # Step 1: page extraction + coordinate stripping
        # ------------------------------------------------------------
        pages_text_lines = {}
        page_heading_candidates = {}
        page_extractor_used = {}
        page_extractor_notes = {}

        qa_removed_top = defaultdict(list)
        qa_removed_bottom = defaultdict(list)

        for i in range(doc.page_count):
            page_no = i + 1

            s, used, note = extract_page_struct_hybrid(doc, pdf_plumber, i)
            page_extractor_used[page_no] = used
            page_extractor_notes[page_no] = note

            kept, top, bottom = strip_by_coordinates(
                s["lines_all"], s["page_height"]
            )

            pages_text_lines[page_no] = kept
            page_heading_candidates[page_no] = select_heading_candidates(
                s["lines_all"], s["p95_font"]
            )

            qa_removed_top[page_no] = top
            qa_removed_bottom[page_no] = bottom

        timer.mark("Step 1: page extraction + coord strip")

        # ------------------------------------------------------------
        # Step 2: repetition-based header/footer removal
        # ------------------------------------------------------------
        pages_text_lines2, common_header, common_footer = (
            remove_repeated_header_footer_lines(pages_text_lines)
        )

        timer.mark("Step 2: repeated header/footer strip")

        # ------------------------------------------------------------
        # Step 3: build pages dataset + final text normalisation
        # ------------------------------------------------------------
        pages_records = []
        for i in range(doc.page_count):
            page_no = i + 1
            raw = "\n".join(pages_text_lines2.get(page_no, [])).strip()
            clean_text = normalize_page_text(raw)

            pages_records.append(
                {
                    "doc_id": DOC_ID,
                    "report_year": report_year,
                    "report_year_source": report_year_source,
                    "period_end_date": period_end_date,
                    "run_date_utc": run_date_utc,
                    "page": page_no,
                    "clean_text": clean_text,
                    "heading_candidates": page_heading_candidates.get(page_no, []),
                    "extractor": page_extractor_used.get(page_no, "unknown"),
                    "extractor_notes": page_extractor_notes.get(page_no, ""),
                }
            )

        pages_df = pd.DataFrame(pages_records)
        timer.mark("Step 3: pages dataframe build")

        # ------------------------------------------------------------
        # Step 4: section construction
        # ------------------------------------------------------------
        sections_df = build_sections_from_pages(pages_df)
        timer.mark("Step 4: section inference")

        # ------------------------------------------------------------
        # Step 5: chunking
        # ------------------------------------------------------------
        chunks = []

        for _, prow in pages_df.iterrows():
            page_no = int(prow["page"])
            text = prow["clean_text"]
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

                chunks.append(
                    {
                        "doc_id": DOC_ID,
                        "report_year": report_year,
                        "report_year_source": report_year_source,
                        "period_end_date": period_end_date,
                        "run_date_utc": run_date_utc,
                        "chunk_id": f"p{page_no:04d}_{j:03d}",
                        "part": part,
                        "section_title": section,
                        "page_start": page_no,
                        "page_end": page_no,
                        "page_list": [page_no],
                        "chunk_text": ctext,
                        "chunk_tokens": count_tokens(ctext, enc),
                        "word_count": wc,
                        "is_table_like": is_table_like(ctext),
                        "many_numbers": contains_many_numbers(ctext),
                    }
                )

        chunks_df = pd.DataFrame(chunks)
        timer.mark("Step 5: chunking")

        # ------------------------------------------------------------
        # Step 6: write outputs
        # ------------------------------------------------------------
        out_dir = OUT_ROOT / DOC_ID
        out_dir.mkdir(parents=True, exist_ok=True)

        pages_df.to_parquet(out_dir / "pages.parquet", index=False)
        sections_df.to_parquet(out_dir / "sections.parquet", index=False)
        chunks_df.to_parquet(out_dir / "chunks.parquet", index=False)

        timer.mark("Step 6: parquet writes")

        # ------------------------------------------------------------
        # Step 7: QA + metrics
        # ------------------------------------------------------------
        metrics = {
            "schema_version": "2.3",
            "doc_id": DOC_ID,
            "counts": {
                "pages": len(pages_df),
                "sections": len(sections_df),
                "chunks": len(chunks_df),
            },
        }

        safe_json_dump(metrics, out_dir / "metrics.json")
        timer.mark("Step 7: QA + metrics")

    doc.close()
    timer.mark("Close documents")

    # ------------------------------------------------------------
    # Final timing report
    # ------------------------------------------------------------
    timer.report()


if __name__ == "__main__":
    main()