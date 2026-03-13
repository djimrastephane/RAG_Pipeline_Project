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
1) Boilerplate removal using page coordinates
   - Standard pages: remove top and bottom strips (header and footer)
   - Rotated or wide table pages: remove left and right strips (side headers)
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

Important notes:
- The fallback is page-scoped, not document-scoped.
- Output schema stays identical, and we record the extractor used per page.
- Boilerplate stripping is orientation-aware:
  - Portrait pages: strip top and bottom
  - Rotated or wide pages: strip left and right
  This handles pages with large tables where the repeated header text moves to the sides.

HOW TO RUN
1) Create or activate your Python environment.
2) Install dependencies:
   pip install pymupdf pdfplumber pandas pyarrow
   Optional (recommended for exact token chunking):
   pip install tiktoken
3) Set PDF_PATH at the top of this file to point to your PDF.
4) Run:
   python preprocess_pdf_rag.py

EXPECTED OUTPUTS
OUT_ROOT/<DOC_ID>/
  pages.parquet        Cleaned text per page + metadata + extractor used
  sections.parquet     Section spans inferred from headings + metadata
  chunks.parquet       Page-accurate chunks + flags + token or word counts
  metrics.json         Run summary (counts, params, summaries)
  qa_report.json       Header or footer removal evidence and samples
  sample_chunks.md     Human-readable sample chunks for quick inspection

SCHEMA NOTES (NEW IN THIS VERSION)
This script adds three fields to support multi-document indexing and page-level evaluation:

1) chunk_id_global (string)
   - Guaranteed unique across a multi-document corpus.
   - Format: "<doc_id>:<chunk_id>"
   - Use this as the primary key for embedding, indexing, retrieval logs, and citations.

2) pages (list[int])
   - A plain list of page numbers covered by the chunk.
   - For page-bounded chunks: pages == [page_start] == [page_end].
   - This is the canonical field to use in retrieval evaluation (page-hit checks).

3) page_list (list[dict])
   - A structured list of dicts using the form: [{"element": <page_no>}, ...]
   - Kept for backward compatibility with earlier parquet readers that expect
     a nested structure (often displayed as "page_list.list" in viewers).

Optional: corpus_id (string)
- A run identifier you can use when ingesting multiple documents into one index.
- Default is DOC_ID in single-document mode.
- You can set CORPUS_ID in CONFIG to a stable value for multi-doc experiments.
"""

import re
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict, OrderedDict
from typing import Any, Optional

try:
    import pymupdf as fitz  # PyMuPDF
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

try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None

import time


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
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/Data/Annual Accounts NHS Grampian/Preliminary_Test/Grampian-2022-2023.pdf"
)
DOC_ID = PDF_PATH.stem

OUT_ROOT = Path(
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed"
)

# Optional: use a stable identifier for multi-document experiments.
# If None, defaults to DOC_ID (single-document run id).
CORPUS_ID: Optional[str] = None

# Chunking settings (tokens if tiktoken is available)
CHUNK_SIZE_TOKENS = 224
CHUNK_OVERLAP_TOKENS = 56

# Header/footer removal by page coordinate strips (fractions of page height)
TOP_STRIP_FRAC = 0.08
BOTTOM_STRIP_FRAC = 0.08

# NEW: side-strip removal for rotated / landscape pages (fractions of page width)
LEFT_STRIP_FRAC = 0.08
RIGHT_STRIP_FRAC = 0.08

# Extra repetition-based removal after coordinate stripping
HEADER_FOOTER_REPEAT_FRAC = 0.40
TOP_LINE_K = 5
BOT_LINE_K = 5

# Heading detection tuning
HEADING_MAX_CHARS = 110
HEADING_MIN_CHARS = 4
HEADING_FONT_BOOST_FRAC = 0.85

# Filters
MIN_CHUNK_WORDS = 20

# Hybrid loader settings
PRIMARY_EXTRACTOR = "pymupdf"

# Fallback triggers
FALLBACK_MIN_CHARS = 80
FALLBACK_ON_BAD_TEXT = True
FALLBACK_ON_EXCEPTION = True


# =============================================================================
# TEXT NORMALIZATION CONSTANTS (PDF ARTIFACTS)
# =============================================================================
LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff"]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def now_utc_iso() -> str:
    """
    Returns the current UTC time as an ISO-8601 string.
    Used to tag outputs so shows which run created the artifacts.
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
    Final page-level text normalization after boilerplate removal.
    """
    s = text.replace("\r", "\n")
    s = s.replace("\u00ad", "")
    s = s.replace("￾", "")

    for ch in ZERO_WIDTH:
        s = s.replace(ch, "")

    s = s.replace("\xa0", " ")

    for k, v in LIGATURES.items():
        s = s.replace(k, v)

    s = s.replace("•", "- ").replace("▶", "- ")
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
# NEW: PAGE LIST NORMALISATION AND GLOBAL CHUNK IDS
# =============================================================================
def _to_int_if_whole(x: Any) -> Optional[int]:
    """
    Convert a value into an integer when it represents a whole number.

    Purpose:
        Parquet readers and intermediate transforms can change numeric types.
        Page fields may appear as int, float (2.0), or str ("2"). This helper
        normalises such values so downstream logic can treat page numbers as
        plain integers.
    """
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if math.isfinite(x) and float(x).is_integer():
            return int(x)
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        if s.isdigit():
            return int(s)
        try:
            f = float(s)
            if math.isfinite(f) and f.is_integer():
                return int(f)
        except Exception:
            return None
    return None


def build_pages_from_span(page_start: Any, page_end: Any) -> list[int]:
    """
    Build a canonical pages list from page_start and page_end.
    """
    ps = _to_int_if_whole(page_start)
    pe = _to_int_if_whole(page_end)
    if ps is None or pe is None:
        return []
    if ps <= pe:
        return list(range(ps, pe + 1))
    return list(range(pe, ps + 1))


def build_page_list_struct(pages: list[int]) -> list[dict]:
    """
    Build the backward-compatible structured page list.
    """
    return [{"element": int(p)} for p in pages]


def make_chunk_id_global(doc_id: str, chunk_id: str) -> str:
    """
    Create a globally unique chunk identifier in the form "<doc_id>:<chunk_id>".
    """
    return f"{doc_id}:{chunk_id}"


# =============================================================================
# REPORT METADATA EXTRACTION (FROM COVER AND FILENAME)
# =============================================================================
def extract_report_year_from_filename(name: str) -> Optional[str]:
    yrs = re.findall(r"(?:19|20)\d{2}", name)
    if len(yrs) >= 2:
        return f"{yrs[0]}-{yrs[1]}"
    if len(yrs) == 1:
        return yrs[0]
    return None


def extract_year_range_from_text(text: str) -> Optional[str]:
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


def extract_period_end_date(text: str) -> Optional[str]:
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


def is_part_label(line: str) -> Optional[str]:
    if re.search(r"\bPart\s+A\b", line, flags=re.IGNORECASE):
        return "Part A"
    if re.search(r"\bPart\s+B\b", line, flags=re.IGNORECASE):
        return "Part B"
    return None


def looks_like_numbered_heading(line: str) -> bool:
    return bool(re.match(r"^\s*\d+(\.\d+)*\s+.+", line))


def looks_like_heading_text_only(line: str) -> bool:
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


def looks_like_lettered_subsection(line: str) -> bool:
    line = line.strip()
    if len(line) < HEADING_MIN_CHARS or len(line) > HEADING_MAX_CHARS:
        return False
    if not re.match(r"^[A-Z](?:[.)])?\s+.+", line):
        return False
    if re.search(r"[•\u2022]", line):
        return False
    if re.search(r"\bpage\s+\d+\b", line, flags=re.IGNORECASE):
        return False
    return True


def extract_top_lines(lines_all: list[dict], k: int) -> list[dict]:
    if not lines_all:
        return []
    sorted_lines = sorted(
        lines_all,
        key=lambda l: (float(l.get("y0", 0.0)), float(l.get("x0", 0.0))),
    )
    top: list[dict] = []
    for ln in sorted_lines[:k]:
        txt = str(ln.get("text", "")).strip()
        if not txt:
            continue
        top.append(
            {
                "text": txt,
                "y0": float(ln.get("y0", 0.0)),
                "y1": float(ln.get("y1", 0.0)),
            }
        )
    return top


def is_table_like(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < 4:
        return False
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, len(text))
    many_spaces = sum(l.count("  ") for l in lines) / max(1, len(lines))
    return digit_ratio > 0.12 and many_spaces > 1.5


def contains_many_numbers(text: str) -> bool:
    digits = sum(ch.isdigit() for ch in text)
    return digits / max(1, len(text)) > 0.10



def is_section_anchor_line(line: str) -> bool:
    """
    Decide whether a removed top-strip line is a *semantic section anchor*
    that should be kept for context.

    Intended positives:
      - PERFORMANCE REPORT
      - ACCOUNTABILITY REPORT
      - CORPORATE GOVERNANCE REPORT
      - PERFORMANCE ANALYSIS

    Intended negatives (global boilerplate):
      - NHS GRAMPIAN
      - ANNUAL REPORT AND ACCOUNTS FOR YEAR ENDED 31 MARCH 2023
      - Anything containing dates/years or generic annual-report titles

    This predicate is conservative. It aims to keep true report/section bands
    while leaving global document boilerplate removed.
    """
    if not isinstance(line, str):
        return False

    s = re.sub(r"\s+", " ", line).strip()
    if not s:
        return False

    # Reasonable length for a banner label, avoids long global titles.
    if len(s) < 8 or len(s) > 48:
        return False

    # Must contain one of the anchor keywords.
    anchor_kw = {"REPORT", "ANALYSIS", "STATEMENT", "GOVERNANCE"}
    words = re.findall(r"[A-Za-z]+", s.upper())
    if not words:
        return False
    if not any(w in anchor_kw for w in words):
        return False

    # Exclude likely global boilerplate tokens.
    # Keep this list short and generic, it should not be document-specific.
    hard_exclude = {
        "ANNUAL",
        "ACCOUNTS",
        "YEAR",
        "ENDED",
        "NHS",
        "BOARD",
        "SCOTLAND",
        "GRAMPIAN",
    }
    if any(w in hard_exclude for w in words):
        return False

    # Exclude anything that looks like a date/year.
    if re.search(r"\b(?:19|20)\d{2}\b", s):
        return False
    if re.search(r"\b(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC)\b", s.upper()):
        return False
    if re.search(r"\b\d{1,2}\b", s) and re.search(r"\b(?:MARCH|APRIL|MAY|JUNE|JULY)\b", s.upper()):
        return False

    # Require uppercase style (typical for these banner bars).
    # Allow some punctuation and a small fraction of lowercase.
    alpha = [c for c in s if c.isalpha()]
    if not alpha:
        return False
    upper_ratio = sum(c.isupper() for c in alpha) / len(alpha)
    if upper_ratio < 0.80:
        return False

    # Avoid lines that are mostly punctuation or very “wide” separators.
    if re.fullmatch(r"[-–—_= ]+", s):
        return False

    return True


def is_global_boilerplate_heading(line: str) -> bool:
    if not isinstance(line, str):
        return False
    s = re.sub(r"\s+", " ", line).strip()
    if not s:
        return False
    words = re.findall(r"[A-Za-z]+", s.upper())
    if not words:
        return False
    hard_exclude = {
        "ANNUAL", "ACCOUNTS", "YEAR", "ENDED",
        "NHS", "BOARD", "SCOTLAND", "GRAMPIAN",
    }
    if any(w in hard_exclude for w in words):
        return True
    return False

# =============================================================================
# TOKENISATION AND CHUNKING
# =============================================================================
def get_encoder():
    if tiktoken is None:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def count_tokens(text: str, enc) -> int:
    if enc is None:
        return max(1, int(len(text.split()) / 0.75))
    return len(enc.encode(text))


def chunk_text_by_tokens(text: str, chunk_tokens: int, overlap_tokens: int, enc) -> list[str]:
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
    Extract page text lines from PyMuPDF with layout metadata.

    Returned lines include x0/x1/y0/y1 so the boilerplate remover can strip
    either top/bottom (portrait pages) or left/right (rotated or wide pages).
    """
    d = page.get_text("dict")
    page_height = float(page.rect.height)
    page_width = float(page.rect.width)
    rotation = int(page.rotation or 0)

    lines_all: list[dict] = []
    sizes: list[float] = []

    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            parts = []
            max_size = 0.0
            x0s, x1s, y0s, y1s = [], [], [], []

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
                x0s.append(float(bbox[0]))
                y0s.append(float(bbox[1]))
                x1s.append(float(bbox[2]))
                y1s.append(float(bbox[3]))

            text = normalize_line(" ".join(parts))
            if not text:
                continue

            lines_all.append(
                {
                    "text": text,
                    "x0": min(x0s) if x0s else 0.0,
                    "x1": max(x1s) if x1s else 0.0,
                    "y0": min(y0s) if y0s else 0.0,
                    "y1": max(y1s) if y1s else 0.0,
                    "max_size": max_size,
                }
            )

    lines_all.sort(key=lambda x: (x["y0"], x["x0"]))

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

    return {
        "lines_all": lines_all,
        "page_height": page_height,
        "page_width": page_width,
        "rotation": rotation,
        "p95_font": float(p95),
    }


def extract_page_struct_pdfplumber(pl_page) -> dict:
    """
    Extract page text lines from PDFPlumber.

    Includes x0/x1/y0/y1 so the same orientation-aware boilerplate stripping
    can be applied even when fallback extraction is used.
    """
    page_height = float(pl_page.height)
    page_width = float(pl_page.width)

    words = pl_page.extract_words(
        use_text_flow=True,
        keep_blank_chars=False,
    ) or []

    if not words:
        return {
            "lines_all": [],
            "page_height": page_height,
            "page_width": page_width,
            "rotation": 0,
            "p95_font": 0.0,
        }

    y_tol = 3.0
    words_sorted = sorted(words, key=lambda w: (float(w.get("top", 0.0)), float(w.get("x0", 0.0))))

    grouped: list[list[dict]] = []
    cur: list[dict] = []
    cur_y: Optional[float] = None

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
        x0 = min(float(w.get("x0", 0.0)) for w in ws)
        x1 = max(float(w.get("x1", 0.0)) for w in ws)

        lines_all.append({"text": txt, "x0": x0, "x1": x1, "y0": y0, "y1": y1, "max_size": 0.0})

    lines_all.sort(key=lambda x: (x["y0"], x["x0"]))

    texts = [l["text"] for l in lines_all]
    texts = dehyphenate_lines(texts)
    if len(texts) == len(lines_all):
        for i in range(len(lines_all)):
            lines_all[i]["text"] = texts[i]

    return {
        "lines_all": lines_all,
        "page_height": page_height,
        "page_width": page_width,
        "rotation": 0,
        "p95_font": 0.0,
    }


def extract_page_struct_hybrid(doc: fitz.Document, pdf_plumber, page_index: int) -> tuple[dict, str, str]:
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


def strip_by_coordinates(
    lines_all: list[dict],
    *,
    page_height: float,
    page_width: float,
    rotation: int,
) -> tuple[list[str], list[str], list[str]]:
    """
    Strip boilerplate using page coordinates.

    Portrait pages:
        - Strip top and bottom using y coordinates.

    Rotated or wide pages (common for long tables):
        - Strip left and right using x coordinates.

    Returns:
        kept_lines,
        removed_primary (top or left),
        removed_secondary (bottom or right)
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
            if i < TOP_LINE_K and l in common_header and not is_section_anchor_line(l):
                continue
            if i >= len(norm) - BOT_LINE_K and l in common_footer and not is_section_anchor_line(l):
                continue
            out.append(l)
        cleaned[pno] = out

    return cleaned, common_header, common_footer


def select_heading_candidates(lines_all: list[dict], page_p95_size: float) -> list[str]:
    size_thr = page_p95_size * HEADING_FONT_BOOST_FRAC if page_p95_size else 0.0
    cands: list[str] = []

    i = 0
    while i < min(30, len(lines_all)):
        ln = lines_all[i]
        txt = ln["text"]
        if not txt:
            i += 1
            continue
        if is_part_label(txt):
            i += 1
            continue

        txt_norm = txt.strip()
        if re.match(r"^[A-Z][.)]?$", txt_norm) and i + 1 < len(lines_all):
            nxt = lines_all[i + 1].get("text", "")
            combined = f"{txt_norm[0]} {nxt.strip()}"
            if looks_like_lettered_subsection(combined):
                cands.append(combined)
                i += 2
                continue

        if looks_like_lettered_subsection(txt):
            cands.append(txt)
            i += 1
            continue
        if looks_like_heading_text_only(txt) and float(ln.get("max_size", 0.0)) >= size_thr:
            cands.append(txt)
        i += 1

    return cands


# =============================================================================
# SECTION BUILDING
# =============================================================================
def build_sections_from_pages(pages_df: pd.DataFrame) -> pd.DataFrame:
    sections = []
    current_part = None
    current_section = "Unknown"
    current_subsection = None
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
                "subsection_title": current_subsection or "Unknown",
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

        raw_top = row.get("top_lines", [])
        top_lines: list[str] = []
        if isinstance(raw_top, (list, tuple)):
            for item in raw_top:
                if isinstance(item, dict):
                    txt = str(item.get("text", ""))
                else:
                    txt = str(item)
                norm = normalize_line(txt)
                if norm:
                    top_lines.append(norm)

        for l in top_lines or lines[:25]:
            p = is_part_label(l)
            if p:
                current_part = p
                break

        raw_candidates = row.get("heading_candidates", [])
        if raw_candidates is None or raw_candidates is False:
            heading_candidates = []
        elif isinstance(raw_candidates, str):
            heading_candidates = []
        elif isinstance(raw_candidates, (list, tuple)):
            heading_candidates = list(raw_candidates)
        elif hasattr(raw_candidates, "__iter__"):
            heading_candidates = list(raw_candidates)
        else:
            heading_candidates = []
        section_found = None
        subsection_found = None

        if top_lines:
            for i, line in enumerate(top_lines):
                if is_part_label(line):
                    continue
                if (
                    re.match(r"^[A-Z][.)]?$", line)
                    and i + 1 < len(top_lines)
                    and subsection_found is None
                ):
                    combined = f"{line[0]} {top_lines[i + 1]}"
                    if looks_like_lettered_subsection(combined):
                        subsection_found = combined
                        continue
                if subsection_found is None and looks_like_lettered_subsection(line):
                    subsection_found = line
                    continue
                if section_found is None and (
                    is_section_anchor_line(line)
                    or (looks_like_heading_text_only(line) and not is_global_boilerplate_heading(line))
                ):
                    section_found = line
                if section_found and subsection_found:
                    break
        elif heading_candidates:
            for cand in heading_candidates:
                if subsection_found is None and looks_like_lettered_subsection(cand):
                    subsection_found = cand
                    continue
                if section_found is None and looks_like_heading_text_only(cand):
                    section_found = cand
                if section_found and subsection_found:
                    break
        else:
            for l in lines[:25]:
                if not is_part_label(l):
                    if subsection_found is None and looks_like_lettered_subsection(l):
                        subsection_found = l
                        continue
                    if section_found is None and looks_like_heading_text_only(l):
                        section_found = l
                if section_found and subsection_found:
                    break

        if subsection_found is None and text:
            m = re.search(r"\b([A-Z])[.)]?\s+([A-Z][A-Z ]{3,})\b", text)
            if m:
                candidate = f"{m.group(1)} {normalize_line(m.group(2))}"
                if looks_like_lettered_subsection(candidate):
                    subsection_found = candidate

        if (section_found or subsection_found) and current_texts:
            flush()
            if section_found:
                current_section = section_found
                current_subsection = None
            if subsection_found:
                current_subsection = subsection_found

        current_pages.append(page_no)
        current_texts.append(text)

    flush()
    df = pd.DataFrame(sections)
    if len(df) > 0:
        df["word_count"] = df["section_text"].str.split().str.len()
    else:
        df["word_count"] = []
    return df


def find_section_for_page(sections_df: pd.DataFrame, page_no: int) -> tuple[str, str, str]:
    if len(sections_df) == 0:
        return "Unknown", "Unknown", "Unknown"

    m = sections_df[(sections_df["page_start"] <= page_no) & (sections_df["page_end"] >= page_no)]
    if len(m) == 0:
        return "Unknown", "Unknown", "Unknown"
    r = m.iloc[-1]
    return str(r["part"]), str(r["section_title"]), str(r.get("subsection_title", "Unknown"))


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    """
    Executes the full document preprocessing pipeline for a single digital PDF
    as preparation for retrieval-augmented generation (RAG).

    Includes orientation-aware boilerplate stripping:
    - portrait pages use top/bottom strips
    - rotated or wide pages use left/right strips
    """
    timer = StepTimer()

    if not PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {PDF_PATH}")

    run_date_utc = now_utc_iso()
    enc = get_encoder()

    corpus_id = CORPUS_ID or DOC_ID

    doc = fitz.open(PDF_PATH)
    timer.mark("Open PDF (PyMuPDF)")

    with pdfplumber.open(str(PDF_PATH)) as pdf_plumber:
        timer.mark("Open PDF (PDFPlumber)")

        pdf_meta = extract_report_metadata_from_pdf(doc, max_pages=2)
        report_year_from_pdf = pdf_meta.get("report_year_from_pdf")
        report_year_from_filename = extract_report_year_from_filename(DOC_ID)

        report_year = report_year_from_pdf or report_year_from_filename
        report_year_source = "pdf_cover" if report_year_from_pdf else "filename"
        period_end_date = pdf_meta.get("period_end_date")

        timer.mark("Step 0: cover metadata extraction")

        pages_text_lines = {}
        page_heading_candidates = {}
        page_top_lines = {}
        page_extractor_used = {}
        page_extractor_notes = {}

        qa_removed_top = defaultdict(list)
        qa_removed_bottom = defaultdict(list)

        for i in range(doc.page_count):
            page_no = i + 1

            s, used, note = extract_page_struct_hybrid(doc, pdf_plumber, i)
            page_extractor_used[page_no] = used
            page_extractor_notes[page_no] = note

            kept, rem_a, rem_b = strip_by_coordinates(
                s["lines_all"],
                page_height=s["page_height"],
                page_width=s["page_width"],
                rotation=s["rotation"],
            )

            pages_text_lines[page_no] = kept
            page_heading_candidates[page_no] = select_heading_candidates(s["lines_all"], s["p95_font"])
            page_top_lines[page_no] = extract_top_lines(s["lines_all"], k=max(TOP_LINE_K, 10))

            qa_removed_top[page_no] = rem_a
            qa_removed_bottom[page_no] = rem_b

        timer.mark("Step 1: page extraction + coord strip")

        pages_text_lines2, common_header, common_footer = remove_repeated_header_footer_lines(
            pages_text_lines
        )

        timer.mark("Step 2: repeated header/footer strip")
        for page_no, lines in page_top_lines.items():
            filtered: list[dict] = []
            for ln in lines:
                txt = normalize_line(str(ln.get("text", "")))
                if not txt:
                    continue
                if (txt in common_header or txt in common_footer) and not is_section_anchor_line(txt):
                    continue
                filtered.append(ln)
            page_top_lines[page_no] = filtered

        pages_records = []
        for i in range(doc.page_count):
            page_no = i + 1
            raw = "\n".join(pages_text_lines2.get(page_no, [])).strip()
            clean_text = normalize_page_text(raw)

            pages_records.append(
                {
                    "doc_id": DOC_ID,
                    "corpus_id": corpus_id,
                    "report_year": report_year,
                    "report_year_source": report_year_source,
                    "period_end_date": period_end_date,
                    "run_date_utc": run_date_utc,
                    "page": page_no,
                    "clean_text": clean_text,
                    "heading_candidates": page_heading_candidates.get(page_no, []),
                    "top_lines": page_top_lines.get(page_no, []),
                    "extractor": page_extractor_used.get(page_no, "unknown"),
                    "extractor_notes": page_extractor_notes.get(page_no, ""),
                }
            )

        pages_df = pd.DataFrame(pages_records)
        timer.mark("Step 3: pages dataframe build")

        sections_df = build_sections_from_pages(pages_df)
        timer.mark("Step 4: section inference")

        chunks = []

        for _, prow in pages_df.iterrows():
            page_no = int(prow["page"])
            text = prow["clean_text"]
            if not text:
                continue

            part, section, subsection = find_section_for_page(sections_df, page_no)
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
                page_start = page_no
                page_end = page_no

                pages = build_pages_from_span(page_start, page_end)
                page_list_struct = build_page_list_struct(pages)

                chunks.append(
                    {
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
                        "subsection_title": subsection,
                        "page_start": page_start,
                        "page_end": page_end,
                        "pages": pages,
                        "page_list": page_list_struct,
                        "chunk_text": ctext,
                        "chunk_tokens": count_tokens(ctext, enc),
                        "word_count": wc,
                        "is_table_like": is_table_like(ctext),
                        "many_numbers": contains_many_numbers(ctext),
                    }
                )

        chunks_df = pd.DataFrame(chunks)

        if len(chunks_df) > 0:
            bad_span = chunks_df[chunks_df["page_start"] != chunks_df["page_end"]]
            if len(bad_span) > 0:
                raise ValueError(
                    f"Found {len(bad_span)} chunks spanning multiple pages. "
                    "Baseline requires page-bounded chunks."
                )

        timer.mark("Step 5: chunking")

        out_dir = OUT_ROOT / DOC_ID
        out_dir.mkdir(parents=True, exist_ok=True)

        pages_df.to_parquet(out_dir / "pages.parquet", index=False)
        sections_df.to_parquet(out_dir / "sections.parquet", index=False)
        chunks_df.to_parquet(out_dir / "chunks.parquet", index=False)

        timer.mark("Step 6: parquet writes")

        metrics = {
            "schema_version": "2.4",
            "doc_id": DOC_ID,
            "corpus_id": corpus_id,
            "counts": {
                "pages": len(pages_df),
                "sections": len(sections_df),
                "chunks": len(chunks_df),
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
                "fallback_min_chars": FALLBACK_MIN_CHARS,
                "fallback_on_bad_text": FALLBACK_ON_BAD_TEXT,
                "fallback_on_exception": FALLBACK_ON_EXCEPTION,
            },
        }

        safe_json_dump(metrics, out_dir / "metrics.json")
        timer.mark("Step 7: QA + metrics")

    doc.close()
    timer.mark("Close documents")
    timer.report()


if __name__ == "__main__":
    main()
