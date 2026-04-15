from __future__ import annotations

from typing import Optional

import re
from datetime import datetime, timezone

try:
    import pymupdf as fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError(
        "Failed to import PyMuPDF.\n"
        "Fix: pip uninstall -y fitz frontend && pip install -U pymupdf\n"
    ) from e

LIGATURES = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
}

ZERO_WIDTH = ["\u200b", "\u200c", "\u200d", "\ufeff"]


def now_utc_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_line(s: str) -> str:
    """Normalize whitespace and trim a string."""
    return re.sub(r"\s+", " ", s).strip()


def normalize_page_text(text: str) -> str:
    """
    Final page-level text normalization after boilerplate removal.

    Handles:
    - Line endings
    - Soft hyphens
    - Zero-width characters
    - Ligatures
    - Non-breaking spaces
    - Bullet points
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


def dehyphenate_lines(lines: list[str]) -> list[str]:
    """
    Merge hyphenated line breaks.

    Example:
        ["develop-", "ment"] → ["development"]
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


# =============================================================================
# REPORT METADATA EXTRACTION
# =============================================================================

def extract_report_year_from_filename(name: str) -> Optional[str]:
    """Extract year range from filename (e.g., 'Report-2022-2023.pdf' → '2022-2023')."""
    yrs = re.findall(r"(?:19|20)\d{2}", name)
    if len(yrs) >= 2:
        return f"{yrs[0]}-{yrs[1]}"
    if len(yrs) == 1:
        return yrs[0]
    return None


def extract_year_range_from_text(text: str) -> Optional[str]:
    """
    Extract year range from cover page text.

    Patterns:
    - 2022-23 → 2022-23
    - 2022-2023 → 2022-2023
    - 2022 to 2023 → 2022-2023
    """
    t = normalize_line(text).replace("–", "-").replace("—", "-")

    # Pattern: 2022-23
    m = re.search(r"\b((?:19|20)\d{2})\s*[-/]\s*(\d{2})\b", t)
    if m:
        y1 = int(m.group(1))
        y2_2 = int(m.group(2))
        y2 = (y1 // 100) * 100 + y2_2
        return f"{y1}-{str(y2)[-2:]}"

    # Pattern: 2022-2023
    m = re.search(r"\b((?:19|20)\d{2})\s*-\s*((?:19|20)\d{2})\b", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # Pattern: 2022 to 2023
    m = re.search(r"\b((?:19|20)\d{2})\s*(?:to|TO)\s*((?:19|20)\d{2})\b", t)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    return None


def extract_period_end_date(text: str) -> Optional[str]:
    """
    Extract period end date from text.

    Pattern: "period ended 31 March 2023" → "2023-03-31"
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
        "january": 1, "february": 2, "march": 3, "april": 4,
        "may": 5, "june": 6, "july": 7, "august": 8,
        "september": 9, "october": 10, "november": 11, "december": 12,
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
    Extract report metadata from cover pages.

    Args:
        doc: PyMuPDF document
        max_pages: Number of pages to scan for metadata

    Returns:
        {
            "report_year_from_pdf": str or None,
            "period_end_date": str or None (ISO format)
        }
    """
    text_parts = []
    for i in range(min(max_pages, doc.page_count)):
        p = doc.load_page(i)
        text_parts.append(p.get_text("text") or "")

    raw = "\n".join(text_parts)
    year_range = extract_year_range_from_text(raw)
    period_end = extract_period_end_date(raw)

    return {"report_year_from_pdf": year_range, "period_end_date": period_end}
