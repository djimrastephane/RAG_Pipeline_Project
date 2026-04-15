from __future__ import annotations

from typing import Optional

import re

from rag_pdf.config import DEFAULT_CONFIG

HEADING_MAX_CHARS = DEFAULT_CONFIG.HEADING_MAX_CHARS
HEADING_MIN_CHARS = DEFAULT_CONFIG.HEADING_MIN_CHARS
HEADING_FONT_BOOST_FRAC = DEFAULT_CONFIG.HEADING_FONT_BOOST_FRAC


def is_part_label(line: str) -> Optional[str]:
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


def looks_like_lettered_subsection(line: str) -> bool:
    """
    Detect lettered subsection labels (e.g., 'A Title', 'B Management Report').
    """
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


def is_global_boilerplate_heading(line: str) -> bool:
    """
    Detect global boilerplate headings (organization or report-wide headers).
    """
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


def select_heading_candidates(lines_all: list[dict], page_p95_size: float) -> list[str]:
    """
    Select potential heading lines using font size and text heuristics.

    Only checks first 30 lines of page (headings appear near top).
    """
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
