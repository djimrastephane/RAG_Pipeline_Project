from __future__ import annotations

from typing import Optional

import re

from rag_pdf.config import DEFAULT_CONFIG
from rag_pdf.text_normalize import normalize_line

TABLE_DIGIT_RATIO = DEFAULT_CONFIG.TABLE_DIGIT_RATIO
TABLE_SPACE_RATIO = DEFAULT_CONFIG.TABLE_SPACE_RATIO
TABLE_MIN_LINES = DEFAULT_CONFIG.TABLE_MIN_LINES
TABLE_DETECT_CFG = DEFAULT_CONFIG.TABLE_DETECT


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
    non_space_chars = sum(1 for ch in text if not ch.isspace())
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, non_space_chars)
    many_spaces = sum(l.count("  ") for l in lines) / max(1, len(lines))
    base = digit_ratio > TABLE_DIGIT_RATIO and many_spaces > TABLE_SPACE_RATIO
    if base:
        return True
    return is_small_financial_table(text)


def count_numeric_tokens(line: str) -> int:
    cfg = TABLE_DETECT_CFG
    count = 0
    for pat in cfg.NUMERIC_TOKEN_PATTERNS:
        count += len(re.findall(pat, line))
    return count


def is_small_financial_table(text: str) -> bool:
    cfg = TABLE_DETECT_CFG
    lines = [l for l in text.splitlines() if l.strip()]
    if len(lines) < cfg.SMALL_MIN_LINES:
        return False

    currency_pattern = re.compile(cfg.SMALL_CURRENCY_PATTERN, re.IGNORECASE)
    if not currency_pattern.search(text):
        return False

    body_like = []
    header_like_indices = []
    for i, line in enumerate(lines):
        tokens = count_numeric_tokens(line)
        has_alpha = bool(re.search(r"[A-Za-z]", line))
        if has_alpha and tokens >= cfg.SMALL_MIN_NUMERIC_TOKENS_PER_BODY_LINE:
            body_like.append(i)
        words = {w.strip(".,;:()").lower() for w in line.split()}
        if sum(1 for w in words if w in cfg.SMALL_HEADER_KEYWORDS) >= 2:
            header_like_indices.append(i)
        if line.count("  ") >= cfg.SMALL_MIN_DOUBLE_SPACES and tokens >= cfg.SMALL_MIN_NUMERIC_TOKENS_PER_BODY_LINE:
            return True

    def _window_has_body(start_idx: int, need: int) -> bool:
        end = min(start_idx + cfg.SMALL_BODY_WINDOW, len(lines))
        return sum(1 for i in body_like if start_idx <= i < end) >= need

    for i in range(len(lines)):
        if _window_has_body(i, cfg.SMALL_REQUIRED_BODY_LINES):
            return True

    for idx in header_like_indices:
        if _window_has_body(idx + 1, cfg.SMALL_REQUIRED_HEADER_BODY_LINES):
            return True

    return False


def is_graphics_table_like(drawings: list[dict]) -> bool:
    """
    Detect table-like grids from vector drawings (lines/rectangles).

    The detector is intentionally conservative:
    - ignores filled rectangles (common in charts/bars)
    - prefers explicit horizontal/vertical line evidence
    - downweights pages dominated by curve paths (common in plots)
    """
    if not drawings:
        return False
    cfg = TABLE_DETECT_CFG
    def _coords_from_item(item):
        if len(item) >= 5:
            return item[1], item[2], item[3], item[4]
        if len(item) == 3:
            p0, p1 = item[1], item[2]
            if isinstance(p0, (tuple, list)) and isinstance(p1, (tuple, list)) and len(p0) == 2 and len(p1) == 2:
                return p0[0], p0[1], p1[0], p1[1]
        if len(item) == 2:
            rect = item[1]
            if isinstance(rect, (tuple, list)) and len(rect) == 4:
                return rect[0], rect[1], rect[2], rect[3]
        return None
    h_lines = 0
    v_lines = 0
    rects = 0
    curve_ops = 0
    for d in drawings:
        drawing_type = str(d.get("type") or "").lower()
        has_fill = d.get("fill") is not None and drawing_type == "f"
        for item in d.get("items", []):
            if not item:
                continue
            op = item[0]
            if op == "c":
                curve_ops += 1
            if op == "l":
                coords = _coords_from_item(item)
                if not coords:
                    continue
                x0, y0, x1, y1 = coords
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                if max(dx, dy) < cfg.GRAPHICS_MIN_LINE_SPAN:
                    continue
                if dy <= cfg.GRAPHICS_AXIS_TOLERANCE and dx >= cfg.GRAPHICS_MIN_LINE_SPAN:
                    h_lines += 1
                elif dx <= cfg.GRAPHICS_AXIS_TOLERANCE and dy >= cfg.GRAPHICS_MIN_LINE_SPAN:
                    v_lines += 1
            elif op == "re":
                # Filled rectangles are often chart bars/background blocks.
                if has_fill:
                    continue
                rects += 1
                coords = _coords_from_item(item)
                if not coords:
                    continue
                x0, y0, x1, y1 = coords
                dx = abs(x1 - x0)
                dy = abs(y1 - y0)
                if dx >= cfg.GRAPHICS_MIN_LINE_SPAN and dy >= cfg.GRAPHICS_MIN_RECT_HEIGHT:
                    h_lines += 2
                    v_lines += 2
    # Plot-heavy pages are curve-dominant and should not be marked as table.
    if curve_ops > (h_lines + v_lines) * cfg.GRAPHICS_CURVE_DOMINANCE_RATIO and rects < cfg.GRAPHICS_MAX_CURVE_RECTS:
        return False

    if rects >= cfg.GRAPHICS_MIN_RECTS and h_lines >= cfg.GRAPHICS_MIN_H_LINES_WITH_RECTS and v_lines >= cfg.GRAPHICS_MIN_V_LINES_WITH_RECTS:
        return True
    return h_lines >= cfg.GRAPHICS_MIN_H_LINES and v_lines >= cfg.GRAPHICS_MIN_V_LINES


def is_column_alignment_table_like(lines_all: list[dict]) -> bool:
    """
    Detect table-like columns by clustered x positions of raw lines.
    """
    cfg = TABLE_DETECT_CFG
    if not lines_all or len(lines_all) < cfg.COLUMN_MIN_LINES:
        return False
    buckets = {}
    digit_lines = 0
    for ln in lines_all:
        x0 = ln.get("x0")
        if x0 is None:
            continue
        bucket = int(round(float(x0) / cfg.COLUMN_BUCKET_WIDTH) * cfg.COLUMN_BUCKET_WIDTH)
        buckets[bucket] = buckets.get(bucket, 0) + 1
        text = str(ln.get("text", ""))
        if any(ch.isdigit() for ch in text):
            digit_lines += 1
    dense_cols = [b for b, c in buckets.items() if c >= cfg.COLUMN_MIN_BUCKET_COUNT]
    if digit_lines < cfg.COLUMN_MIN_DIGIT_LINES:
        return False
    return len(dense_cols) >= cfg.COLUMN_MIN_DENSE_COLS


if __name__ == "__main__":
    small_table = """\
Consolidated Net Assets (£000's)
Item 2024/25 2023/24
Total assets 1,559,285 1,512,300
Total liabilities (64,947) (60,100)
Net assets 1,494,338 1,452,200
"""
    narrative = """\
This report includes a single reference to £000 in the notes. The narrative
explains performance but does not present tabular figures or columns.
"""
    big_table = """\
Header  2024  2023  2022
Line A  1,000  2,000  3,000
Line B  4,000  5,000  6,000
Line C  7,000  8,000  9,000
Line D  10,000  11,000  12,000
"""
    print("small_table:", is_table_like(small_table))
    print("narrative:", is_table_like(narrative))
    print("big_table:", is_table_like(big_table))


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
    cfg = TABLE_DETECT_CFG
    if not lines or len(lines) < cfg.RAW_MIN_LINES:
        return False

    text = "\n".join(lines)

    # Check for financial table keywords
    text_lower = text.lower()
    table_keyword_patterns = [
        re.compile(pat) for pat in cfg.RAW_KEYWORD_PATTERNS
    ]
    keyword_hits = sum(1 for kw in cfg.RAW_KEYWORDS if kw in text_lower)
    keyword_hits += sum(1 for pat in table_keyword_patterns if pat.search(text_lower))

    # Small-table detector: line has 2+ numeric bands + table keyword signal.
    range_pattern = re.compile(cfg.RAW_RANGE_PATTERN)
    for line in lines:
        if len(range_pattern.findall(line)) >= cfg.RAW_MIN_RANGE_MATCHES and keyword_hits >= cfg.RAW_MIN_KEYWORD_HITS:
            return True

    # Check digit ratio
    non_space_chars = sum(1 for ch in text if not ch.isspace())
    digit_ratio = sum(ch.isdigit() for ch in text) / max(1, non_space_chars)
    if digit_ratio < cfg.RAW_DIGIT_RATIO:
        return False

    # Strong signal: multiple keywords + high digits
    if keyword_hits >= cfg.RAW_STRONG_KEYWORD_HITS and digit_ratio > cfg.RAW_DIGIT_RATIO:
        return True

    # Check for tabular spacing (multiple aligned columns)
    lines_with_content = [l for l in lines if l.strip()]
    if len(lines_with_content) >= cfg.RAW_SPACING_MIN_LINES:
        # Look for consistent spacing patterns (tabs or multiple spaces)
        multi_space_lines = sum(1 for l in lines_with_content if "  " in l or "\t" in l)
        if multi_space_lines / len(lines_with_content) > cfg.RAW_MULTI_SPACE_RATIO:
            return True

    return False


def contains_many_numbers(text: str) -> bool:
    """Check if text has high numeric content (>10% digits)."""
    cfg = TABLE_DETECT_CFG
    digits = sum(ch.isdigit() for ch in text)
    non_space_chars = sum(1 for ch in text if not ch.isspace())
    return digits / max(1, non_space_chars) > cfg.MANY_NUMBERS_DIGIT_RATIO


# =============================================================================
# TABLE CLASSIFICATION
# =============================================================================

def detect_table_type(text: str) -> Optional[str]:
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
    cfg = TABLE_DETECT_CFG
    text_norm = normalize_line(text.lower())

    # Score each table type by keyword matches
    scores = {}
    for table_type, keywords in cfg.TYPE_PATTERNS.items():
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
