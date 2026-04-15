from __future__ import annotations

from collections import Counter

from rag_pdf.config import DEFAULT_CONFIG
from rag_pdf.headings import is_section_anchor_line
from rag_pdf.text_normalize import normalize_line

from .rotation_handler import get_strip_fractions_for_rotation
from .config import STRIP_FRACTIONS_NORMAL

# Legacy constants (kept for backward compatibility with repeat detection)
TOP_STRIP_FRAC = DEFAULT_CONFIG.TOP_STRIP_FRAC
BOTTOM_STRIP_FRAC = DEFAULT_CONFIG.BOTTOM_STRIP_FRAC
LEFT_STRIP_FRAC = DEFAULT_CONFIG.LEFT_STRIP_FRAC
RIGHT_STRIP_FRAC = DEFAULT_CONFIG.RIGHT_STRIP_FRAC

HEADER_FOOTER_REPEAT_FRAC = DEFAULT_CONFIG.HEADER_FOOTER_REPEAT_FRAC
TOP_LINE_K = DEFAULT_CONFIG.TOP_LINE_K
BOT_LINE_K = DEFAULT_CONFIG.BOT_LINE_K


def strip_by_coordinates(
        lines_all: list[dict],
        *,
        page_height: float,
        page_width: float,
        rotation: int = 0,
) -> tuple[list[str], list[str], list[str]]:
    """
    Strip boilerplate using page coordinates (rotation-aware).

    For rotated pages (90°/270°) or landscape pages, uses minimal
    stripping (2% edges) to preserve full-page table content.

    For normal portrait pages, uses standard stripping (8% edges).

    Portrait pages: Strip top and bottom
    Rotated/landscape pages: Strip left and right

    Args:
        lines_all: List of line dictionaries with coordinates
        page_height: Page height in points
        page_width: Page width in points
        rotation: Page rotation in degrees (0, 90, 180, 270)

    Returns:
        (kept_lines, removed_primary, removed_secondary)
    """
    rot = rotation % 360
    is_rotated = rot in (90, 270)
    is_landscape = page_width / max(page_height, 1.0) > 1.2
    use_side_strips = is_rotated or is_landscape

    # Get rotation-aware strip fractions
    fractions = get_strip_fractions_for_rotation(
        rotation=rotation,
        page_width=page_width,
        page_height=page_height,
        default_fractions=STRIP_FRACTIONS_NORMAL
    )

    kept: list[str] = []
    removed_a: list[str] = []
    removed_b: list[str] = []

    if use_side_strips:
        # For rotated/landscape: strip left and right with rotation-aware fractions
        left_x = page_width * fractions['left']
        right_x = page_width * (1.0 - fractions['right'])

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

    # For normal portrait: strip top and bottom with rotation-aware fractions
    top_y = page_height * fractions['top']
    bot_y = page_height * (1.0 - fractions['bottom'])

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
            if i < TOP_LINE_K and l in common_header and not is_section_anchor_line(l):
                continue
            if i >= len(norm) - BOT_LINE_K and l in common_footer and not is_section_anchor_line(l):
                continue
            out.append(l)
        cleaned[pno] = out

    return cleaned, common_header, common_footer
