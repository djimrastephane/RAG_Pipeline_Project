from __future__ import annotations

from typing import Any

try:
    import fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError(
        "Failed to import PyMuPDF.\n"
        "Fix: pip uninstall -y fitz frontend && pip install -U pymupdf\n"
    ) from e

import pdfplumber  # noqa: F401

from rag_pdf.config import DEFAULT_CONFIG
from rag_pdf.text_normalize import dehyphenate_lines, normalize_line

PRIMARY_EXTRACTOR = DEFAULT_CONFIG.PRIMARY_EXTRACTOR
FALLBACK_MIN_CHARS = DEFAULT_CONFIG.FALLBACK_MIN_CHARS
FALLBACK_ON_BAD_TEXT = DEFAULT_CONFIG.FALLBACK_ON_BAD_TEXT
FALLBACK_ON_EXCEPTION = DEFAULT_CONFIG.FALLBACK_ON_EXCEPTION


def _join_lines_text(lines_all: list[dict]) -> str:
    """Join structured lines into single page text for quality checks."""
    return "\n".join(
        [normalize_line(l.get("text", "")) for l in lines_all if normalize_line(l.get("text", ""))]
    ).strip()


def _is_bad_page_text(text: str, min_chars: int) -> tuple[bool, str]:
    """
    Identify whether extracted page text is unreliable.

    Triggers:
    - Empty or very short text
    - Contains replacement characters (�)
    - Low alphabetic ratio (likely encoding issues)

    Returns:
        (is_bad, reason)
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
# PDF EXTRACTION AND CLEANUP
# =============================================================================

def extract_page_struct_pymupdf(page: fitz.Page) -> dict:
    """
    Extract page text lines from PyMuPDF with layout metadata.

    Returns lines with coordinates for boilerplate removal.
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

            lines_all.append({
                "text": text,
                "x0": min(x0s) if x0s else 0.0,
                "x1": max(x1s) if x1s else 0.0,
                "y0": min(y0s) if y0s else 0.0,
                "y1": max(y1s) if y1s else 0.0,
                "max_size": max_size,
            })

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


def extract_page_struct_pdfplumber(pl_page: Any) -> dict:
    """Extract page text lines from PDFPlumber with coordinates."""
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


def extract_page_struct_hybrid(
    doc: fitz.Document,
    pdf_plumber: Any,
    page_index: int,
) -> tuple[dict, str, str]:
    """
    Hybrid page extraction with automatic fallback.

    Uses primary extractor (PyMuPDF) by default, falls back to
    secondary (pdfplumber) if quality checks fail.

    Returns:
        (page_struct, extractor_used, quality_note)
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
