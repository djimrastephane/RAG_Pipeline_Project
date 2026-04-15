from __future__ import annotations

from typing import Any, Optional
import os

try:
    import pymupdf as fitz  # PyMuPDF
except Exception as e:
    raise RuntimeError(
        "Failed to import PyMuPDF.\n"
        "Fix: pip uninstall -y fitz frontend && pip install -U pymupdf\n"
    ) from e

import pdfplumber  # noqa: F401

try:
    import pytesseract
    from pdf2image import convert_from_path
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

from rag_pdf.config import DEFAULT_CONFIG
from rag_pdf.ocr_quality import evaluate_ocr_quality
from rag_pdf.text_normalize import dehyphenate_lines, normalize_line
from rag_pdf.ocr_table_fallback import normalize_for_ocr

# Import rotation handling
from .rotation_handler import (
    is_rotated,
    should_use_alternative_extractor,
    get_rotation_metadata,
    log_rotation_handling
)

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


def extract_page_with_ocr(pdf_path: str, page_index: int, rotation_deg: int = 0) -> str:
    """Extract text from image-based page using OCR."""
    if not OCR_AVAILABLE:
        return ""

    try:
        if os.path.exists("/opt/homebrew/bin/tesseract"):
            pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
        poppler_path = None
        if os.path.exists("/opt/homebrew/bin/pdftoppm"):
            poppler_path = "/opt/homebrew/bin"
        images = convert_from_path(
            pdf_path,
            first_page=page_index + 1,
            last_page=page_index + 1,
            dpi=300,
            poppler_path=poppler_path,
        )

        if not images:
            return ""

        # Normalize using known PDF rotation first.
        img = normalize_for_ocr(images[0], rotation_deg)
        # Then try OSD-based correction for scan irregularities.
        try:
            osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
            rotate_deg = int(osd.get("rotate", 0) or 0)
            if rotate_deg in (90, 180, 270):
                # PIL rotates counter-clockwise; negate to deskew to upright text.
                img = img.rotate(-rotate_deg, expand=True)
        except Exception:
            # OSD may fail on noisy pages; continue with original image.
            pass

        text = pytesseract.image_to_string(img, lang="eng")
        return text

    except Exception as e:
        if os.getenv("OCR_DEBUG") == "1":
            print(f"[OCR] page {page_index + 1} failed: {type(e).__name__}: {e}")
        return ""


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

    # Try to get rotation from pdfplumber (may not always be available)
    rotation = 0
    try:
        # pdfplumber pages may have rotation attribute
        rotation = int(getattr(pl_page, 'rotation', 0) or 0)
    except (AttributeError, ValueError, TypeError):
        rotation = 0

    words = pl_page.extract_words(
        use_text_flow=True,
        keep_blank_chars=False,
    ) or []

    if not words:
        return {
            "lines_all": [],
            "page_height": page_height,
            "page_width": page_width,
            "rotation": rotation,
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
        "rotation": rotation,
        "p95_font": 0.0,
    }


def extract_page_struct_hybrid(
        doc: fitz.Document,
        pdf_plumber: Any,
        page_index: int,
        pdf_path: Optional[str] = None,
) -> tuple[dict, str, str]:
    """
    Hybrid page extraction with automatic fallback and rotation awareness.

    Uses primary extractor (PyMuPDF) by default, falls back to
    secondary (pdfplumber) if quality checks fail.

    For rotated pages (90°/270°) that yield very little text with the
    primary extractor, automatically tries the alternative extractor.

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

    def maybe_use_ocr(s_base: dict, used_base: str, note_base: str) -> tuple[dict, str, str]:
        if not (OCR_AVAILABLE and pdf_path):
            return s_base, used_base, note_base

        text_base = _join_lines_text(s_base.get("lines_all", []))
        if len(text_base) >= 50:
            return s_base, used_base, note_base

        ocr_text = extract_page_with_ocr(
            pdf_path,
            page_index,
            int(s_base.get("rotation", 0) or 0),
        )
        if len(ocr_text) <= 50:
            return s_base, used_base, f"{note_base};ocr_raw_attempted;ocr_raw_too_short"

        quality = evaluate_ocr_quality(
            ocr_text,
            min_chars=int(DEFAULT_CONFIG.OCR_QUALITY_MIN_CHARS),
            min_alpha_words=int(DEFAULT_CONFIG.OCR_QUALITY_MIN_ALPHA_WORDS),
            max_symbol_ratio=float(DEFAULT_CONFIG.OCR_QUALITY_MAX_SYMBOL_RATIO),
            repeat_token_max_count=int(DEFAULT_CONFIG.OCR_QUALITY_REPEAT_TOKEN_MAX_COUNT),
            repeat_token_max_len=int(DEFAULT_CONFIG.OCR_QUALITY_REPEAT_TOKEN_MAX_LEN),
            min_non_empty_lines=int(DEFAULT_CONFIG.OCR_QUALITY_MIN_NON_EMPTY_LINES),
            reject_min_flags=int(DEFAULT_CONFIG.OCR_QUALITY_REJECT_MIN_FLAGS),
        )
        if quality.get("reject_ocr"):
            reason = ",".join(quality.get("active_flags", [])) or "quality_flags"
            return s_base, used_base, f"{note_base};ocr_raw_attempted;ocr_raw_rejected_quality:{reason}"

        page_width = float(s_base.get("page_width", 0.0) or 0.0)
        page_height = float(s_base.get("page_height", 0.0) or 0.0)
        if page_width <= 0 or page_height <= 0:
            try:
                page = doc.load_page(page_index)
                page_width = float(page.rect.width)
                page_height = float(page.rect.height)
            except Exception:
                page_width = float(page_width or 0.0)
                page_height = float(page_height or 0.0)

        x0 = page_width * 0.1 if page_width > 0 else 0.0
        x1 = page_width * 0.9 if page_width > 0 else 0.0
        y0 = page_height * 0.5 if page_height > 0 else 0.0
        y1 = y0

        lines_all: list[dict] = []
        for line in ocr_text.splitlines():
            norm = normalize_line(line)
            if not norm:
                continue
            lines_all.append({
                "text": norm,
                "x0": x0,
                "x1": x1,
                "y0": y0,
                "y1": y1,
                "max_size": 0.0,
            })

        if not lines_all:
            return s_base, used_base, f"{note_base};ocr_raw_attempted;ocr_raw_too_short"

        print(f"[OCR-RAW] page {page_index + 1} used in extraction")
        return {
            "lines_all": lines_all,
            "page_width": page_width,
            "page_height": page_height,
            # Preserve source page rotation so downstream strip/classification
            # keeps rotated-page handling instead of reverting to portrait mode.
            "rotation": int(s_base.get("rotation", 0) or 0),
            "p95_font": 0.0,
        }, "ocr", f"{note_base};ocr_raw_attempted;ocr_raw_used"

    try:
        s, used = run_primary()

        # Get page metadata for rotation checks
        rotation = s.get("rotation", 0)
        page_width = s.get("page_width", 0)
        page_height = s.get("page_height", 0)
        text = _join_lines_text(s.get("lines_all", []))
        text_length = len(text)

        # Check if this is a rotated page with low text yield
        use_alt_for_rotation, rotation_reason = should_use_alternative_extractor(
            rotation=rotation,
            text_length=text_length,
            page_width=page_width,
            page_height=page_height
        )

        # If rotated page has low yield, try alternative extractor first
        if use_alt_for_rotation:
            try:
                s2, used2 = run_fallback()
                text2 = _join_lines_text(s2.get("lines_all", []))

                # Use alternative if it yields significantly more text
                if len(text2) > len(text) * 2:  # At least 2x more text
                    log_rotation_handling(
                        page_number=page_index + 1,
                        rotation=rotation,
                        text_length_before=text_length,
                        text_length_after=len(text2),
                        extraction_method=used2
                    )
                    return maybe_use_ocr(s2, used2, f"rotated_page_alt_better:{rotation_reason}")
                else:
                    # Keep primary result even if short (might be genuinely sparse page)
                    return maybe_use_ocr(s, used, f"rotated_page_low_yield:{rotation_reason}")

            except Exception as e2:
                # Fallback failed, use primary result
                return maybe_use_ocr(s, used, f"rotated_fallback_failed:{type(e2).__name__}:{rotation_reason}")

        # Standard quality-based fallback logic (existing behavior)
        if FALLBACK_ON_BAD_TEXT:
            bad, reason = _is_bad_page_text(text, FALLBACK_MIN_CHARS)
            if bad:
                try:
                    s2, used2 = run_fallback()
                    text2 = _join_lines_text(s2.get("lines_all", []))
                    bad2, reason2 = _is_bad_page_text(text2, FALLBACK_MIN_CHARS)

                    if (not bad2) and bad:
                        return maybe_use_ocr(s2, used2, f"fallback_used:{reason}")
                    if len(text2) > len(text):
                        return maybe_use_ocr(
                            s2, used2, f"fallback_used:{reason};fallback_quality:{reason2}"
                        )
                    return maybe_use_ocr(
                        s, used, f"fallback_not_better:{reason};fallback_quality:{reason2}"
                    )

                except Exception as e2:
                    return maybe_use_ocr(s, used, f"fallback_failed:{type(e2).__name__}:{reason}")

        # Log rotation handling for debugging
        if is_rotated(rotation):
            log_rotation_handling(
                page_number=page_index + 1,
                rotation=rotation,
                text_length_before=text_length,
                text_length_after=text_length,
                extraction_method=used
            )

        return maybe_use_ocr(s, used, "ok")

    except Exception as e1:
        if not FALLBACK_ON_EXCEPTION:
            raise
        s2, used2 = run_fallback()
        return maybe_use_ocr(s2, used2, f"primary_failed:{type(e1).__name__}")
