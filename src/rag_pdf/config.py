from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PreprocessConfig:
    PDF_PATH: Path = Path(".")
    OUT_ROOT: Path = Path(".")
    CORPUS_ID: Optional[str] = None

    CHUNK_SIZE_TOKENS: int = 224
    CHUNK_OVERLAP_TOKENS: int = 56
    SEGMENT_AWARE_CHUNKING: bool = True
    WHOLE_DOC_MARKDOWN_MODE: bool = False
    MARKDOWN_HEADER_CARRY_FORWARD: bool = True
    MARKDOWN_TABLE_INJECTION: bool = True
    TABLE_CHUNKING_STRATEGY: str = "baseline"
    TABLE_PAGE_BACKUP_TEXT_CHUNKS: bool = False
    TABLE_EXTRACT_RETURN_ALL_TABLES: bool = False
    TABLE_EXTRACT_SECONDARY_BOTTOM_PASS: bool = False

    TOP_STRIP_FRAC: float = 0.08
    BOTTOM_STRIP_FRAC: float = 0.08
    LEFT_STRIP_FRAC: float = 0.08
    RIGHT_STRIP_FRAC: float = 0.08

    HEADER_FOOTER_REPEAT_FRAC: float = 0.40
    TOP_LINE_K: int = 5
    BOT_LINE_K: int = 5

    HEADING_MAX_CHARS: int = 110
    HEADING_MIN_CHARS: int = 4
    HEADING_FONT_BOOST_FRAC: float = 0.85

    MIN_CHUNK_WORDS: int = 20

    PRIMARY_EXTRACTOR: str = "pymupdf"
    FALLBACK_MIN_CHARS: int = 80
    FALLBACK_ON_BAD_TEXT: bool = True
    FALLBACK_ON_EXCEPTION: bool = True

    TABLE_DIGIT_RATIO: float = 0.15
    TABLE_SPACE_RATIO: float = 0.3
    TABLE_MIN_LINES: int = 1

    CAMELOT_LATTICE_ACCURACY_THRESHOLD: int = 85
    CAMELOT_LATTICE_WHITESPACE_MAX: int = 40
    CAMELOT_HYBRID_ACCURACY_THRESHOLD: int = 80
    CAMELOT_HYBRID_WHITESPACE_MAX: int = 45
    CAMELOT_LINE_SCALE: int = 40
    CAMELOT_RESOLUTION: int = 400
    CAMELOT_STREAM_ROW_TOL: int = 5
    CAMELOT_STREAM_EDGE_TOL: int = 200
    TABLE_SUMMARY_MAX_ROWS: int = 5

    OCR_MIN_ALPHA_RATIO: float = 0.3
    OCR_MIN_DIGIT_RATIO: float = 0.6
    OCR_QUALITY_MIN_CHARS: int = 200
    OCR_QUALITY_MIN_ALPHA_WORDS: int = 30
    OCR_QUALITY_MAX_SYMBOL_RATIO: float = 0.35
    OCR_QUALITY_REPEAT_TOKEN_MAX_COUNT: int = 20
    OCR_QUALITY_REPEAT_TOKEN_MAX_LEN: int = 4
    OCR_QUALITY_MIN_NON_EMPTY_LINES: int = 4
    OCR_QUALITY_REJECT_MIN_FLAGS: int = 2

# ============================================================================
# ROTATION HANDLING CONFIGURATION
# ============================================================================

# Boilerplate strip fractions for normal (portrait) pages
STRIP_FRACTIONS_NORMAL = {
    'left': 0.08,
    'right': 0.08,
    'top': 0.08,
    'bottom': 0.08,
}

# Boilerplate strip fractions for rotated/landscape pages
# (Minimal stripping to preserve full-page tables)
STRIP_FRACTIONS_ROTATED = {
    'left': 0.02,
    'right': 0.02,
    'top': 0.02,
    'bottom': 0.02,
}

# Aspect ratio threshold for landscape detection
LANDSCAPE_THRESHOLD_RATIO = 1.2

# Text length threshold for triggering alternative extraction
LOW_YIELD_THRESHOLD_CHARS = 100

DEFAULT_CONFIG = PreprocessConfig()
