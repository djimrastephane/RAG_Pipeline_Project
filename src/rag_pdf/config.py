from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class PreprocessConfig:
    PDF_PATH: Path = Path(".")
    OUT_ROOT: Path = Path(".")
    CORPUS_ID: Optional[str] = None

    CHUNK_SIZE_TOKENS: int = 320
    CHUNK_OVERLAP_TOKENS: int = 90

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

    CAMELOT_LATTICE_ACCURACY_THRESHOLD: int = 70
    TABLE_SUMMARY_MAX_ROWS: int = 5


DEFAULT_CONFIG = PreprocessConfig()
