from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class TableDetectConfig:
    NUMERIC_TOKEN_PATTERNS: tuple[str, ...] = field(
        default_factory=lambda: (
            r"\(\d[\d,]*\)",
            r"\d{1,3}(?:,\d{3})+(?:\.\d+)?",
            r"\d+(?:\.\d+)?%",
            r"\d+(?:\.\d+)?",
            r"\b-\b",
        )
    )
    SMALL_MIN_LINES: int = 4
    SMALL_CURRENCY_PATTERN: str = r"£\s*'?000'?s?|\(£\s*'?000'?\)"
    SMALL_HEADER_KEYWORDS: tuple[str, ...] = field(
        default_factory=lambda: (
            "limit", "actual", "variance", "reported", "consolidated",
            "assets", "liabilities", "outturn", "surplus", "deficit", "net",
        )
    )
    SMALL_MIN_NUMERIC_TOKENS_PER_BODY_LINE: int = 2
    SMALL_MIN_DOUBLE_SPACES: int = 3
    SMALL_BODY_WINDOW: int = 12
    SMALL_REQUIRED_BODY_LINES: int = 3
    SMALL_REQUIRED_HEADER_BODY_LINES: int = 2
    RAW_MIN_LINES: int = 2
    RAW_KEYWORDS: tuple[str, ...] = field(
        default_factory=lambda: (
            "note", "£", "£000", "£'000",
            "total", "balance", "expenditure", "income", "assets",
            "liabilities", "depreciation", "impairment",
        )
    )
    RAW_KEYWORD_PATTERNS: tuple[str, ...] = field(
        default_factory=lambda: (
            r"\b20\d{2}/\d{2}\b",
            r"£\s?[\d,]+(?:\.\d+)?",
        )
    )
    RAW_RANGE_PATTERN: str = r"\b\d+\s*[-–]\s*\d+\b"
    RAW_MIN_RANGE_MATCHES: int = 2
    RAW_MIN_KEYWORD_HITS: int = 1
    RAW_DIGIT_RATIO: float = 0.15
    RAW_STRONG_KEYWORD_HITS: int = 2
    RAW_SPACING_MIN_LINES: int = 3
    RAW_MULTI_SPACE_RATIO: float = 0.5
    MANY_NUMBERS_DIGIT_RATIO: float = 0.10
    GRAPHICS_MIN_LINE_SPAN: float = 30.0
    GRAPHICS_AXIS_TOLERANCE: float = 2.0
    GRAPHICS_MIN_RECT_HEIGHT: float = 12.0
    GRAPHICS_CURVE_DOMINANCE_RATIO: float = 3.0
    GRAPHICS_MAX_CURVE_RECTS: int = 2
    GRAPHICS_MIN_RECTS: int = 4
    GRAPHICS_MIN_H_LINES_WITH_RECTS: int = 8
    GRAPHICS_MIN_V_LINES_WITH_RECTS: int = 4
    GRAPHICS_MIN_H_LINES: int = 10
    GRAPHICS_MIN_V_LINES: int = 5
    COLUMN_MIN_LINES: int = 12
    COLUMN_BUCKET_WIDTH: float = 12.0
    COLUMN_MIN_BUCKET_COUNT: int = 6
    COLUMN_MIN_DIGIT_LINES: int = 6
    COLUMN_MIN_DENSE_COLS: int = 5
    TYPE_PATTERNS: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "cash_flow": (
                "cash flow",
                "non-cash transaction",
                "note 2a",
                "note 2b",
                "reconciliation of net cash",
            ),
            "balance_sheet": (
                "balance sheet",
                "statement of financial position",
                "net assets",
                "total assets",
            ),
            "income_statement": (
                "statement of comprehensive net expenditure",
                "socne",
                "income and expenditure",
                "operating costs",
            ),
            "staff_costs": (
                "staff costs",
                "employee benefit",
                "remuneration",
                "pension costs",
            ),
            "property": (
                "property, plant and equipment",
                "ppe",
                "intangible assets",
                "additions to assets",
            ),
            "provisions": (
                "provisions",
                "contingent liabilities",
                "clinical negligence",
            ),
            "financial_instruments": (
                "financial instruments",
                "financial assets",
                "financial liabilities",
            ),
        }
    )


@dataclass
class TableExtractConfig:
    CAMELOT_LATTICE_ACCURACY_THRESHOLD: int = 85
    CAMELOT_LATTICE_WHITESPACE_MAX: int = 40
    CAMELOT_HYBRID_ACCURACY_THRESHOLD: int = 80
    CAMELOT_HYBRID_WHITESPACE_MAX: int = 45
    CAMELOT_LINE_SCALE: int = 40
    CAMELOT_RESOLUTION: int = 400
    CAMELOT_STREAM_ROW_TOL: int = 5
    CAMELOT_STREAM_EDGE_TOL: int = 200
    TABLE_SUMMARY_MAX_ROWS: int = 5
    TABLE_MARKDOWN_MAX_ROWS: int = 30
    TABLE_MARKDOWN_MAX_COLS: int = 10
    TABLE_HEADER_INJECTION_MAX_ROWS: int = 80
    TABLE_HEADER_INJECTION_MAX_FACTS: int = 300
    TABLE_SUMMARY_WORD_TARGET: int = 140
    TABLE_ROW_CHUNK_WORD_TARGET: int = 320
    TABLE_ROW_CHUNK_WORD_HARD_MAX: int = 450
    TABLE_ROW_CHUNK_MAX_ROWS: int = 10
    TABLE_LOCAL_FACTS_MAX: int = 24
    TABLE_SUMMARY_KEY_ROWS_MAX: int = 5


@dataclass
class RegionConfig:
    ENABLE_DIAGNOSTICS: bool = False
    MIN_LINES_PER_REGION: int = 2
    Y_GAP_MULTIPLIER: float = 1.8
    MIN_REGION_HEIGHT: float = 18.0
    MERGE_GAP_TOLERANCE: float = 10.0


@dataclass
class PreprocessConfig:
    PDF_PATH: Path = Path(".")
    OUT_ROOT: Path = Path(".")
    CORPUS_ID: Optional[str] = None

    CHUNK_SIZE_TOKENS: int = 224
    CHUNK_OVERLAP_TOKENS: int = 56
    CROSS_PAGE_SENTENCE_OVERLAP: bool = False
    CROSS_PAGE_OVERLAP_MAX_CHARS: int = 320
    SEGMENT_AWARE_CHUNKING: bool = True
    SEGMENT_BOUNDARY_INSERT_PATTERNS: tuple[str, ...] = field(
        default_factory=lambda: (
            # Synthetic inserts are restricted to heading-like numbered labels.
            # Broad decimal matching was fragmenting narrative finance text
            # (e.g. 15.634, 2.976, 1.229) into tiny chunks.
            r"(\b\d+(?:\.\d+){1,4}\s+[A-Z][A-Za-z][^\n]{0,120})",
        )
    )
    SEGMENT_BOUNDARY_MATCH_PATTERNS: tuple[str, ...] = field(
        default_factory=lambda: (
            r"^\d+(?:\.\d+){1,5}\b",
            r"(?i)^[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,3}\s+"
            r"(?:integration\s+joint\s+board\s*\(ijb\)|ijb)\b",
        )
    )
    SEGMENT_BOUNDARY_SEARCH_PATTERNS: tuple[str, ...] = field(
        default_factory=lambda: (
            r"(?i)\b(?:integration\s+joint\s+boards?|ijbs?)\b",
        )
    )
    SEGMENT_UPPERCASE_HEADING_PATTERN: str = r"^[A-Z][A-Z0-9 ,/&()\-]{8,}$"
    SEGMENT_UPPERCASE_HEADING_MAX_WORDS: int = 14
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
    TABLE_DETECT: TableDetectConfig = field(default_factory=TableDetectConfig)
    TABLE_EXTRACT: TableExtractConfig = field(default_factory=TableExtractConfig)
    REGION: RegionConfig = field(default_factory=RegionConfig)

    OCR_MIN_ALPHA_RATIO: float = 0.3
    OCR_MIN_DIGIT_RATIO: float = 0.6
    OCR_QUALITY_MIN_CHARS: int = 200
    OCR_QUALITY_MIN_ALPHA_WORDS: int = 30
    OCR_QUALITY_MAX_SYMBOL_RATIO: float = 0.35
    OCR_QUALITY_REPEAT_TOKEN_MAX_COUNT: int = 20
    OCR_QUALITY_REPEAT_TOKEN_MAX_LEN: int = 4
    OCR_QUALITY_MIN_NON_EMPTY_LINES: int = 4
    OCR_QUALITY_REJECT_MIN_FLAGS: int = 2

    def __getattr__(self, name: str):
        table_detect_aliases = {
            "TABLE_NUMERIC_TOKEN_PATTERNS": "NUMERIC_TOKEN_PATTERNS",
            "SMALL_TABLE_MIN_LINES": "SMALL_MIN_LINES",
            "SMALL_TABLE_CURRENCY_PATTERN": "SMALL_CURRENCY_PATTERN",
            "SMALL_TABLE_HEADER_KEYWORDS": "SMALL_HEADER_KEYWORDS",
            "SMALL_TABLE_MIN_NUMERIC_TOKENS_PER_BODY_LINE": "SMALL_MIN_NUMERIC_TOKENS_PER_BODY_LINE",
            "SMALL_TABLE_MIN_DOUBLE_SPACES": "SMALL_MIN_DOUBLE_SPACES",
            "SMALL_TABLE_BODY_WINDOW": "SMALL_BODY_WINDOW",
            "SMALL_TABLE_REQUIRED_BODY_LINES": "SMALL_REQUIRED_BODY_LINES",
            "SMALL_TABLE_REQUIRED_HEADER_BODY_LINES": "SMALL_REQUIRED_HEADER_BODY_LINES",
            "RAW_TABLE_MIN_LINES": "RAW_MIN_LINES",
            "RAW_TABLE_KEYWORDS": "RAW_KEYWORDS",
            "RAW_TABLE_KEYWORD_PATTERNS": "RAW_KEYWORD_PATTERNS",
            "RAW_TABLE_RANGE_PATTERN": "RAW_RANGE_PATTERN",
            "RAW_TABLE_MIN_RANGE_MATCHES": "RAW_MIN_RANGE_MATCHES",
            "RAW_TABLE_MIN_KEYWORD_HITS": "RAW_MIN_KEYWORD_HITS",
            "RAW_TABLE_DIGIT_RATIO": "RAW_DIGIT_RATIO",
            "RAW_TABLE_STRONG_KEYWORD_HITS": "RAW_STRONG_KEYWORD_HITS",
            "RAW_TABLE_SPACING_MIN_LINES": "RAW_SPACING_MIN_LINES",
            "RAW_TABLE_MULTI_SPACE_RATIO": "RAW_MULTI_SPACE_RATIO",
            "MANY_NUMBERS_DIGIT_RATIO": "MANY_NUMBERS_DIGIT_RATIO",
            "GRAPHICS_TABLE_MIN_LINE_SPAN": "GRAPHICS_MIN_LINE_SPAN",
            "GRAPHICS_TABLE_AXIS_TOLERANCE": "GRAPHICS_AXIS_TOLERANCE",
            "GRAPHICS_TABLE_MIN_RECT_HEIGHT": "GRAPHICS_MIN_RECT_HEIGHT",
            "GRAPHICS_TABLE_CURVE_DOMINANCE_RATIO": "GRAPHICS_CURVE_DOMINANCE_RATIO",
            "GRAPHICS_TABLE_MAX_CURVE_RECTS": "GRAPHICS_MAX_CURVE_RECTS",
            "GRAPHICS_TABLE_MIN_RECTS": "GRAPHICS_MIN_RECTS",
            "GRAPHICS_TABLE_MIN_H_LINES_WITH_RECTS": "GRAPHICS_MIN_H_LINES_WITH_RECTS",
            "GRAPHICS_TABLE_MIN_V_LINES_WITH_RECTS": "GRAPHICS_MIN_V_LINES_WITH_RECTS",
            "GRAPHICS_TABLE_MIN_H_LINES": "GRAPHICS_MIN_H_LINES",
            "GRAPHICS_TABLE_MIN_V_LINES": "GRAPHICS_MIN_V_LINES",
            "COLUMN_ALIGNMENT_MIN_LINES": "COLUMN_MIN_LINES",
            "COLUMN_ALIGNMENT_BUCKET_WIDTH": "COLUMN_BUCKET_WIDTH",
            "COLUMN_ALIGNMENT_MIN_BUCKET_COUNT": "COLUMN_MIN_BUCKET_COUNT",
            "COLUMN_ALIGNMENT_MIN_DIGIT_LINES": "COLUMN_MIN_DIGIT_LINES",
            "COLUMN_ALIGNMENT_MIN_DENSE_COLS": "COLUMN_MIN_DENSE_COLS",
            "TABLE_TYPE_PATTERNS": "TYPE_PATTERNS",
        }
        table_extract_aliases = {
            "CAMELOT_LATTICE_ACCURACY_THRESHOLD": "CAMELOT_LATTICE_ACCURACY_THRESHOLD",
            "CAMELOT_LATTICE_WHITESPACE_MAX": "CAMELOT_LATTICE_WHITESPACE_MAX",
            "CAMELOT_HYBRID_ACCURACY_THRESHOLD": "CAMELOT_HYBRID_ACCURACY_THRESHOLD",
            "CAMELOT_HYBRID_WHITESPACE_MAX": "CAMELOT_HYBRID_WHITESPACE_MAX",
            "CAMELOT_LINE_SCALE": "CAMELOT_LINE_SCALE",
            "CAMELOT_RESOLUTION": "CAMELOT_RESOLUTION",
            "CAMELOT_STREAM_ROW_TOL": "CAMELOT_STREAM_ROW_TOL",
            "CAMELOT_STREAM_EDGE_TOL": "CAMELOT_STREAM_EDGE_TOL",
            "TABLE_SUMMARY_MAX_ROWS": "TABLE_SUMMARY_MAX_ROWS",
            "TABLE_MARKDOWN_MAX_ROWS": "TABLE_MARKDOWN_MAX_ROWS",
            "TABLE_MARKDOWN_MAX_COLS": "TABLE_MARKDOWN_MAX_COLS",
            "TABLE_HEADER_INJECTION_MAX_ROWS": "TABLE_HEADER_INJECTION_MAX_ROWS",
            "TABLE_HEADER_INJECTION_MAX_FACTS": "TABLE_HEADER_INJECTION_MAX_FACTS",
            "TABLE_SUMMARY_WORD_TARGET": "TABLE_SUMMARY_WORD_TARGET",
            "TABLE_ROW_CHUNK_WORD_TARGET": "TABLE_ROW_CHUNK_WORD_TARGET",
            "TABLE_ROW_CHUNK_WORD_HARD_MAX": "TABLE_ROW_CHUNK_WORD_HARD_MAX",
            "TABLE_ROW_CHUNK_MAX_ROWS": "TABLE_ROW_CHUNK_MAX_ROWS",
            "TABLE_LOCAL_FACTS_MAX": "TABLE_LOCAL_FACTS_MAX",
            "TABLE_SUMMARY_KEY_ROWS_MAX": "TABLE_SUMMARY_KEY_ROWS_MAX",
        }
        alias = table_detect_aliases.get(name)
        if alias is not None:
            return getattr(self.TABLE_DETECT, alias)
        alias = table_extract_aliases.get(name)
        if alias is not None:
            return getattr(self.TABLE_EXTRACT, alias)
        raise AttributeError(f"{self.__class__.__name__!s} object has no attribute {name!r}")

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
