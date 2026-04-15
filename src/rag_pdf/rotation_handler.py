"""
Rotation handler for PDF pages.

Handles special cases for pages rotated 90° or 270°, which require
less aggressive boilerplate removal to preserve table content.

Author: Generated for RAG Pipeline Project
Date: 2025-02-06
"""
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


def is_rotated(rotation: int) -> bool:
    """
    Check if page rotation requires special handling.

    Pages rotated 90° or 270° often contain full-page tables or
    financial schedules that need minimal boilerplate stripping.

    Args:
        rotation: Page rotation in degrees (0, 90, 180, 270)

    Returns:
        True if rotation is 90° or 270°

    Examples:
        >>> is_rotated(0)
        False
        >>> is_rotated(90)
        True
        >>> is_rotated(270)
        True
    """
    normalized_rotation = rotation % 360
    return normalized_rotation in (90, 270)


def is_landscape(page_width: float, page_height: float) -> bool:
    """
    Check if page is landscape orientation (wider than tall).

    Args:
        page_width: Page width in points
        page_height: Page height in points

    Returns:
        True if width/height ratio > 1.2

    Examples:
        >>> is_landscape(792, 612)  # Letter landscape
        True
        >>> is_landscape(612, 792)  # Letter portrait
        False
    """
    if page_height == 0:
        return False
    ratio = page_width / page_height
    return ratio > 1.2


def get_strip_fractions_for_rotation(
        rotation: int,
        page_width: float,
        page_height: float,
        default_fractions: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Get appropriate boilerplate strip fractions based on page orientation.

    Rotated pages (90°/270°) get minimal stripping (2%) to preserve
    full-page tables. Normal pages get standard stripping (8%).

    Args:
        rotation: Page rotation in degrees
        page_width: Page width in points
        page_height: Page height in points
        default_fractions: Default strip fractions for normal pages

    Returns:
        Dictionary with 'left', 'right', 'top', 'bottom' fractions

    Examples:
        >>> fractions = get_strip_fractions_for_rotation(90, 792, 612)
        >>> fractions['left']
        0.02
        >>> fractions = get_strip_fractions_for_rotation(0, 612, 792)
        >>> fractions['left']
        0.08
    """
    if default_fractions is None:
        default_fractions = {
            'left': 0.08,
            'right': 0.08,
            'top': 0.08,
            'bottom': 0.08
        }

    # Check if page needs minimal stripping
    needs_minimal_strip = (
            is_rotated(rotation) or
            is_landscape(page_width, page_height)
    )

    if needs_minimal_strip:
        # Minimal stripping for rotated/landscape pages (full-page tables)
        return {
            'left': 0.02,
            'right': 0.02,
            'top': 0.02,
            'bottom': 0.02
        }
    else:
        # Standard stripping for normal portrait pages
        return default_fractions.copy()


def should_use_alternative_extractor(
        rotation: int,
        text_length: int,
        page_width: float,
        page_height: float
) -> Tuple[bool, str]:
    """
    Determine if alternative extraction method should be used.

    For rotated pages that yield very little text with primary extractor,
    suggest using alternative method (e.g., pdfplumber).

    Args:
        rotation: Page rotation in degrees
        text_length: Length of extracted text
        page_width: Page width in points
        page_height: Page height in points

    Returns:
        (should_use_alternative, reason)

    Examples:
        >>> should_use_alternative_extractor(90, 10, 792, 612)
        (True, 'rotated_page_low_yield')
        >>> should_use_alternative_extractor(0, 1000, 612, 792)
        (False, 'normal_extraction_ok')
    """
    # Check if rotated and low text yield
    if is_rotated(rotation) and text_length < 100:
        return True, "rotated_page_low_yield"

    # Check if landscape and low text yield
    if is_landscape(page_width, page_height) and text_length < 100:
        return True, "landscape_page_low_yield"

    return False, "normal_extraction_ok"


def get_rotation_metadata(
        rotation: int,
        page_width: float,
        page_height: float,
        text_length: int
) -> Dict[str, any]:
    """
    Get metadata about page rotation and extraction decisions.

    Useful for debugging and quality assessment.

    Args:
        rotation: Page rotation in degrees
        page_width: Page width in points
        page_height: Page height in points
        text_length: Length of extracted text

    Returns:
        Dictionary with rotation metadata

    Examples:
        >>> metadata = get_rotation_metadata(90, 792, 612, 50)
        >>> metadata['is_rotated']
        True
        >>> metadata['strip_mode']
        'minimal'
    """
    rotated = is_rotated(rotation)
    landscape = is_landscape(page_width, page_height)
    use_alt, reason = should_use_alternative_extractor(
        rotation, text_length, page_width, page_height
    )

    strip_mode = "minimal" if (rotated or landscape) else "standard"

    return {
        'rotation_degrees': rotation,
        'is_rotated': rotated,
        'is_landscape': landscape,
        'strip_mode': strip_mode,
        'needs_alternative_extractor': use_alt,
        'alternative_reason': reason,
        'aspect_ratio': page_width / page_height if page_height > 0 else 0,
    }


def log_rotation_handling(
        page_number: int,
        rotation: int,
        text_length_before: int,
        text_length_after: int,
        extraction_method: str
):
    """
    Log rotation handling for debugging.

    Args:
        page_number: Page number (1-indexed)
        rotation: Page rotation in degrees
        text_length_before: Text length before stripping
        text_length_after: Text length after stripping
        extraction_method: Method used for extraction
    """
    if is_rotated(rotation):
        reduction_pct = (
            (1 - text_length_after / text_length_before) * 100
            if text_length_before > 0 else 0
        )

        logger.debug(
            f"Page {page_number}: Rotated {rotation}°, "
            f"extracted {text_length_after} chars "
            f"(reduced {reduction_pct:.1f}% by minimal strip), "
            f"method={extraction_method}"
        )


# Configuration constants
ROTATION_CONFIG = {
    'minimal_strip_fraction': 0.02,
    'standard_strip_fraction': 0.08,
    'landscape_threshold_ratio': 1.2,
    'low_yield_threshold_chars': 100,
}


def get_rotation_config() -> Dict[str, float]:
    """
    Get rotation handling configuration.

    Returns:
        Dictionary with configuration values
    """
    return ROTATION_CONFIG.copy()