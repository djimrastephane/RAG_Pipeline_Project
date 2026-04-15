from __future__ import annotations

from .schemas import OCRConfig


def page_needs_ocr(text: str, config: OCRConfig) -> bool:
    stripped = (text or "").strip()
    if len(stripped) < config.min_chars_before_fallback:
        return True
    alpha = sum(char.isalpha() for char in stripped) / max(len(stripped), 1)
    digit = sum(char.isdigit() for char in stripped) / max(len(stripped), 1)
    return alpha < config.min_alpha_ratio and digit < config.min_digit_ratio
