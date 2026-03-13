from __future__ import annotations

import re
from collections import Counter
from typing import Any

_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]+")
_REPEATED_SHORT_TOKEN_RE = re.compile(r"\b([A-Za-z]{2,4})\b(?:\s+\1){4,}", flags=re.IGNORECASE)


def evaluate_ocr_quality(
    text: str,
    *,
    min_chars: int = 200,
    min_alpha_words: int = 30,
    max_symbol_ratio: float = 0.35,
    repeat_token_max_count: int = 20,
    repeat_token_max_len: int = 4,
    min_non_empty_lines: int = 4,
    reject_min_flags: int = 2,
) -> dict[str, Any]:
    """
    Compute lightweight OCR quality flags and a deterministic rejection decision.

    Flags:
    - low_text_density
    - high_symbol_ratio
    - repeated_garbage
    - low_line_count

    Rejection:
    - reject_ocr = True when number of active flags >= reject_min_flags
    """
    raw = text or ""
    lines = [ln.strip() for ln in raw.splitlines() if ln and ln.strip()]
    alpha_tokens = _ALPHA_TOKEN_RE.findall(raw)
    alpha_tokens_lower = [t.lower() for t in alpha_tokens]
    alpha_word_count = len(alpha_tokens_lower)

    token_counter = Counter(alpha_tokens_lower)
    most_common_token = ""
    most_common_count = 0
    if token_counter:
        most_common_token, most_common_count = token_counter.most_common(1)[0]

    # Treat common financial punctuation as non-noisy symbols.
    allowed_symbols = set("£%.,()/-:'’+&")
    non_ws_chars = [c for c in raw if not c.isspace()]
    noisy_symbol_count = sum(
        1 for c in non_ws_chars if (not c.isalnum()) and (c not in allowed_symbols)
    )
    symbol_ratio = noisy_symbol_count / max(len(non_ws_chars), 1)

    repeated_garbage = bool(_REPEATED_SHORT_TOKEN_RE.search(raw))
    if (
        most_common_count > int(repeat_token_max_count)
        and len(most_common_token) <= int(repeat_token_max_len)
    ):
        repeated_garbage = True

    low_text_density = len(raw) < int(min_chars) or alpha_word_count < int(min_alpha_words)
    high_symbol_ratio = symbol_ratio > float(max_symbol_ratio)
    low_line_count = len(lines) < int(min_non_empty_lines)

    flags = {
        "low_text_density": bool(low_text_density),
        "high_symbol_ratio": bool(high_symbol_ratio),
        "repeated_garbage": bool(repeated_garbage),
        "low_line_count": bool(low_line_count),
    }
    active_flags = [k for k, v in flags.items() if v]
    reject_ocr = len(active_flags) >= int(reject_min_flags)

    return {
        "reject_ocr": bool(reject_ocr),
        "flags": flags,
        "active_flags": active_flags,
        "num_active_flags": int(len(active_flags)),
        "char_count": int(len(raw)),
        "non_empty_lines": int(len(lines)),
        "alpha_word_count": int(alpha_word_count),
        "symbol_ratio": float(symbol_ratio),
        "most_common_token": str(most_common_token),
        "most_common_count": int(most_common_count),
    }
