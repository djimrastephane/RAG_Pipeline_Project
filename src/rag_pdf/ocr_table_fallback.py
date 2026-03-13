from __future__ import annotations

import re
from collections import Counter
from typing import Any


def normalize_for_ocr(image: Any, rotation_deg: int) -> Any:
    """
    Normalize image orientation before OCR using page rotation metadata.
    """
    rot = int(rotation_deg or 0) % 360
    if rot == 90:
        return image.rotate(-90, expand=True)
    if rot == 180:
        return image.rotate(180, expand=True)
    if rot == 270:
        return image.rotate(90, expand=True)
    return image


def _tokenize_words(text: str) -> list[str]:
    # Alphabetic tokens only for OCR corruption checks.
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text or "")


def _non_empty_lines(text: str) -> list[str]:
    return [ln.strip() for ln in str(text or "").splitlines() if ln.strip()]


def is_corrupted_ocr(text: str) -> bool:
    lines = _non_empty_lines(text)
    if len(lines) < 4:
        return True

    words = _tokenize_words(text)
    if len(words) < 30:
        return True

    low_words = [w.lower() for w in words]
    # Exclude ultra-common function words from repetition signal.
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "was", "are", "were",
        "of", "to", "in", "on", "at", "as", "by", "or", "an", "a",
        # Frequent table placeholders / OCR fragments that should not mark corruption.
        "na", "n", "nia", "niaa", "ie",
    }
    rep_words = [w for w in low_words if w not in stop]
    if low_words:
        most_tok, most_cnt = Counter(rep_words or low_words).most_common(1)[0]
        if most_cnt > 20 and len(most_tok) <= 4:
            return True
    return False


def _normalize_spaces(line: str) -> str:
    return re.sub(r"\s+", " ", str(line or "")).strip()


def _clean_ocr_table_text(text: str) -> str:
    lines = _non_empty_lines(text)
    if not lines:
        return ""

    cleaned = []
    for ln in lines:
        c = _normalize_spaces(ln)
        if not c:
            continue
        if len(c) <= 1:
            continue
        cleaned.append(c)
    return "\n".join(cleaned).strip()


def _count_currency_hits(text_upper: str) -> int:
    patterns = [r"£", r"£000", r"£'000", r"£’000", r"\(£", r"CETV"]
    hits = 0
    for p in patterns:
        hits += len(re.findall(re.escape(p), text_upper))
    return hits


def _num_lines_with_2plus_nums(lines: list[str]) -> int:
    pat = re.compile(r"\d[\d,]*")
    count = 0
    for ln in lines:
        if len(pat.findall(ln)) >= 2:
            count += 1
    return count


def _contains_any(text_upper: str, phrases: list[str]) -> list[str]:
    return [p for p in phrases if p.upper() in text_upper]


def accept_and_classify_ocr_table(
    ocr_text: str,
    page_no: int,
    rotation_deg: int,
) -> tuple[bool, str, str, dict]:
    # Case-insensitive matching surface
    text_upper = str(ocr_text or "").upper()
    lines = _non_empty_lines(ocr_text)
    total_chars = len(ocr_text or "")
    digit_ratio = (
        sum(ch.isdigit() for ch in (ocr_text or "")) / max(1, total_chars)
    )

    remuneration_triggers = ["REMUNERATION REPORT", "REMUNERATION"]
    remuneration_support = [
        "SALARY", "BENEFITS", "PENSION", "CETV", "BAND",
        "PERFORMANCE PAY", "BENEFITS IN KIND", "CASH EQUIVALENT TRANSFER VALUE",
    ]
    cash_flow_triggers = ["CASH FLOW STATEMENT", "CASH FLOW"]
    balance_sheet_triggers = ["BALANCE SHEET", "STATEMENT OF FINANCIAL POSITION"]
    net_expenditure_triggers = [
        "NET EXPENDITURE",
        "COMPREHENSIVE NET EXPENDITURE",
        "STATEMENT OF COMPREHENSIVE NET EXPENDITURE",
    ]
    exit_packages_triggers = ["EXIT PACKAGES"]
    statutory_targets_triggers = [
        "REVENUE RESOURCE LIMIT", "CAPITAL RESOURCE LIMIT",
        "CASH REQUIREMENT", "ACTUAL OUTTURN", "VARIANCE",
    ]

    # Kept for compatibility with previous summary/debug semantics.
    financial_core_kw = [
        "BALANCE SHEET", "STATEMENT OF FINANCIAL POSITION",
        "CASH FLOW", "NET EXPENDITURE", "COMPREHENSIVE NET",
        "AS AT 31 MARCH", "FOR THE YEAR ENDED",
        "TOTAL ASSETS", "TOTAL LIABILITIES",
    ]

    currency_hits = _count_currency_hits(text_upper)
    num_lines_2plus = _num_lines_with_2plus_nums(lines)
    matched_rem_trig = _contains_any(text_upper, remuneration_triggers)
    matched_rem_support = _contains_any(text_upper, remuneration_support)
    matched_cash = _contains_any(text_upper, cash_flow_triggers)
    matched_bs = _contains_any(text_upper, balance_sheet_triggers)
    matched_ne = _contains_any(text_upper, net_expenditure_triggers)
    matched_exit = _contains_any(text_upper, exit_packages_triggers)
    matched_stat = _contains_any(text_upper, statutory_targets_triggers)

    corrupted = is_corrupted_ocr(ocr_text)
    remuneration_branch = bool(matched_rem_trig) and (
        len(matched_rem_support) >= 2 or digit_ratio >= 0.05
    )
    strong_numeric_branch = digit_ratio >= 0.10 and (
        currency_hits >= 1 or num_lines_2plus >= 3
    )
    known_header_branch = bool(
        matched_cash or matched_bs or matched_ne or matched_exit or matched_stat
    )

    if corrupted:
        accept = False
    elif remuneration_branch:
        accept = True
    elif strong_numeric_branch:
        accept = True
    elif known_header_branch:
        accept = True
    else:
        accept = False

    # Classification priority (exact order requested).
    if remuneration_branch or (matched_rem_trig and len(matched_rem_support) >= 2):
        table_type = "remuneration_disclosure"
    elif matched_exit:
        table_type = "exit_packages"
    elif matched_cash:
        table_type = "financial_core_cash_flow"
    elif matched_bs:
        table_type = "financial_core_balance_sheet"
    elif matched_ne:
        table_type = "financial_core_net_expenditure"
    elif matched_stat:
        table_type = "statutory_targets"
    else:
        table_type = "scan_table_other"

    cleaned_body = _clean_ocr_table_text(ocr_text)
    cleaned_text = (
        f"TABLE_TYPE: {table_type}\n"
        f"PAGE: {int(page_no)}\n"
        f"ROTATION: {int(rotation_deg)}\n"
        f"{cleaned_body}"
    ).strip()

    debug = {
        "digit_ratio": float(digit_ratio),
        "currency_hits": int(currency_hits),
        "num_lines_with_2plus_nums": int(num_lines_2plus),
        "matched_keywords": sorted(set(_contains_any(text_upper, financial_core_kw))),
        "matched_triggers": sorted(
            set(
                matched_rem_trig
                + matched_cash
                + matched_bs
                + matched_ne
                + matched_exit
                + matched_stat
            )
        ),
        "corrupted_flag": bool(corrupted),
        "rule_a": bool(strong_numeric_branch),
        "rule_b": bool(known_header_branch),
        "rule_c": bool(remuneration_branch),
    }
    return bool(accept), table_type, cleaned_text, debug
