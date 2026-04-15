from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from rag_pdf.headings import (
    is_global_boilerplate_heading,
    is_part_label,
    is_section_anchor_line,
    looks_like_heading_text_only,
    looks_like_lettered_subsection,
    looks_like_numbered_heading,
)
from rag_pdf.text_normalize import normalize_line
from rag_pdf.toc import (
    build_toc_spans,
    detect_toc_pages,
    fuzzy_match_title,
    normalize_title,
    parse_toc,
    resolve_printed_to_pdf_offset,
    toc_prior_for_page,
)


def _clean_heading_label(text: str) -> str:
    s = normalize_line(str(text or ""))
    if not s:
        return "Unknown"
    s = re.sub(r"\(\s*cont(?:inued)?\.?\s*\)", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\bcontinued\b", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip()
    if not s or s.lower() == "unknown":
        return "Unknown"
    return s


def _canonicalize_section_title(text: str) -> str:
    s = _clean_heading_label(text)
    if s == "Unknown":
        return s
    n = normalize_title(s)
    # Legacy financial sections often appear as statement/note headings rather than the
    # literal "Financial Statements" label.
    if (
        "financial statements" in n
        or "consolidated statement of" in n
        or "statement of consolidated" in n
        or "statement of financial position" in n
        or "statement of comprehensive net expenditure" in n
        or "statement of cash flow" in n
        or "statement of cashflows" in n
        or "statement of cash flows" in n
        or "statement of changes in taxpayer" in n
        or "notes to the accounts" in n
        or n.startswith("note ")
        or n.startswith("accounting policies")
        or n.startswith("accounting convention")
    ):
        return "FINANCIAL STATEMENTS"
    if "performance report" in n:
        return "PERFORMANCE REPORT"
    if "accountability report" in n:
        return "ACCOUNTABILITY REPORT"
    if "independent auditor" in n:
        return "INDEPENDENT AUDITOR'S REPORT"
    if "directions by the scottish ministers" in n:
        return "DIRECTIONS BY THE SCOTTISH MINISTERS"
    return s.upper()


def _canonicalize_subsection_title(text: str) -> str:
    s = _clean_heading_label(text)
    if s == "Unknown":
        return s
    note = _extract_note_subsection(s)
    if note:
        return note.upper()
    # normalize letter and roman prefixes for consistency
    m_letter = re.match(r"^\s*([A-Za-z])[.)]?\s+(.+)$", s)
    if m_letter:
        return f"{m_letter.group(1).upper()}) {m_letter.group(2).upper()}"
    m_roman = re.match(r"^\s*((?:i|ii|iii|iv|v|vi|vii|viii|ix|x))[.)]?\s+(.+)$", s, flags=re.IGNORECASE)
    if m_roman:
        return f"{m_roman.group(1).lower()}) {m_roman.group(2).upper()}"
    return s.upper()


def _iter_top_lines(raw_top: Any) -> list[str]:
    top_lines: list[str] = []
    if raw_top is None:
        return top_lines
    if isinstance(raw_top, np.ndarray):
        raw_top = raw_top.tolist()
    if isinstance(raw_top, (list, tuple)):
        for item in raw_top:
            if isinstance(item, dict):
                txt = str(item.get("text", ""))
            else:
                txt = str(item)
            norm = normalize_line(txt)
            if norm:
                top_lines.append(norm)
    return top_lines


def _iter_heading_candidates(raw_candidates: Any) -> list[str]:
    if raw_candidates is None or raw_candidates is False:
        return []
    if isinstance(raw_candidates, np.ndarray):
        raw_candidates = raw_candidates.tolist()
    if isinstance(raw_candidates, str):
        return []
    if isinstance(raw_candidates, (list, tuple)):
        return [str(x) for x in raw_candidates if normalize_line(str(x))]
    if hasattr(raw_candidates, "__iter__"):
        return [str(x) for x in raw_candidates if normalize_line(str(x))]
    return []


SIGNATURE_NOISE_PATTERNS = [
    re.compile(r"^\s*docusign\s+envelope\s+id\s*:\s*", flags=re.IGNORECASE),
    re.compile(r"^\s*digitally\s+signed\b", flags=re.IGNORECASE),
    re.compile(r"^\s*electronic(?:ally)?\s+signed\b", flags=re.IGNORECASE),
    re.compile(r"^\s*signature\s+verified\b", flags=re.IGNORECASE),
    re.compile(r"^\s*verification\s+(?:code|id|url|link|status)\b", flags=re.IGNORECASE),
    re.compile(r"^\s*certificate(?:\s+of\s+completion)?\b", flags=re.IGNORECASE),
    re.compile(r"^\s*certification\s+stamp\b", flags=re.IGNORECASE),
]

NOTE_SUBSECTION_RE = re.compile(r"^\s*note\s+(\d+)\.?\s*(.*)$", flags=re.IGNORECASE)


def _is_signature_or_verification_line(line: str) -> bool:
    s = normalize_line(line)
    if not s:
        return False
    return any(p.search(s) for p in SIGNATURE_NOISE_PATTERNS)


def _extract_note_subsection(line: str) -> str | None:
    s = normalize_line(line)
    if not s:
        return None
    m = NOTE_SUBSECTION_RE.match(s)
    if not m:
        return None
    num = m.group(1)
    tail = normalize_line(m.group(2) or "")
    if tail:
        return f"Note {num}. {tail}"
    return f"Note {num}"


def _detect_page_heading_signals(row: pd.Series) -> tuple[str | None, str | None, list[str], list[str]]:
    text = str(row.get("clean_text") or "")
    lines = [normalize_line(x) for x in text.splitlines() if normalize_line(x)]
    top_lines = _iter_top_lines(row.get("top_lines"))
    heading_candidates = _iter_heading_candidates(row.get("heading_candidates"))

    # Remove signature/verification/certification artifacts before heading detection.
    lines = [ln for ln in lines if not _is_signature_or_verification_line(ln)]
    top_lines = [ln for ln in top_lines if not _is_signature_or_verification_line(ln)]
    heading_candidates = [ln for ln in heading_candidates if not _is_signature_or_verification_line(ln)]

    section_found = None
    subsection_found = None

    def _looks_like_subsection(line: str) -> bool:
        return looks_like_lettered_subsection(line) or looks_like_numbered_heading(line)

    if top_lines:
        for i, line in enumerate(top_lines):
            if is_part_label(line):
                continue
            if (
                re.match(r"^[A-Z][.)]?$", line)
                and i + 1 < len(top_lines)
                and subsection_found is None
            ):
                combined = f"{line[0]} {top_lines[i + 1]}"
                if _looks_like_subsection(combined):
                    subsection_found = combined
                    continue
            if subsection_found is None and _looks_like_subsection(line):
                subsection_found = line
                continue
            if section_found is None and (
                is_section_anchor_line(line)
                or (looks_like_heading_text_only(line) and not is_global_boilerplate_heading(line))
            ):
                section_found = line
            if section_found and subsection_found:
                break
    elif heading_candidates:
        for cand in heading_candidates:
            if subsection_found is None and _looks_like_subsection(cand):
                subsection_found = cand
                continue
            if section_found is None and looks_like_heading_text_only(cand):
                section_found = cand
            if section_found and subsection_found:
                break
    else:
        for line in lines[:25]:
            if not is_part_label(line):
                if subsection_found is None and _looks_like_subsection(line):
                    subsection_found = line
                    continue
                if section_found is None and looks_like_heading_text_only(line):
                    section_found = line
            if section_found and subsection_found:
                break

    if subsection_found is None and text:
        m = re.search(r"\b([A-Z])[.)]?\s+([A-Z][A-Z ]{3,})\b", text)
        if m:
            candidate = f"{m.group(1)} {normalize_line(m.group(2))}"
            if _looks_like_subsection(candidate):
                subsection_found = candidate

    if subsection_found is None:
        for ln in top_lines[:20] + lines[:50]:
            note = _extract_note_subsection(ln)
            if note:
                subsection_found = note
                break

    return section_found, subsection_found, top_lines, lines


def _labels_to_sections(
    pages_df: pd.DataFrame,
    labels: list[dict[str, Any]],
) -> pd.DataFrame:
    if not labels:
        return pd.DataFrame(
            columns=[
                "doc_id",
                "report_year",
                "period_end_date",
                "report_year_source",
                "run_date_utc",
                "part",
                "section_title",
                "subsection_title",
                "page_start",
                "page_end",
                "section_text",
                "word_count",
            ]
        )

    by_page_text = {
        int(r["page"]): str(r.get("clean_text") or "")
        for _, r in pages_df.iterrows()
    }
    rows: list[dict[str, Any]] = []
    cur = None
    start_page = None
    text_buf: list[str] = []

    for lab in sorted(labels, key=lambda x: int(x["page"])):
        page = int(lab["page"])
        key = (str(lab["part"]), str(lab["section_title"]), str(lab["subsection_title"]))
        if cur is None:
            cur = key
            start_page = page
            text_buf = [by_page_text.get(page, "")]
            continue
        if key == cur and page == (rows[-1]["page_end"] + 1 if rows else start_page + len(text_buf)):
            text_buf.append(by_page_text.get(page, ""))
            continue
        rows.append(
            {
                "doc_id": pages_df["doc_id"].iloc[0],
                "report_year": pages_df["report_year"].iloc[0],
                "period_end_date": pages_df["period_end_date"].iloc[0],
                "report_year_source": pages_df["report_year_source"].iloc[0],
                "run_date_utc": pages_df["run_date_utc"].iloc[0],
                "part": cur[0] or "Unknown",
                "section_title": cur[1] or "Unknown",
                "subsection_title": cur[2] or "Unknown",
                "page_start": int(start_page),
                "page_end": int(start_page + len(text_buf) - 1),
                "section_text": "\n".join(text_buf).strip(),
            }
        )
        cur = key
        start_page = page
        text_buf = [by_page_text.get(page, "")]

    rows.append(
        {
            "doc_id": pages_df["doc_id"].iloc[0],
            "report_year": pages_df["report_year"].iloc[0],
            "period_end_date": pages_df["period_end_date"].iloc[0],
            "report_year_source": pages_df["report_year_source"].iloc[0],
            "run_date_utc": pages_df["run_date_utc"].iloc[0],
            "part": cur[0] or "Unknown",
            "section_title": cur[1] or "Unknown",
            "subsection_title": cur[2] or "Unknown",
            "page_start": int(start_page),
            "page_end": int(start_page + len(text_buf) - 1),
            "section_text": "\n".join(text_buf).strip(),
        }
    )

    df = pd.DataFrame(rows)
    df["word_count"] = df["section_text"].fillna("").astype(str).str.split().str.len()
    return df


def _infer_labels(
    pages_df: pd.DataFrame,
    *,
    toc_spans: list[dict[str, Any]] | None = None,
    toc_confidence: float = 0.0,
    toc_confidence_threshold: float = 0.6,
    allow_cross_major_override: bool = False,
    subsection_allow_list: list[str] | None = None,
    token_overlap_threshold: float = 0.6,
    sequence_threshold: float = 0.75,
) -> tuple[list[dict[str, Any]], dict[str, Any], pd.DataFrame]:
    toc_spans = toc_spans or []
    subsection_allow_list = subsection_allow_list or []
    toc_active = bool(toc_spans) and float(toc_confidence) >= float(toc_confidence_threshold)

    labels: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    override_count = 0
    toc_labeled_pages = 0
    subsection_reject_count = 0

    current_part = "Unknown"
    current_section = "Unknown"
    current_subsection = "Unknown"

    for _, row in pages_df.sort_values("page").iterrows():
        page_no = int(row["page"])
        text = str(row.get("clean_text") or "")

        top_lines = _iter_top_lines(row.get("top_lines"))
        for l in top_lines:
            p = is_part_label(l)
            if p:
                current_part = p
                break

        prior_section, prior_subsection = (
            toc_prior_for_page(toc_spans, page_no) if toc_active else ("Unknown", "Unknown")
        )
        if toc_active and (prior_section != "Unknown" or prior_subsection != "Unknown"):
            toc_labeled_pages += 1
        if toc_active:
            if prior_section != "Unknown":
                current_section = _canonicalize_section_title(prior_section)
            if prior_subsection != "Unknown":
                current_subsection = _canonicalize_subsection_title(prior_subsection)

        section_found, subsection_found, _, raw_lines = _detect_page_heading_signals(row)

        if section_found:
            accept_section = True
            if toc_active:
                if (
                    prior_section != "Unknown"
                    and normalize_title(prior_section) != normalize_title(section_found)
                    and not allow_cross_major_override
                ):
                    accept_section = False
            if accept_section:
                if toc_active and prior_section != "Unknown" and normalize_title(prior_section) != normalize_title(section_found):
                    override_count += 1
                current_section = _canonicalize_section_title(str(section_found))
                current_subsection = "Unknown"

        if subsection_found:
            accept_subsection = True
            chosen_subsection = str(subsection_found)

            # Financial statements are structured by notes; ignore non-note subsection proposals.
            if _canonicalize_section_title(current_section) == "FINANCIAL STATEMENTS":
                if _extract_note_subsection(chosen_subsection) is None:
                    accept_subsection = False

            if toc_active and subsection_allow_list and not looks_like_numbered_heading(chosen_subsection):
                ok, matched, score = fuzzy_match_title(
                    proposed=chosen_subsection,
                    allow_list=subsection_allow_list,
                    token_overlap_threshold=token_overlap_threshold,
                    sequence_threshold=sequence_threshold,
                )
                if not ok:
                    accept_subsection = False
                    subsection_reject_count += 1
                    rejected_rows.append(
                        {
                            "doc_id": str(row.get("doc_id", "")),
                            "page": page_no,
                            "proposed_subsection": chosen_subsection,
                            "toc_subsection": prior_subsection if prior_subsection != "Unknown" else "",
                            "reason": f"allow_list_reject(score={score:.3f})",
                            "top_lines_snippet": " | ".join(raw_lines[:5]),
                        }
                    )
                elif matched:
                    chosen_subsection = str(matched)
            if accept_subsection:
                if toc_active and prior_subsection != "Unknown" and normalize_title(prior_subsection) != normalize_title(chosen_subsection):
                    override_count += 1
                current_subsection = _canonicalize_subsection_title(chosen_subsection)
            elif toc_active:
                current_subsection = _canonicalize_subsection_title(prior_subsection) if prior_subsection != "Unknown" else "Unknown"

        labels.append(
            {
                "page": page_no,
                "part": current_part or "Unknown",
                "section_title": _canonicalize_section_title(current_section or "Unknown"),
                "subsection_title": _canonicalize_subsection_title(current_subsection or "Unknown"),
                "toc_prior_section": _canonicalize_section_title(prior_section),
                "toc_prior_subsection": _canonicalize_subsection_title(prior_subsection),
            }
        )

    diag = {
        "toc_coverage_pages": int(toc_labeled_pages),
        "toc_override_count": int(override_count),
        "subsection_reject_count": int(subsection_reject_count),
        "toc_active": bool(toc_active),
    }
    rejected_df = pd.DataFrame(rejected_rows)
    return labels, diag, rejected_df


def build_sections_from_pages(
    pages_df: pd.DataFrame,
    *,
    return_diagnostics: bool = False,
    toc_confidence_threshold: float = 0.6,
    allow_cross_major_override: bool = False,
    token_overlap_threshold: float = 0.6,
    sequence_threshold: float = 0.75,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Infer document sections from page-level headings.

    Default behavior remains compatible with the previous interface.
    If `return_diagnostics=True`, returns `(sections_df, diagnostics)`.
    """
    if len(pages_df) == 0:
        empty = pd.DataFrame(
            columns=[
                "doc_id",
                "report_year",
                "period_end_date",
                "report_year_source",
                "run_date_utc",
                "part",
                "section_title",
                "subsection_title",
                "page_start",
                "page_end",
                "section_text",
                "word_count",
            ]
        )
        if return_diagnostics:
            return empty, {}
        return empty

    n_pages = int(pages_df["page"].max())
    toc_pages = detect_toc_pages(pages_df)
    toc_items = parse_toc(pages_df, toc_pages)
    offset_info = resolve_printed_to_pdf_offset(pages_df, toc_items)
    toc_spans = build_toc_spans(toc_items, int(offset_info["offset"]), n_pages)
    toc_allow = [str(x.get("title_raw", "")) for x in toc_items if int(x.get("level", 99)) >= 2]

    # Baseline (heuristic-only) unknown rate for diagnostics.
    baseline_labels, _, _ = _infer_labels(pages_df, toc_spans=[], toc_confidence=0.0)
    baseline_sub_unknown = float(
        np.mean(
            [
                1.0 if str(x.get("subsection_title", "Unknown")).strip().lower() == "unknown" else 0.0
                for x in baseline_labels
            ]
        )
    )

    labels, infer_diag, rejected_df = _infer_labels(
        pages_df,
        toc_spans=toc_spans,
        toc_confidence=float(offset_info.get("confidence", 0.0)),
        toc_confidence_threshold=float(toc_confidence_threshold),
        allow_cross_major_override=bool(allow_cross_major_override),
        subsection_allow_list=toc_allow,
        token_overlap_threshold=float(token_overlap_threshold),
        sequence_threshold=float(sequence_threshold),
    )
    sections_df = _labels_to_sections(pages_df, labels)

    final_sub_unknown = float(
        np.mean(
            [
                1.0 if str(x.get("subsection_title", "Unknown")).strip().lower() == "unknown" else 0.0
                for x in labels
            ]
        )
    )

    toc_rows: list[dict[str, Any]] = []
    for it in toc_items:
        mapped = int(it.get("start_printed_page", 0)) + int(offset_info.get("offset", 0))
        toc_rows.append(
            {
                "level": int(it.get("level", 0)),
                "title_raw": str(it.get("title_raw", "")),
                "title_norm": str(it.get("title_norm", "")),
                "start_printed_page": int(it.get("start_printed_page", 0)),
                "start_pdf_page": int(max(1, min(n_pages, mapped))),
                "confidence": float(offset_info.get("confidence", 0.0)),
            }
        )
    toc_df = pd.DataFrame(toc_rows)

    diagnostics = {
        "toc_detected": bool(len(toc_pages) > 0 and len(toc_items) > 0),
        "toc_pages_count": int(len(toc_pages)),
        "toc_items_count": int(len(toc_items)),
        "toc_offset": int(offset_info.get("offset", 0)),
        "toc_offset_support_count": int(offset_info.get("support_count", 0)),
        "toc_offset_confidence": float(offset_info.get("confidence", 0.0)),
        "toc_coverage_pct": float(infer_diag["toc_coverage_pages"] / max(1, len(labels))),
        "toc_override_rate": float(infer_diag["toc_override_count"] / max(1, len(labels))),
        "subsection_reject_count": int(infer_diag["subsection_reject_count"]),
        "subsection_unknown_pct_before": float(baseline_sub_unknown),
        "subsection_unknown_pct_after": float(final_sub_unknown),
        "toc_spans_count": int(len(toc_spans)),
        "toc_df": toc_df,
        "rejected_subsections_df": rejected_df,
    }

    if return_diagnostics:
        return sections_df, diagnostics
    return sections_df


def find_section_for_page(sections_df: pd.DataFrame, page_no: int) -> tuple[str, str, str]:
    """
    Find which section a page belongs to.

    Returns:
        (part, section_title, subsection_title)
    """
    if len(sections_df) == 0:
        return "Unknown", "Unknown", "Unknown"

    m = sections_df[(sections_df["page_start"] <= page_no) & (sections_df["page_end"] >= page_no)]
    if len(m) == 0:
        return "Unknown", "Unknown", "Unknown"
    r = m.iloc[-1]
    return str(r["part"]), str(r["section_title"]), str(r.get("subsection_title", "Unknown"))
