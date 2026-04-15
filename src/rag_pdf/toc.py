from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from rag_pdf.text_normalize import normalize_line


MAJOR_TITLE_HINTS = (
    "performance report",
    "accountability report",
    "financial statements",
    "independent auditor",
    "directions by the scottish ministers",
)

ROMAN_RE = r"(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)"
PREFIX_L2_RE = re.compile(r"^\s*([a-z])(?:[.)])\s+(.+)$", flags=re.IGNORECASE)
PREFIX_L3_RE = re.compile(rf"^\s*({ROMAN_RE})(?:[.)])\s+(.+)$", flags=re.IGNORECASE)
TOC_LINE_RE = re.compile(r"^\s*(.+?)\s+(\d{1,4})\s*$")
TOC_DOT_RE = re.compile(r"^\s*(.+?)\s?\.{2,}\s?(\d{1,4})\s*$")
LEADING_NUM_RE = re.compile(r"^\s*\d{1,4}\b")
TRAILING_PAGE_RE = re.compile(r"\b(\d{1,4})\s*$")


def normalize_title(s: str) -> str:
    t = normalize_line(str(s or "")).lower()
    t = t.replace("’", "'").replace("`", "'")
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9\s.']", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\bthe accountability report\b", "accountability report", t)
    t = re.sub(r"\bthe performance report\b", "performance report", t)
    t = re.sub(r"\bperformance report\b", "performance report", t)
    return t


def _iter_top_texts(raw_top: Any) -> list[str]:
    out: list[str] = []
    if raw_top is None:
        return out
    if isinstance(raw_top, np.ndarray):
        raw_top = raw_top.tolist()
    if isinstance(raw_top, (list, tuple)):
        for item in raw_top:
            if isinstance(item, dict):
                txt = normalize_line(str(item.get("text", "")))
            else:
                txt = normalize_line(str(item))
            if txt:
                out.append(txt)
    return out


def _clean_lines(text: str) -> list[str]:
    return [normalize_line(x) for x in str(text or "").splitlines() if normalize_line(x)]


def _line_is_page_number(ln: str, n_pages: int) -> bool:
    s = normalize_line(ln)
    if not s or not s.isdigit():
        return False
    pv = int(s)
    return 1 <= pv <= max(n_pages + 30, 250)


def _extract_stream_lines(row: pd.Series) -> list[str]:
    top = _iter_top_texts(row.get("top_lines"))
    # Prefer top_lines for ToC parsing: they preserve visual row order better than flattened clean_text.
    if len(top) >= 5:
        return [normalize_line(x) for x in top if normalize_line(x)]
    return _clean_lines(str(row.get("clean_text", "")))


def _extract_toc_line(line: str, n_pages: int) -> tuple[str, int] | None:
    m = TOC_DOT_RE.match(line) or TOC_LINE_RE.match(line)
    if not m:
        return None
    title = normalize_line(m.group(1))
    try:
        page_no = int(m.group(2))
    except Exception:
        return None
    if not title:
        return None
    # Reject obvious non-ToC years or invalid page ranges.
    if page_no <= 0 or page_no > max(n_pages + 30, 250):
        return None
    if len(title) < 3 or len(title) > 180:
        return None
    return title, page_no


def detect_toc_pages(pages_df: pd.DataFrame) -> list[int]:
    scores: dict[int, float] = {}
    n_pages = int(pages_df["page"].max()) if len(pages_df) else 0
    for _, row in pages_df.iterrows():
        p = int(row.get("page", 0) or 0)
        if p <= 0:
            continue
        top_texts = _iter_top_texts(row.get("top_lines"))
        lines = _clean_lines(str(row.get("clean_text", "")))
        if not lines:
            continue
        s = 0.0
        top_blob = " ".join(top_texts[:8]).lower()
        if "contents" in top_blob:
            s += 0.65
        numbered_lines = 0
        trailing_page_lines = 0
        for ln in lines[:200]:
            if LEADING_NUM_RE.match(ln):
                numbered_lines += 1
            if _extract_toc_line(ln, n_pages) is not None:
                trailing_page_lines += 1
        density = trailing_page_lines / max(1, min(len(lines), 200))
        if trailing_page_lines >= 6:
            s += 0.35
        if density >= 0.15:
            s += 0.35
        elif density >= 0.08:
            s += 0.2
        if numbered_lines >= 8:
            s += 0.15
        scores[p] = s

    # Seed pages above threshold and extend short consecutive windows.
    seed = sorted([p for p, sc in scores.items() if sc >= 0.55])
    if not seed:
        return []
    out: set[int] = set(seed)
    for p in seed:
        for q in (p - 1, p + 1):
            if scores.get(q, 0.0) >= 0.28:
                out.add(q)
    return sorted(out)


def _infer_level(title: str) -> int:
    u = normalize_title(title)
    if PREFIX_L3_RE.match(title):
        return 3
    if PREFIX_L2_RE.match(title):
        return 2
    if any(h in u for h in MAJOR_TITLE_HINTS):
        return 1
    # fallback: short all-caps headings likely level 1.
    alpha = [c for c in title if c.isalpha()]
    upper_ratio = (sum(c.isupper() for c in alpha) / len(alpha)) if alpha else 0.0
    if upper_ratio >= 0.75:
        return 1
    return 2


def parse_toc(pages_df: pd.DataFrame, toc_page_indices: list[int]) -> list[dict[str, Any]]:
    if not toc_page_indices:
        return []
    by_page = {int(r["page"]): r for _, r in pages_df.iterrows()}
    entries: list[dict[str, Any]] = []
    carry_title: str | None = None
    carry_level: int | None = None

    n_pages = int(pages_df["page"].max()) if len(pages_df) else 0
    for p in sorted(toc_page_indices):
        row = by_page.get(p)
        if row is None:
            continue
        lines = _extract_stream_lines(row)
        for raw in lines:
            ln = normalize_line(raw)
            if not ln:
                continue
            if ln.lower() in {"contents", "page"}:
                continue

            # Common ToC pattern from top_lines: title line followed by page number line.
            if _line_is_page_number(ln, n_pages):
                if carry_title:
                    lvl = carry_level if carry_level is not None else _infer_level(carry_title)
                    entries.append(
                        {
                            "toc_page_pdf": p,
                            "start_printed_page": int(ln),
                            "title_raw": carry_title,
                            "title_norm": normalize_title(carry_title),
                            "level": int(lvl),
                        }
                    )
                    carry_title = None
                    carry_level = None
                continue

            parsed = _extract_toc_line(ln, n_pages)
            if parsed is not None:
                title, pp = parsed
                if carry_title:
                    title = normalize_line(f"{carry_title} {title}")
                    lvl = carry_level if carry_level is not None else _infer_level(title)
                    carry_title = None
                    carry_level = None
                else:
                    lvl = _infer_level(title)
                entries.append(
                    {
                        "toc_page_pdf": p,
                        "start_printed_page": int(pp),
                        "title_raw": title,
                        "title_norm": normalize_title(title),
                        "level": int(lvl),
                    }
                )
            else:
                # likely wrapped title line waiting for trailing page number line.
                if TRAILING_PAGE_RE.search(ln):
                    continue
                if len(ln.split()) <= 1:
                    continue
                if ln.lower().startswith("note:"):
                    continue
                if carry_title and _infer_level(ln) == 1 and _infer_level(carry_title) >= 2:
                    # preserve an unlabeled major heading; page will be inferred from next numbered entry.
                    entries.append(
                        {
                            "toc_page_pdf": p,
                            "start_printed_page": None,
                            "title_raw": carry_title,
                            "title_norm": normalize_title(carry_title),
                            "level": int(carry_level if carry_level is not None else _infer_level(carry_title)),
                        }
                    )
                    carry_title = ln
                    carry_level = _infer_level(carry_title)
                else:
                    carry_title = normalize_line(f"{carry_title or ''} {ln}")
                    carry_level = _infer_level(carry_title)

        # Fallback for flattened OCR/text where entire ToC appears as one line.
        if len(entries) < 3:
            blob = normalize_line(str(row.get("clean_text", "")))
            if blob:
                entries.extend(_parse_inline_toc_blob(blob, p, n_pages))

    # Flush trailing title if it is a plausible major heading without explicit page number.
    if carry_title and _infer_level(carry_title) == 1:
        entries.append(
            {
                "toc_page_pdf": int(sorted(toc_page_indices)[-1]),
                "start_printed_page": None,
                "title_raw": carry_title,
                "title_norm": normalize_title(carry_title),
                "level": 1,
            }
        )

    # Infer printed pages for unlabeled major headings from the next explicit entry.
    for i, e in enumerate(entries):
        if e.get("start_printed_page") is not None:
            continue
        inferred = None
        for j in range(i + 1, len(entries)):
            nxt = entries[j].get("start_printed_page")
            if isinstance(nxt, int) and nxt > 0:
                inferred = int(nxt)
                break
        if inferred is None:
            for j in range(i - 1, -1, -1):
                prv = entries[j].get("start_printed_page")
                if isinstance(prv, int) and prv > 0:
                    inferred = int(prv)
                    break
        if inferred is not None:
            e["start_printed_page"] = int(inferred)

    # de-duplicate by normalized title + printed page while preserving order.
    dedup: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for e in entries:
        pp = int(e["start_printed_page"]) if e.get("start_printed_page") is not None else -1
        if pp <= 0:
            continue
        key = (str(e["title_norm"]), pp)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(e)
    return dedup


def _clean_segment_title(seg: str) -> str:
    s = normalize_line(seg)
    if not s:
        return s
    if s.lower().startswith("page "):
        s = normalize_line(s[5:])
    # Keep suffix after latest subsection/numbering marker if multiple chunks merged.
    parts = re.split(r"(?=(?:\b[a-z]\)|\b(?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\)|\b\d+(?:\.\d+)+\b))", s, flags=re.IGNORECASE)
    if len(parts) > 1:
        cand = normalize_line(parts[-1])
        if cand:
            s = cand
    # If mixed with multiple major bands, prefer the latest band.
    for hint in (
        "Financial Statements",
        "Independent Auditor",
        "Accountability Report",
        "Performance Report",
    ):
        idx = s.lower().rfind(hint.lower())
        if idx > 0:
            s = normalize_line(s[idx:])
    return s


def _parse_inline_toc_blob(blob: str, toc_page_pdf: int, n_pages: int) -> list[dict[str, Any]]:
    toks = blob.split()
    if toks and toks[0].lower() == "page":
        toks = toks[1:]
    if not toks:
        return []
    idxs: list[int] = []
    for i, tk in enumerate(toks):
        if tk.isdigit():
            pv = int(tk)
            if 1 <= pv <= max(n_pages + 30, 250):
                idxs.append(i)
    out: list[dict[str, Any]] = []
    last = -1
    for i in idxs:
        if i <= last + 1:
            last = i
            continue
        title = _clean_segment_title(" ".join(toks[last + 1 : i]))
        if not title or len(title.split()) < 2:
            last = i
            continue
        pp = int(toks[i])
        out.append(
            {
                "toc_page_pdf": toc_page_pdf,
                "start_printed_page": pp,
                "title_raw": title,
                "title_norm": normalize_title(title),
                "level": _infer_level(title),
            }
        )
        last = i
    return out


def _title_search_tokens(norm_title: str) -> list[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", norm_title) if len(t) >= 3]
    stop = {"the", "and", "for", "year", "to", "of", "report", "statement", "accounts"}
    out = [t for t in toks if t not in stop]
    return out or toks


def resolve_printed_to_pdf_offset(pages_df: pd.DataFrame, toc_items: list[dict[str, Any]]) -> dict[str, Any]:
    if not toc_items:
        return {"offset": 0, "confidence": 0.0, "support_count": 0}
    work = sorted(
        toc_items,
        key=lambda x: (len(str(x.get("title_norm", ""))), int(x.get("start_printed_page", 0))),
        reverse=True,
    )
    candidates = [x for x in work if len(_title_search_tokens(str(x.get("title_norm", "")))) >= 2][:5]
    if not candidates:
        candidates = work[:5]
    offsets: list[int] = []

    toc_pages = set(detect_toc_pages(pages_df))
    pages = pages_df[["page", "clean_text"]].copy()
    pages["clean_norm"] = pages["clean_text"].fillna("").astype(str).map(normalize_title)
    for item in candidates:
        printed = int(item.get("start_printed_page", 0) or 0)
        if printed <= 0:
            continue
        toks = _title_search_tokens(str(item.get("title_norm", "")))
        if not toks:
            continue
        best_page = None
        best_ratio = 0.0
        for _, r in pages.iterrows():
            pdf_page = int(r["page"])
            if pdf_page in toc_pages:
                continue
            txt = str(r["clean_norm"])
            hit = sum(1 for t in toks if t in txt)
            ratio = float(hit / max(1, len(toks)))
            if hit >= min(2, len(toks)) and ratio > best_ratio:
                best_ratio = ratio
                best_page = pdf_page
        if best_page is not None and best_ratio >= 0.34:
            offsets.append(int(best_page - printed))

    if not offsets:
        return {"offset": 0, "confidence": 0.15, "support_count": 0}
    c = Counter(offsets)
    best_offset, best_count = c.most_common(1)[0]
    mode_ratio = float(best_count / max(1, len(offsets)))
    toc_strength = min(1.0, float(len(toc_items) / 12.0))
    conf = float((0.5 * mode_ratio) + (0.5 * toc_strength))
    return {"offset": int(best_offset), "confidence": conf, "support_count": int(best_count)}


@dataclass
class TocSpan:
    start_page_pdf: int
    end_page_pdf: int
    section_title: str
    subsection_title: str
    level: int
    title_norm: str


def build_toc_spans(toc_items: list[dict[str, Any]], offset: int, n_pdf_pages: int) -> list[dict[str, Any]]:
    if not toc_items:
        return []
    items: list[dict[str, Any]] = []
    for it in toc_items:
        pp = int(it.get("start_printed_page", 0) or 0)
        if pp <= 0:
            continue
        sp = pp + int(offset)
        sp = max(1, min(n_pdf_pages, sp))
        x = dict(it)
        x["start_pdf_page"] = int(sp)
        items.append(x)
    if not items:
        return []
    items = sorted(items, key=lambda x: (int(x["start_pdf_page"]), int(x.get("level", 9))))

    spans: list[dict[str, Any]] = []
    current_section = "Unknown"
    for i, it in enumerate(items):
        start = int(it["start_pdf_page"])
        next_start = int(items[i + 1]["start_pdf_page"]) if i + 1 < len(items) else (n_pdf_pages + 1)
        end = max(start, min(n_pdf_pages, next_start - 1))
        lvl = int(it.get("level", 2))
        title_raw = str(it.get("title_raw", "")).strip() or "Unknown"
        if lvl == 1:
            current_section = title_raw
            subsection = "Unknown"
        else:
            subsection = title_raw
        spans.append(
            {
                "start_page_pdf": int(start),
                "end_page_pdf": int(end),
                "section_title": current_section,
                "subsection_title": subsection if lvl >= 2 else "Unknown",
                "level": int(lvl),
                "title_raw": title_raw,
                "title_norm": str(it.get("title_norm", "")),
                "start_printed_page": int(it.get("start_printed_page", 0)),
            }
        )
    return spans


def toc_prior_for_page(spans: list[dict[str, Any]], page_no: int) -> tuple[str, str]:
    for sp in spans:
        if int(sp["start_page_pdf"]) <= page_no <= int(sp["end_page_pdf"]):
            return str(sp.get("section_title", "Unknown")), str(sp.get("subsection_title", "Unknown"))
    return "Unknown", "Unknown"


def token_set_overlap_ratio(a: str, b: str) -> float:
    ta = set(re.findall(r"[a-z0-9]+", normalize_title(a)))
    tb = set(re.findall(r"[a-z0-9]+", normalize_title(b)))
    if not ta or not tb:
        return 0.0
    return float(len(ta.intersection(tb)) / len(ta.union(tb)))


def fuzzy_match_title(
    proposed: str,
    allow_list: list[str],
    token_overlap_threshold: float = 0.6,
    sequence_threshold: float = 0.75,
) -> tuple[bool, str | None, float]:
    if not proposed or not allow_list:
        return False, None, 0.0
    p = normalize_title(proposed)
    best_title = None
    best_score = 0.0
    for cand in allow_list:
        c = normalize_title(cand)
        tok = token_set_overlap_ratio(p, c)
        seq = difflib.SequenceMatcher(a=p, b=c).ratio()
        score = max(tok, seq)
        if score > best_score:
            best_score = score
            best_title = cand
        if tok >= token_overlap_threshold or seq >= sequence_threshold:
            return True, cand, score
    return False, best_title, best_score
