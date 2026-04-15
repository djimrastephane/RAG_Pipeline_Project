from __future__ import annotations

import re
from typing import Any, Optional

import pandas as pd


NUMERIC_PATTERN = re.compile(r"^\(?-?\d[\d,]*(?:\.\d+)?\)?%?$")


def _normalize_key(text: str) -> str:
    """Normalize text into a lowercase snake_case key for matching."""
    t = str(text or "").strip().lower()
    t = t.replace("&", " and ")
    t = re.sub(r"[^a-z0-9]+", "_", t)
    t = re.sub(r"_+", "_", t).strip("_")
    return t


def _detect_quarter(header: str) -> Optional[str]:
    """Extract quarter token (q1..q4) from a header string when present."""
    h = str(header or "").strip().lower()
    for q in ("q1", "q2", "q3", "q4"):
        if q in h:
            return q
    return None


def _coerce_number(value_raw: str) -> tuple[Optional[float], bool]:
    """Parse numeric-looking cell text into float and percent flag."""
    txt = str(value_raw or "").strip()
    if not txt or txt in {"-", "—", "n/a", "na", "N/A"}:
        return None, False
    if not NUMERIC_PATTERN.match(txt):
        return None, False
    is_percent = txt.endswith("%")
    neg = txt.startswith("(") and txt.endswith(")")
    cleaned = txt.strip("()%").replace(",", "")
    try:
        val = float(cleaned)
        if neg:
            val = -val
        return val, is_percent
    except Exception:
        return None, False


def extract_table_facts_from_dataframe(
    df: pd.DataFrame,
    *,
    doc_id: str,
    corpus_id: str,
    report_year: Optional[str],
    period_end_date: Optional[str],
    run_date_utc: str,
    page: int,
    table_id: str,
    table_type: Optional[str],
) -> list[dict[str, Any]]:
    """
    Canonicalize a table into row/column numeric facts.

    Strategy:
    - Treat first column as row label.
    - Treat each remaining column header as a dimension.
    - Emit one fact for each numeric-like cell.
    """
    if df is None or len(df) == 0:
        return []

    view = df.fillna("").astype(str)
    if view.shape[1] < 2:
        return []

    headers = [str(c).strip() for c in view.columns]
    if not headers[0]:
        headers[0] = "row_label"

    out: list[dict[str, Any]] = []
    for _, row in view.iterrows():
        row_label_raw = str(row.iloc[0]).strip()
        if not row_label_raw:
            continue
        row_label_norm = _normalize_key(row_label_raw)
        for col_idx in range(1, len(headers)):
            col_raw = headers[col_idx]
            col_norm = _normalize_key(col_raw)
            quarter = _detect_quarter(col_raw)
            value_raw = str(row.iloc[col_idx]).strip()
            value_num, is_percent = _coerce_number(value_raw)
            if value_num is None:
                continue
            out.append(
                {
                    "doc_id": doc_id,
                    "corpus_id": corpus_id,
                    "report_year": report_year,
                    "period_end_date": period_end_date,
                    "run_date_utc": run_date_utc,
                    "page": int(page),
                    "table_id": table_id,
                    "table_type": table_type or "unknown",
                    "row_label_raw": row_label_raw,
                    "row_label_norm": row_label_norm,
                    "column_header_raw": col_raw,
                    "column_header_norm": col_norm,
                    "quarter": quarter,
                    "value_raw": value_raw,
                    "value_num": float(value_num),
                    "is_percent": bool(is_percent),
                }
            )
    return out


def _split_md_row(line: str) -> list[str]:
    """Split one markdown table row into trimmed cells."""
    return [p.strip() for p in line.strip().strip("|").split("|")]


def _is_sep_row(line: str) -> bool:
    """Check whether a markdown row is an alignment separator row."""
    if "|" not in line:
        return False
    cells = _split_md_row(line)
    return bool(cells) and all(c and set(c) <= set("-: ") and "-" in c for c in cells)


def markdown_table_to_dataframe(table_md: str) -> pd.Optional[DataFrame]:
    """Convert a markdown table block into a DataFrame."""
    lines = [ln.rstrip() for ln in (table_md or "").splitlines() if "|" in ln]
    if not lines:
        return None
    sep_idx = None
    for i, line in enumerate(lines):
        if _is_sep_row(line):
            sep_idx = i
            break
    if sep_idx is None or sep_idx == 0:
        return None

    header = _split_md_row(lines[sep_idx - 1])
    rows = [_split_md_row(ln) for ln in lines[sep_idx + 1 :] if ln.strip() and not _is_sep_row(ln)]
    width = max([len(header)] + [len(r) for r in rows] + [1])
    header = (header + [""] * width)[:width]
    rows = [(r + [""] * width)[:width] for r in rows]
    if not rows:
        return None
    return pd.DataFrame(rows, columns=header)


def extract_table_facts_from_markdown(
    table_md: str,
    *,
    doc_id: str,
    corpus_id: str,
    report_year: Optional[str],
    period_end_date: Optional[str],
    run_date_utc: str,
    page: int,
    table_id: str,
    table_type: Optional[str],
) -> list[dict[str, Any]]:
    df = markdown_table_to_dataframe(table_md)
    if df is None:
        return []
    return extract_table_facts_from_dataframe(
        df,
        doc_id=doc_id,
        corpus_id=corpus_id,
        report_year=report_year,
        period_end_date=period_end_date,
        run_date_utc=run_date_utc,
        page=page,
        table_id=table_id,
        table_type=table_type,
    )
