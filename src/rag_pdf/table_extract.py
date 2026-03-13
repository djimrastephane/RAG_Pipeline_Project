from __future__ import annotations

from typing import Optional, Union

from pathlib import Path
import re
from dataclasses import dataclass

import pandas as pd

try:
    import camelot  # type: ignore
except Exception:
    camelot = None
    print("WARNING: camelot-py not installed. Table extraction will use pdfplumber only.")

from rag_pdf.chunking import count_tokens
from rag_pdf.config import DEFAULT_CONFIG
from rag_pdf.ocr_table_fallback import accept_and_classify_ocr_table
from rag_pdf.schemas import build_page_list_struct, make_chunk_id_global
from rag_pdf.table_canonicalize import extract_table_facts_from_dataframe

CAMELOT_LATTICE_ACCURACY_THRESHOLD = DEFAULT_CONFIG.CAMELOT_LATTICE_ACCURACY_THRESHOLD
CAMELOT_LATTICE_WHITESPACE_MAX = DEFAULT_CONFIG.CAMELOT_LATTICE_WHITESPACE_MAX
CAMELOT_HYBRID_ACCURACY_THRESHOLD = DEFAULT_CONFIG.CAMELOT_HYBRID_ACCURACY_THRESHOLD
CAMELOT_HYBRID_WHITESPACE_MAX = DEFAULT_CONFIG.CAMELOT_HYBRID_WHITESPACE_MAX
CAMELOT_LINE_SCALE = DEFAULT_CONFIG.CAMELOT_LINE_SCALE
CAMELOT_RESOLUTION = DEFAULT_CONFIG.CAMELOT_RESOLUTION
CAMELOT_STREAM_ROW_TOL = DEFAULT_CONFIG.CAMELOT_STREAM_ROW_TOL
CAMELOT_STREAM_EDGE_TOL = DEFAULT_CONFIG.CAMELOT_STREAM_EDGE_TOL
TABLE_SUMMARY_MAX_ROWS = DEFAULT_CONFIG.TABLE_SUMMARY_MAX_ROWS
TABLE_MARKDOWN_MAX_ROWS = 30
TABLE_MARKDOWN_MAX_COLS = 10
TABLE_HEADER_INJECTION_MAX_ROWS = 80
TABLE_HEADER_INJECTION_MAX_FACTS = 300


@dataclass
class TableResult:
    page_no: int
    flavor: str
    dataframe: pd.DataFrame
    parsing_report: dict[str, Union[float, int, str]]
    logs: list[str]


def _to_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)  # type: ignore[arg-type]
    except Exception:
        return default


def _extract_parsing_report(table_obj: object) -> dict[str, Union[float, int, str]]:
    report = {}
    raw = getattr(table_obj, "parsing_report", None)
    if isinstance(raw, dict):
        report = dict(raw)
    return {
        "accuracy": _to_float(report.get("accuracy"), 0.0),
        "whitespace": _to_float(report.get("whitespace"), 100.0),
        "order": int(_to_float(report.get("order"), 0.0)),
        "page": str(report.get("page", "")),
    }


def _best_camelot_table(tables) -> Optional[object]:
    if not tables:
        return None
    candidates = []
    for t in tables:
        try:
            df = t.df
            rows, cols = df.shape
            if rows == 0 or cols == 0:
                continue
            rep = _extract_parsing_report(t)
            candidates.append((rows * cols, cols, rep.get("accuracy", 0.0), t))
        except Exception:
            continue
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: (x[0], x[1], x[2]))
    return candidates[0][3]


def _table_signature(df: pd.DataFrame) -> tuple[int, int, tuple[str, ...]]:
    if df is None or not isinstance(df, pd.DataFrame):
        return (0, 0, tuple())
    rows, cols = df.shape
    probes: list[str] = []
    for ridx in range(min(2, rows)):
        row = []
        for cidx in range(min(3, cols)):
            row.append(str(df.iat[ridx, cidx]).strip().lower())
        probes.append("|".join(row))
    return (rows, cols, tuple(probes))


def _clean_valid_tables_from_camelot_list(tables) -> list[pd.DataFrame]:
    out: list[pd.DataFrame] = []
    seen: set[tuple[int, int, tuple[str, ...]]] = set()
    if not tables:
        return out
    for t in tables:
        try:
            df = getattr(t, "df", None)
            if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
                continue
            cleaned = clean_table_dataframe(df, table_type=None)
            if cleaned is None or len(cleaned) == 0:
                continue
            sig = _table_signature(cleaned)
            if sig in seen:
                continue
            seen.add(sig)
            out.append(cleaned)
        except Exception:
            continue
    return out


def extract_tables_for_page(pdf_path: Path, page_no: int, config: Optional[dict] = None) -> list[TableResult]:
    """
    Extract table(s) for one page with Camelot passes:
    1) lattice with strict gate
    2) hybrid with moderate gate
    3) stream fallback (table existence only)
    """
    cfg = dict(config or {})
    lattice_acc = int(cfg.get("lattice_accuracy_threshold", CAMELOT_LATTICE_ACCURACY_THRESHOLD))
    lattice_ws = int(cfg.get("lattice_whitespace_max", CAMELOT_LATTICE_WHITESPACE_MAX))
    hybrid_acc = int(cfg.get("hybrid_accuracy_threshold", CAMELOT_HYBRID_ACCURACY_THRESHOLD))
    hybrid_ws = int(cfg.get("hybrid_whitespace_max", CAMELOT_HYBRID_WHITESPACE_MAX))
    line_scale = int(cfg.get("line_scale", CAMELOT_LINE_SCALE))
    resolution = int(cfg.get("resolution", CAMELOT_RESOLUTION))
    row_tol = int(cfg.get("row_tol", CAMELOT_STREAM_ROW_TOL))
    edge_tol = int(cfg.get("edge_tol", CAMELOT_STREAM_EDGE_TOL))
    return_all_tables = bool(cfg.get("return_all_tables", False))
    secondary_bottom_area = cfg.get("secondary_bottom_area")
    logs: list[str] = []

    if camelot is None:
        logs.append(f"page {page_no}: camelot unavailable")
        print(logs[-1])
        return []

    passes = [
        {
            "name": "lattice",
            "kwargs": {
                "flavor": "lattice",
                "strip_text": " .\n",
                "split_text": True,
                "copy_text": ["v"],
                "line_scale": line_scale,
                "resolution": resolution,
            },
            "gate": (lattice_acc, lattice_ws),
            "strict_gate": True,
        },
        {
            "name": "hybrid",
            "kwargs": {
                "flavor": "hybrid",
                "strip_text": " .\n",
                "split_text": True,
                "copy_text": ["v"],
                "line_scale": line_scale,
                "resolution": resolution,
                "row_tol": row_tol,
                "edge_tol": edge_tol,
            },
            "gate": (hybrid_acc, hybrid_ws),
            "strict_gate": True,
        },
        {
            "name": "stream",
            "kwargs": {
                "flavor": "stream",
                "strip_text": " .\n",
                "split_text": True,
                "row_tol": row_tol,
                "edge_tol": edge_tol,
            },
            "gate": None,
            "strict_gate": False,
        },
    ]

    for p in passes:
        pname = str(p["name"])
        try:
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=str(page_no),
                **p["kwargs"],  # type: ignore[arg-type]
            )
        except Exception as e:
            logs.append(f"page {page_no}: {pname} exception: {type(e).__name__}: {e}")
            print(logs[-1])
            continue

        if not tables or len(tables) == 0:
            logs.append(f"page {page_no}: {pname} failed (no tables)")
            print(logs[-1])
            continue

        best_table = _best_camelot_table(tables)
        if best_table is None:
            logs.append(f"page {page_no}: {pname} failed (empty/invalid parsed tables)")
            print(logs[-1])
            continue

        report = _extract_parsing_report(best_table)
        acc = float(report.get("accuracy", 0.0))
        ws = float(report.get("whitespace", 100.0))

        if p["strict_gate"]:
            gate_acc, gate_ws = p["gate"]  # type: ignore[misc]
            if acc < float(gate_acc):
                logs.append(
                    f"page {page_no}: {pname} failed (accuracy {acc:.2f} < {gate_acc})"
                )
                print(logs[-1])
                continue
            if ws > float(gate_ws):
                logs.append(
                    f"page {page_no}: {pname} failed (whitespace {ws:.2f} > {gate_ws})"
                )
                print(logs[-1])
                continue
        else:
            logs.append(
                f"page {page_no}: {pname} accepted (accuracy={acc:.2f}, whitespace={ws:.2f})"
            )
            print(logs[-1])

        if return_all_tables:
            cleaned_tables = _clean_valid_tables_from_camelot_list(tables)
            if not cleaned_tables:
                logs.append(f"page {page_no}: {pname} failed (no valid cleaned tables)")
                print(logs[-1])
                continue
            logs.append(
                f"page {page_no}: {pname} succeeded (accuracy={acc:.2f}, whitespace={ws:.2f}, tables={len(cleaned_tables)})"
            )
            print(logs[-1])
            out = [
                TableResult(
                    page_no=page_no,
                    flavor=pname,
                    dataframe=tbl,
                    parsing_report=report,
                    logs=logs.copy(),
                )
                for tbl in cleaned_tables
            ]
            if secondary_bottom_area and pname == "stream":
                try:
                    extra = camelot.read_pdf(
                        str(pdf_path),
                        pages=str(page_no),
                        flavor="stream",
                        strip_text=" .\n",
                        split_text=True,
                        row_tol=row_tol,
                        edge_tol=edge_tol,
                        table_areas=[str(secondary_bottom_area)],
                    )
                    extra_tables = _clean_valid_tables_from_camelot_list(extra)
                    seen = {_table_signature(r.dataframe) for r in out}
                    add_n = 0
                    for tbl in extra_tables:
                        sig = _table_signature(tbl)
                        if sig in seen:
                            continue
                        seen.add(sig)
                        out.append(
                            TableResult(
                                page_no=page_no,
                                flavor="stream_bottom",
                                dataframe=tbl,
                                parsing_report={"accuracy": 0.0, "whitespace": 0.0, "order": 0, "page": str(page_no)},
                                logs=logs.copy(),
                            )
                        )
                        add_n += 1
                    if add_n:
                        logs.append(f"page {page_no}: stream_bottom added {add_n} table(s)")
                        print(logs[-1])
                except Exception as e:
                    logs.append(f"page {page_no}: stream_bottom exception: {type(e).__name__}: {e}")
                    print(logs[-1])
            return out

        df = getattr(best_table, "df", None)
        if df is None or not isinstance(df, pd.DataFrame) or len(df) == 0:
            logs.append(f"page {page_no}: {pname} failed (best table has no dataframe)")
            print(logs[-1])
            continue
        cleaned = clean_table_dataframe(df, table_type=None)
        if cleaned is None or len(cleaned) == 0:
            logs.append(f"page {page_no}: {pname} failed (cleaned dataframe empty)")
            print(logs[-1])
            continue

        logs.append(
            f"page {page_no}: {pname} succeeded (accuracy={acc:.2f}, whitespace={ws:.2f})"
        )
        print(logs[-1])
        return [
            TableResult(
                page_no=page_no,
                flavor=pname,
                dataframe=cleaned,
                parsing_report=report,
                logs=logs.copy(),
            )
        ]

    logs.append(f"page {page_no}: all camelot passes failed")
    print(logs[-1])
    return []


def clean_table_dataframe(df: pd.DataFrame, table_type: Optional[str]) -> pd.DataFrame:
    """
    Clean extracted table dataframe.

    Operations:
    - Strip whitespace from all cells
    - Remove empty rows
    - Convert numeric-looking strings to numbers
    - Set first row as column headers if appropriate

    Args:
        df: Raw extracted dataframe
        table_type: Detected table type (for type-specific cleaning)

    Returns:
        Cleaned dataframe
    """
    if df is None or len(df) == 0:
        return df

    # Strip whitespace
    df = df.map(lambda x: str(x).strip() if pd.notna(x) else "")

    # Remove fully empty rows
    df = df.loc[~(df == "").all(axis=1)]

    # Try to detect and set header row
    # If first row has mostly text and subsequent rows have numbers, promote it
    if len(df) > 1:
        first_row = df.iloc[0]
        has_text = sum(
            1 for v in first_row
            if str(v).strip() and not str(v).replace(",", "").replace(".", "").replace("-", "").isdigit()
        )

        if has_text / len(first_row) > 0.5:  # More than half are text labels
            df.columns = [str(v) for v in first_row]
            df = df.iloc[1:].reset_index(drop=True)

    return df


def extract_table_camelot(pdf_path: Path, page_no: int) -> pd.Optional[DataFrame]:
    """
    Extract table using Camelot (lattice then stream methods).

    Args:
        pdf_path: Path to PDF file
        page_no: Page number (1-indexed)

    Returns:
        Extracted dataframe or None if extraction fails
    """
    if camelot is None:
        return None

    def _best_camelot_df(tables) -> pd.Optional[DataFrame]:
        if not tables:
            return None
        candidates = []
        for t in tables:
            try:
                df = t.df
                rows, cols = df.shape
                if rows == 0 or cols == 0:
                    continue
                candidates.append((rows * cols, cols, t.accuracy, df))
            except Exception:
                continue
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][3]

    try:
        # Try lattice method first (works for bordered tables)
        tables = camelot.read_pdf(
            str(pdf_path),
            pages=str(page_no),
            flavor="lattice",
            strip_text="\n",
        )

        if len(tables) > 0 and tables[0].accuracy >= CAMELOT_LATTICE_ACCURACY_THRESHOLD:
            best = _best_camelot_df(tables)
            if best is not None:
                return best
    except Exception:
        pass

    try:
        # Try stream method (works for borderless tables)
        tables = camelot.read_pdf(
            str(pdf_path),
            pages=str(page_no),
            flavor="stream",
        )

        if len(tables) > 0:
            best = _best_camelot_df(tables)
            if best is not None:
                return best
    except Exception:
        pass

    return None


def extract_table_pdfplumber(pdf_plumber, page_no: int) -> pd.Optional[DataFrame]:
    """
    Extract table using pdfplumber (fallback method).

    Args:
        pdf_plumber: Open pdfplumber PDF object
        page_no: Page number (1-indexed)

    Returns:
        Extracted dataframe or None if extraction fails
    """
    try:
        page_idx = page_no - 1
        page = pdf_plumber.pages[page_idx]
        tables = page.extract_tables()
        if not tables:
            return None
        candidates = []
        for table in tables:
            if table and len(table) > 1:
                df = pd.DataFrame(table[1:], columns=table[0])
                rows, cols = df.shape
                if rows == 0 or cols == 0:
                    continue
                candidates.append((rows * cols, cols, df))
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][2]
    except Exception:
        pass

    return None


def extract_table_cells(
    pdf_path: Path,
    pdf_plumber,
    page_no: int,
    table_type: Optional[str],
) -> pd.Optional[DataFrame]:
    """
    Multi-method table extraction with validation.

    Extraction cascade:
    1. Camelot lattice (bordered tables) - highest accuracy
    2. Camelot stream (borderless tables)
    3. pdfplumber (fallback)

    Args:
        pdf_path: Path to PDF file
        pdf_plumber: Open pdfplumber PDF object
        page_no: Page number (1-indexed)
        table_type: Detected table type (for type-specific cleaning)

    Returns:
        Cleaned dataframe or None if all methods fail
    """
    # Try configured Camelot 3-pass extraction.
    table_results = extract_tables_for_page(
        pdf_path=pdf_path,
        page_no=page_no,
        config={
            "lattice_accuracy_threshold": CAMELOT_LATTICE_ACCURACY_THRESHOLD,
            "lattice_whitespace_max": CAMELOT_LATTICE_WHITESPACE_MAX,
            "hybrid_accuracy_threshold": CAMELOT_HYBRID_ACCURACY_THRESHOLD,
            "hybrid_whitespace_max": CAMELOT_HYBRID_WHITESPACE_MAX,
            "line_scale": CAMELOT_LINE_SCALE,
            "resolution": CAMELOT_RESOLUTION,
            "row_tol": CAMELOT_STREAM_ROW_TOL,
            "edge_tol": CAMELOT_STREAM_EDGE_TOL,
        },
    )
    if table_results:
        df = table_results[0].dataframe
        if df is not None and len(df) > 0:
            return clean_table_dataframe(df, table_type)

    # Fallback to pdfplumber
    df = extract_table_pdfplumber(pdf_plumber, page_no)
    if df is not None and len(df) > 0:
        return clean_table_dataframe(df, table_type)

    return None


def generate_table_summary(
    df: pd.DataFrame,
    table_type: Optional[str],
    page_no: int,
) -> str:
    """
    Convert structured table to searchable text description.

    Creates a natural language summary that can be indexed for RAG search
    while preserving key information for answer generation.

    Args:
        df: Parsed table dataframe
        table_type: Classified table type
        page_no: Source page number

    Returns:
        Text summary string

    Example output:
        "Financial table on page 111 (Cash Flow - Non-cash adjustments).
         Contains 8 line items across 6 columns.
         Key items: Depreciation | 25,586 | 25,586 | 24,825;
         Impairments | 6,985 | 6,985 | 1,343..."
    """
    rows, cols = df.shape

    summary_parts = [
        f"Financial table on page {page_no}",
    ]

    if table_type:
        type_label = table_type.replace("_", " ").title()
        summary_parts.append(f"({type_label})")

    summary_parts.append(f"Contains {rows} line items across {cols} columns.")

    # Add column headers when they look meaningful (non-default).
    header_vals = [str(c).strip() for c in df.columns]
    if header_vals and any(h and not h.isdigit() for h in header_vals):
        summary_parts.append("Column headers: " + " | ".join(header_vals[:6]) + ".")

    # Add first-column labels to capture table semantics.
    first_col_labels = []
    if cols > 0:
        for i in range(min(TABLE_SUMMARY_MAX_ROWS, len(df))):
            v = str(df.iloc[i, 0]).strip()
            if v:
                first_col_labels.append(v)
    if first_col_labels:
        summary_parts.append("Row labels: " + "; ".join(first_col_labels[:8]) + ".")

    # Extract key rows (first few non-empty rows)
    key_items = []
    for i in range(min(TABLE_SUMMARY_MAX_ROWS, len(df))):
        row_values = [str(v).strip() for v in df.iloc[i] if str(v).strip()]
        if len(row_values) >= 2:  # At least a label and one value
            row_text = " | ".join(row_values[:4])  # Limit to first 4 columns
            key_items.append(row_text)

    if key_items:
        summary_parts.append("Key items: " + "; ".join(key_items) + ".")

    return " ".join(summary_parts)


def _sanitize_md_cell(val: str) -> str:
    cleaned = str(val or "").replace("\n", " ").replace("\r", " ").strip()
    return cleaned.replace("|", "\\|")


def table_to_markdown(
    df: pd.DataFrame,
    max_rows: int = TABLE_MARKDOWN_MAX_ROWS,
    max_cols: int = TABLE_MARKDOWN_MAX_COLS,
) -> str:
    """
    Render a dataframe as a pipe table to preserve structure for retrieval.
    """
    if df is None or len(df) == 0:
        return ""

    view = df.copy()
    if len(view.columns) > max_cols:
        view = view.iloc[:, :max_cols]
    if len(view) > max_rows:
        view = view.iloc[:max_rows]

    headers = []
    for i, col in enumerate(view.columns, start=1):
        name = str(col).strip()
        headers.append(name if name else f"col_{i}")

    header_row = "| " + " | ".join(_sanitize_md_cell(h) for h in headers) + " |"
    sep_row = "| " + " | ".join("---" for _ in headers) + " |"

    body_rows = []
    for _, row in view.iterrows():
        cells = [_sanitize_md_cell(v) for v in row.tolist()]
        body_rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header_row, sep_row] + body_rows)


def _split_md_row(line: str) -> list[str]:
    parts = [p.strip() for p in line.strip().strip("|").split("|")]
    return parts


def _is_sep_row(line: str) -> bool:
    if "|" not in line:
        return False
    cells = _split_md_row(line)
    if not cells:
        return False
    return all(
        c and all(ch in "-: " for ch in c) and "-" in c
        for c in cells
    )


def _is_numeric_like(text: str) -> bool:
    s = str(text or "").strip().lower()
    if not s or s in {"n/a", "na", "-", "—"}:
        return False
    return bool(re.search(r"\d", s))


def _is_unit_fragment(text: str) -> bool:
    s = str(text or "").strip()
    if not s or " " in s:
        return False
    if any(sym in s for sym in ("£", "$", "€", "%")):
        return True
    return bool(re.fullmatch(r"[A-Za-z/().,-]{1,8}", s))


def _normalize_rows(rows: list[list[str]], width: int) -> list[list[str]]:
    out = []
    for r in rows:
        if len(r) < width:
            r = r + [""] * (width - len(r))
        elif len(r) > width:
            r = r[:width]
        out.append(r)
    return out


def _merge_headers(header_rows: list[list[str]], width: int) -> list[str]:
    merged = []
    for col in range(width):
        fragments = []
        units = []
        for row in header_rows:
            if col >= len(row):
                continue
            frag = str(row[col]).strip()
            if not frag:
                continue
            if _is_unit_fragment(frag):
                units.append(frag)
            else:
                fragments.append(frag)
        base = " ".join(fragments).strip()
        if units:
            unit_str = " ".join(units).strip()
            base = f"{base} {unit_str}".strip()
        merged.append(base)
    for i, name in enumerate(merged):
        if not name:
            merged[i] = f"col_{i}"
    merged[0] = "Row label"
    return merged


def enrich_table_markdown(table_md: str) -> str:
    """
    Append compact header and row-map signals to a markdown table string.
    """
    lines = [ln.rstrip() for ln in (table_md or "").splitlines()]
    if not lines:
        return table_md

    table_lines = [ln for ln in lines if "|" in ln]
    if not table_lines:
        return table_md

    sep_idx = None
    for i, line in enumerate(table_lines):
        if _is_sep_row(line):
            sep_idx = i
            break
    if sep_idx is None or sep_idx == 0:
        return table_md

    header_row = _split_md_row(table_lines[sep_idx - 1])
    body_rows = [_split_md_row(ln) for ln in table_lines[sep_idx + 1 :]]
    width = max([len(header_row)] + [len(r) for r in body_rows] + [1])
    header_row = _normalize_rows([header_row], width)[0]
    body_rows = _normalize_rows(body_rows, width)

    header_row_numeric = all(
        (not c) or _is_numeric_like(c) for c in header_row
    )

    body_start = None
    for idx, row in enumerate(body_rows):
        first = str(row[0]).strip()
        numeric_cells = sum(1 for c in row if _is_numeric_like(c))
        if first and numeric_cells >= 2:
            body_start = idx
            break
    if body_start is None:
        body_start = len(body_rows)
    if header_row_numeric:
        body_start = max(body_start, 1)

    raw_header_rows = [header_row] + body_rows[:body_start]
    header_rows = []
    for row in raw_header_rows:
        row0 = str(row[0]).strip()
        non_empty = sum(1 for c in row if str(c).strip())
        if row0 in {"Remuneration of:", "Executive Members", "Non Executive Members"}:
            break
        if len(row0) > 60 and non_empty == 1:
            continue
        header_rows.append(row)
    data_rows = body_rows[body_start:]

    headers = _merge_headers(header_rows, width)
    header_line = "Column headers: " + " | ".join(headers)

    alias_headers = ["col_0"] * width
    for j, full in enumerate(headers):
        low = full.lower()
        if j == 0:
            alias_headers[j] = "Row label"
        elif "salary" in low:
            alias_headers[j] = "Salary"
        elif "bonus" in low:
            alias_headers[j] = "Bonus"
        elif "benefits in kind" in low:
            alias_headers[j] = "Benefits"
        elif "sub total" in low or "subtotal" in low:
            alias_headers[j] = "Subtotal"
        elif "pension" in low:
            alias_headers[j] = "Pension"
        elif "total" in low and "remuneration" in low:
            alias_headers[j] = "Total remuneration"
        else:
            alias_headers[j] = f"col_{j}"

    row_candidates = []
    for idx, row in enumerate(data_rows):
        row_label = str(row[0]).strip()
        other_cells = row[1:]
        if not row_label:
            continue
        if not any(str(c).strip() for c in other_cells):
            continue
        numeric_cells = [j for j, c in enumerate(row) if _is_numeric_like(c)]
        if len(numeric_cells) < 2:
            continue
        row_candidates.append((idx, len(numeric_cells), row_label, row))

    top_rows = sorted(row_candidates, key=lambda x: (-x[1], x[0]))[:25]
    top_indices = {idx for idx, _, _, _ in top_rows}
    row_map_lines = []
    for idx, _, row_label, row in row_candidates:
        if idx not in top_indices:
            continue
        pairs = []
        for j, cell in enumerate(row[1:], start=1):
            cell_str = str(cell).strip()
            if not cell_str or not _is_numeric_like(cell_str):
                continue
            pairs.append(f"{alias_headers[j]}={cell_str}")
        if not pairs:
            continue
        row_map_lines.append(f"- {row_label} -> " + " ; ".join(pairs))
        if len(row_map_lines) >= 25:
            break

    if not row_map_lines:
        return table_md

    appended = "\n".join(
        [
            "",
            header_line,
            "Row map:",
            *row_map_lines,
        ]
    )
    return table_md.rstrip() + appended + "\n"


def build_header_injected_facts(table_md: str) -> str:
    """
    Build explicit cell-level facts with header paths for embedding.

    Output format:
      - <row label> > <column header path> : <value>
    """
    lines = [ln.rstrip() for ln in (table_md or "").splitlines()]
    if not lines:
        return ""

    table_lines = [ln for ln in lines if "|" in ln]
    if not table_lines:
        return ""

    sep_idx = None
    for i, line in enumerate(table_lines):
        if _is_sep_row(line):
            sep_idx = i
            break
    if sep_idx is None or sep_idx == 0:
        return ""

    header_row = _split_md_row(table_lines[sep_idx - 1])
    body_rows = [_split_md_row(ln) for ln in table_lines[sep_idx + 1 :]]
    width = max([len(header_row)] + [len(r) for r in body_rows] + [1])
    header_row = _normalize_rows([header_row], width)[0]
    body_rows = _normalize_rows(body_rows, width)

    header_row_numeric = all((not c) or _is_numeric_like(c) for c in header_row)

    body_start = None
    for idx, row in enumerate(body_rows):
        first = str(row[0]).strip()
        numeric_cells = sum(1 for c in row if _is_numeric_like(c))
        if first and numeric_cells >= 1:
            body_start = idx
            break
    if body_start is None:
        body_start = len(body_rows)
    if header_row_numeric:
        body_start = max(body_start, 1)

    raw_header_rows = [header_row] + body_rows[:body_start]
    header_rows = []
    for row in raw_header_rows:
        row0 = str(row[0]).strip()
        non_empty = sum(1 for c in row if str(c).strip())
        if row0 in {"Remuneration of:", "Executive Members", "Non Executive Members"}:
            break
        if len(row0) > 60 and non_empty == 1:
            continue
        header_rows.append(row)
    data_rows = body_rows[body_start : body_start + TABLE_HEADER_INJECTION_MAX_ROWS]

    headers = _merge_headers(header_rows, width)
    facts: list[str] = []
    for row in data_rows:
        row_label = str(row[0]).strip()
        if not row_label:
            continue
        for j, cell in enumerate(row[1:], start=1):
            val = str(cell).strip()
            if not val:
                continue
            col_path = str(headers[j]).strip() if j < len(headers) else f"col_{j}"
            if not col_path:
                col_path = f"col_{j}"
            facts.append(f"- {row_label} > {col_path} : {val}")
            if len(facts) >= TABLE_HEADER_INJECTION_MAX_FACTS:
                break
        if len(facts) >= TABLE_HEADER_INJECTION_MAX_FACTS:
            break

    return "\n".join(facts)


def _self_test_staff_costs() -> None:
    """
    Print merged headers and sample row-map lines for a staff costs table.
    """
    sample_path = (
        Path("data_processed")
        / "Grampian-2024-2025"
        / "tables_markdown"
        / "table_p0083_staff_costs.md"
    )
    if not sample_path.exists():
        print(f"Missing sample: {sample_path}")
        return
    raw = sample_path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    table_lines = [ln for ln in lines if ln.strip().startswith("|")]
    table_md = "\n".join(table_lines)
    enriched = enrich_table_markdown(table_md)
    out_lines = enriched.splitlines()
    header_line = next((ln for ln in out_lines if ln.startswith("Column headers: ")), "")
    row_map_lines = [ln for ln in out_lines if ln.startswith("- ")]
    print(header_line)
    for ln in row_map_lines[:3]:
        print(ln)


def _norm_ws(s: object) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _table_caption_from_summary(table_summary: str) -> str:
    line = _norm_ws(str(table_summary or "").splitlines()[0] if str(table_summary or "").splitlines() else "")
    if not line:
        return "Unknown"
    return line[:120]


def _table_headers_from_df(df: pd.DataFrame) -> list[str]:
    headers = [_norm_ws(h) for h in list(df.columns)]
    if any(headers):
        return [h if h else "-" for h in headers]
    return [f"col_{i+1}" for i in range(len(df.columns))]


def _table_rows_from_df(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for _, row in df.iterrows():
        vals = [_norm_ws(v) for v in row.tolist()]
        vals = [v if v else "-" for v in vals]
        out.append(" | ".join(vals))
    return out


def _pack_lines_by_token_budget(
    lines: list[str],
    *,
    prefix_lines: list[str],
    chunk_size_tokens: int,
    enc,
) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    prefix_text = "\n".join([ln for ln in prefix_lines if _norm_ws(ln)])
    prefix_tokens = count_tokens(prefix_text, enc) if prefix_text else 0
    budget = max(40, int(chunk_size_tokens))

    for ln in lines:
        candidate = "\n".join(cur + [ln]).strip()
        candidate_tokens = count_tokens(candidate, enc) + prefix_tokens
        if cur and candidate_tokens > budget:
            body = "\n".join(cur).strip()
            if body:
                chunk = f"{prefix_text}\n{body}".strip() if prefix_text else body
                chunks.append(chunk)
            cur = [ln]
        else:
            cur.append(ln)

    if cur:
        body = "\n".join(cur).strip()
        if body:
            chunk = f"{prefix_text}\n{body}".strip() if prefix_text else body
            chunks.append(chunk)
    return chunks


def _build_table_chunk_texts(
    *,
    strategy: str,
    page_no: int,
    table_summary: str,
    raw_table: pd.DataFrame,
    header_injected_facts: str,
    table_markdown: str,
    chunk_size_tokens: int,
    enc,
) -> list[str]:
    # Baseline: preserve current behavior exactly.
    if strategy == "baseline":
        parts = [table_summary]
        if header_injected_facts:
            parts.append("Table (header-injected facts):")
            parts.append(header_injected_facts)
        if table_markdown:
            parts.append("Table (markdown):")
            parts.append(table_markdown)
        return ["\n\n".join(parts).strip()]

    caption = _table_caption_from_summary(table_summary)
    table_prefix = f"TABLE | page={page_no} | {caption}"
    headers = _table_headers_from_df(raw_table)
    header_line = " | ".join(headers)
    row_lines = _table_rows_from_df(raw_table)

    if strategy == "row_preserving":
        prefix_lines = [table_prefix, f"COLUMNS: {header_line}"]
        return _pack_lines_by_token_budget(
            row_lines,
            prefix_lines=prefix_lines,
            chunk_size_tokens=chunk_size_tokens,
            enc=enc,
        )

    # two_stage
    first_rows = row_lines[: min(5, len(row_lines))]
    units_line = ""
    year_line = ""
    for ln in first_rows:
        if not units_line and any(tok in ln for tok in ("£", "%", "000", "million", "m ")):
            units_line = ln
        if not year_line and re.search(r"\b(19|20)\d{2}(?:/\d{2,4})?\b", ln):
            year_line = ln
    header_chunk_lines = [table_prefix, f"COLUMNS: {header_line}"]
    if units_line:
        header_chunk_lines.append(f"UNITS: {units_line}")
    if year_line:
        header_chunk_lines.append(f"YEAR_LABELS: {year_line}")
    header_chunk = "\n".join(header_chunk_lines).strip()

    body_prefix = [table_prefix, f"COLUMNS: {header_line}"]
    body_chunks = _pack_lines_by_token_budget(
        row_lines,
        prefix_lines=body_prefix,
        chunk_size_tokens=chunk_size_tokens,
        enc=enc,
    )
    return [header_chunk] + body_chunks


def process_table_pages(
    table_pages: list[dict],
    pdf_path: Path,
    pdf_plumber,
    doc_id: str,
    corpus_id: str,
    report_year: Optional[str],
    period_end_date: Optional[str],
    report_year_source: Optional[str],
    run_date_utc: str,
    enc,
    chunk_size_tokens: int = 224,
    table_chunking_strategy: str = "baseline",
    return_all_tables: bool = False,
    enable_secondary_bottom_pass: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """
    Extract tables and create dual representation.

    For each table page:
    1. Extract structured cells → tables_structured.parquet
    2. Generate text summary → chunks.parquet (for RAG search)

    Args:
        table_pages: List of classified table pages
        pdf_path: Path to source PDF
        pdf_plumber: Open pdfplumber PDF object
        doc_id: Document identifier
        corpus_id: Corpus identifier
        report_year: Report year range
        period_end_date: Period end date (ISO format)
        report_year_source: Source of report year ("pdf_cover" or "filename")
        run_date_utc: Processing timestamp
        enc: Tiktoken encoder

    Returns:
        (text_chunks_df, structured_tables_df, table_facts_df, rejected_ocr_table_pages)
        - text_chunks_df: Table summaries for RAG (same schema as text chunks)
        - structured_tables_df: Parsed cells with metadata
        - table_facts_df: Canonical row/column/value facts for robust table QA
        - rejected_ocr_table_pages: pages that should be re-chunked as normal text
    """
    text_chunks = []
    structured_tables = []
    table_facts = []
    rejected_ocr_table_pages: list[dict] = []

    for tpage in table_pages:
        page_no = tpage["page"]
        table_type = tpage["table_type"]

        table_results = extract_tables_for_page(
            pdf_path=pdf_path,
            page_no=page_no,
            config={
                "lattice_accuracy_threshold": CAMELOT_LATTICE_ACCURACY_THRESHOLD,
                "lattice_whitespace_max": CAMELOT_LATTICE_WHITESPACE_MAX,
                "hybrid_accuracy_threshold": CAMELOT_HYBRID_ACCURACY_THRESHOLD,
                "hybrid_whitespace_max": CAMELOT_HYBRID_WHITESPACE_MAX,
                "line_scale": CAMELOT_LINE_SCALE,
                "resolution": CAMELOT_RESOLUTION,
                "row_tol": CAMELOT_STREAM_ROW_TOL,
                "edge_tol": CAMELOT_STREAM_EDGE_TOL,
                "return_all_tables": bool(return_all_tables),
                "secondary_bottom_area": (
                    f"0,0,{float(tpage.get('page_width', 0.0) or 0.0):.1f},{max(1.0, float(tpage.get('page_height', 0.0) or 0.0) * 0.45):.1f}"
                    if enable_secondary_bottom_pass and float(tpage.get("page_width", 0.0) or 0.0) > 0.0 and float(tpage.get("page_height", 0.0) or 0.0) > 0.0
                    else None
                ),
            },
        )

        camelot_tables_found = len(table_results)
        page_extractor = str(tpage.get("extractor", "") or "").lower()
        page_rotation = int(tpage.get("rotation", 0) or 0)
        page_text = str(tpage.get("text", "") or "")
        page_raw_text = str(tpage.get("raw_text", "") or "")

        # OCR-table fallback: scanned/ocr pages where Camelot found no tables.
        if (
            camelot_tables_found == 0
            and bool(tpage.get("is_table", False))
            and page_extractor == "ocr"
        ):
            accept, ocr_table_type, ocr_chunk_text, debug = accept_and_classify_ocr_table(
                page_raw_text or page_text,
                page_no=page_no,
                rotation_deg=page_rotation,
            )
            print(
                f"[OCR_TABLE_FALLBACK] page={page_no} rotation={page_rotation} "
                f"accept={accept} table_type={ocr_table_type} "
                f"digit_ratio={debug.get('digit_ratio', 0.0):.3f} "
                f"currency_hits={debug.get('currency_hits', 0)} "
                f"num_lines_with_2plus_nums={debug.get('num_lines_with_2plus_nums', 0)} "
                f"corrupted_flag={debug.get('corrupted_flag', False)} "
                f"matched_triggers={debug.get('matched_triggers', [])}"
            )

            if accept:
                chunk_id_local = f"table_ocr_p{page_no:04d}"
                pages = [page_no]
                page_list_struct = build_page_list_struct(pages)
                text_chunks.append({
                    "doc_id": doc_id,
                    "corpus_id": corpus_id,
                    "report_year": report_year,
                    "report_year_source": report_year_source,
                    "period_end_date": period_end_date,
                    "run_date_utc": run_date_utc,
                    "chunk_id": chunk_id_local,
                    "chunk_id_global": make_chunk_id_global(doc_id, chunk_id_local),
                    "part": "Unknown",
                    "section_title": "Financial Tables",
                    "subsection_title": None,
                    "page_start": page_no,
                    "page_end": page_no,
                    "pages": pages,
                    "page_list": page_list_struct,
                    "chunk_text": ocr_chunk_text,
                    "chunk_tokens": count_tokens(ocr_chunk_text, enc),
                    "word_count": len(ocr_chunk_text.split()),
                    "is_table_like": True,
                    "many_numbers": True,
                    "is_table": True,
                    "table_type": ocr_table_type,
                    "table_ref": None,
                    "table_source": "ocr_fallback",
                    "parsing_report_accuracy": None,
                    "parsing_report_whitespace": None,
                    "ocr_fallback_digit_ratio": float(debug.get("digit_ratio", 0.0)),
                    "ocr_fallback_currency_hits": int(debug.get("currency_hits", 0)),
                    "ocr_fallback_num_lines_with_2plus_nums": int(debug.get("num_lines_with_2plus_nums", 0)),
                    "ocr_fallback_corrupted_flag": bool(debug.get("corrupted_flag", False)),
                    "ocr_fallback_matched_keywords": "|".join(debug.get("matched_keywords", [])),
                })
                continue

            # Rejected OCR-table: send back to normal text chunking path.
            rejected_ocr_table_pages.append({
                "page": page_no,
                "text": page_text,
            })
            continue

        if not table_results:
            # Final fallback: pdfplumber (keeps pipeline resilient if Camelot fails).
            df_fallback = extract_table_pdfplumber(pdf_plumber, page_no)
            if df_fallback is not None and len(df_fallback) > 0:
                table_results = [
                    TableResult(
                        page_no=page_no,
                        flavor="pdfplumber",
                        dataframe=clean_table_dataframe(df_fallback, table_type),
                        parsing_report={"accuracy": 0.0, "whitespace": 0.0, "order": 0, "page": str(page_no)},
                        logs=[f"page {page_no}: pdfplumber fallback succeeded"],
                    )
                ]
                print(f"page {page_no}: camelot all-pass failed; pdfplumber fallback used")

        for idx, tresult in enumerate(table_results):
            raw_table = tresult.dataframe
            if raw_table is None or len(raw_table) == 0:
                continue

            table_summary = generate_table_summary(raw_table, table_type, page_no)
            table_markdown = enrich_table_markdown(table_to_markdown(raw_table))
            header_injected_facts = build_header_injected_facts(table_markdown)

            # Store structured version
            table_id = f"table_p{page_no:04d}" if len(table_results) == 1 else f"table_p{page_no:04d}_{idx:02d}"

            structured_tables.append({
                "doc_id": doc_id,
                "corpus_id": corpus_id,
                "report_year": report_year,
                "period_end_date": period_end_date,
                "run_date_utc": run_date_utc,
                "page": page_no,
                "page_no": page_no,
                "table_id": table_id,
                "table_type": table_type or "unknown",
                "rows": len(raw_table),
                "cols": len(raw_table.columns),
                "extraction_method": "camelot" if tresult.flavor in {"lattice", "hybrid", "stream", "stream_bottom"} else "pdfplumber",
                "flavor": tresult.flavor,
                "parsing_report_accuracy": float(tresult.parsing_report.get("accuracy", 0.0)),
                "parsing_report_whitespace": float(tresult.parsing_report.get("whitespace", 0.0)),
                "parsing_report_order": int(_to_float(tresult.parsing_report.get("order", 0.0))),
                "parsing_report_page": str(tresult.parsing_report.get("page", "")),
                "table_summary": table_summary,
                "table_markdown": table_markdown,
                "table_header_injection": header_injected_facts,
            })
            table_facts.extend(
                extract_table_facts_from_dataframe(
                    raw_table,
                    doc_id=doc_id,
                    corpus_id=corpus_id,
                    report_year=report_year,
                    period_end_date=period_end_date,
                    run_date_utc=run_date_utc,
                    page=page_no,
                    table_id=table_id,
                    table_type=table_type,
                )
            )

            chunk_texts = _build_table_chunk_texts(
                strategy=str(table_chunking_strategy or "baseline"),
                page_no=page_no,
                table_summary=table_summary,
                raw_table=raw_table,
                header_injected_facts=header_injected_facts,
                table_markdown=table_markdown,
                chunk_size_tokens=int(chunk_size_tokens),
                enc=enc,
            )
            for cidx, summary in enumerate(chunk_texts):
                base_id = f"table_p{page_no:04d}" if len(table_results) == 1 else f"table_p{page_no:04d}_{idx:02d}"
                chunk_id_local = base_id if len(chunk_texts) == 1 else f"{base_id}_s{cidx:02d}"
                pages = [page_no]
                page_list_struct = build_page_list_struct(pages)
                text_chunks.append({
                    "doc_id": doc_id,
                    "corpus_id": corpus_id,
                    "report_year": report_year,
                    "report_year_source": report_year_source,
                    "period_end_date": period_end_date,
                    "run_date_utc": run_date_utc,
                    "chunk_id": chunk_id_local,
                    "chunk_id_global": make_chunk_id_global(doc_id, chunk_id_local),
                    "part": "Unknown",  # Tables don't have part classification
                    "section_title": "Financial Tables",
                    "subsection_title": None,
                    "page_start": page_no,
                    "page_end": page_no,
                    "pages": pages,
                    "page_list": page_list_struct,
                    "chunk_text": summary,
                    "chunk_tokens": count_tokens(summary, enc),
                    "word_count": len(summary.split()),
                    "is_table_like": True,
                    "many_numbers": True,
                    "is_table": True,
                    "table_type": table_type,
                    "table_ref": table_id,
                })

    return (
        pd.DataFrame(text_chunks),
        pd.DataFrame(structured_tables),
        pd.DataFrame(table_facts),
        rejected_ocr_table_pages,
    )
