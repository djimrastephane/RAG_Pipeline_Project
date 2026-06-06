"""
backfill_front_section_tables.py

One-time correction for the front-section table extraction gap.

Background
----------
PyMuPDF reads NHS performance/financial-targets tables (pages 16-28 of each
annual report) as a single inline text stream with no line breaks.  The
existing heuristics (digit-ratio, double-space count) require multiple lines
and therefore never fire, so those pages are indexed as prose and get
is_table=False in chunks.parquet.

After fixing table_detect.is_nhs_financial_performance_table() this script:
1. Identifies affected pages (new heuristic fires; is_table still False in
   the stored pages.parquet)
2. Runs Camelot lattice extraction from the source PDF
3. Creates proper table chunk records and appends them to chunks.parquet
4. Rebuilds the FAISS index via build_index.py

Usage
-----
    python scripts/backfill_front_section_tables.py [--dry-run]

Nothing is modified when --dry-run is passed.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_pdf.table_detect import is_nhs_financial_performance_table
from rag_pdf.table_markdown import table_to_markdown, build_header_injected_facts

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None

def _count_tokens(text: str) -> int:
    if _ENC is None:
        return len(text.split())
    from rag_pdf.chunking import count_tokens
    return count_tokens(text, _ENC)

try:
    import camelot  # type: ignore
except ImportError:
    print("ERROR: camelot-py is required.  pip install camelot-py[cv]")
    sys.exit(1)

_CONDA_PYTHON = Path("/opt/anaconda3/envs/rag-pipeline/bin/python")
PYTHON = str(_CONDA_PYTHON) if _CONDA_PYTHON.exists() else sys.executable

DATA_BASE = REPO / "data_processed"
PDF_BASE = REPO / "Data" / "Annual Accounts NHS Grampian" / "Preliminary_Test"

DOC_PDF_MAP: dict[str, Path] = {
    "Grampian-2020-2021": PDF_BASE / "Grampian-2020-2021.pdf",
    "Grampian-2021-2022": PDF_BASE / "Grampian-2021-2022.pdf",
    "Grampian-2022-2023": PDF_BASE / "Grampian-2022-2023.pdf",
    "Grampian-2023-2024": PDF_BASE / "Grampian-2023-2024.pdf",
    "Grampian-2024-2025": PDF_BASE / "Grampian-2024-2025.pdf",
}

# Minimum Camelot accuracy to accept a table (relaxed for stream mode)
LATTICE_ACC_MIN = 80
STREAM_ACC_MIN  = 60
MIN_ROWS        = 3
MIN_COLS        = 2


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _affected_pages(doc_id: str) -> list[int]:
    """Return page numbers where new heuristic fires but is_table was False."""
    pages_df = pd.read_parquet(DATA_BASE / doc_id / "pages.parquet")
    hit = []
    for _, row in pages_df.iterrows():
        text = str(row.get("clean_text") or "")
        if not bool(row.get("is_table", False)) and is_nhs_financial_performance_table(text):
            hit.append(int(row["page"]))
    return sorted(hit)


def _already_backfilled(doc_id: str, page: int) -> bool:
    """True if chunks.parquet already has an is_table=True chunk on this page."""
    ck = pd.read_parquet(DATA_BASE / doc_id / "chunks.parquet")
    tbl = ck[ck["is_table"] == True]
    for _, r in tbl.iterrows():
        ps, pe = int(r.get("page_start", -1)), int(r.get("page_end", -1))
        if ps <= page <= pe:
            return True
    return False


# ---------------------------------------------------------------------------
# Camelot extraction
# ---------------------------------------------------------------------------

def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize whitespace in all cells."""
    return df.map(lambda v: re.sub(r"\s+", " ", str(v or "")).strip())


def _extract_table_camelot(pdf_path: Path, page: int) -> pd.DataFrame | None:
    """Try lattice then stream; return first acceptable DataFrame or None."""
    for flavor, acc_min in [("lattice", LATTICE_ACC_MIN), ("stream", STREAM_ACC_MIN)]:
        try:
            tables = camelot.read_pdf(
                str(pdf_path),
                pages=str(page),
                flavor=flavor,
                strip_text=" .\n",
            )
        except Exception as exc:
            print(f"    camelot {flavor} error: {exc}")
            continue

        for t in tables:
            acc = float(t.parsing_report.get("accuracy", 0))
            df = _clean_df(t.df)
            if acc >= acc_min and df.shape[0] >= MIN_ROWS and df.shape[1] >= MIN_COLS:
                print(f"    camelot {flavor}: acc={acc:.1f} shape={df.shape} ✓")
                return df

        print(f"    camelot {flavor}: no table met quality threshold")

    return None


# ---------------------------------------------------------------------------
# Chunk text generation — prose-first for better MiniLM embedding
# ---------------------------------------------------------------------------

# Statutory financial targets row labels (order matters for parsing)
_STAT_ROWS = re.compile(
    r'(Core Revenue Resource Limit'
    r'|Non-Core Revenue Resource Limit'
    r'|Total Revenue Resource Limit'
    r'|Core Capital Resource Limit'
    r'|Non-Core Capital Resource Limit'
    r'|Total Capital Resource Limit'
    r'|Cash Requirement'
    r'|Total(?:\s+(?:Revenue|Capital)\s+Resource\s+Limit)?)'
    r'\s+([\d,]+|-|\(\d[\d,]*\))'
    r'\s+([\d,]+|-|\(\d[\d,]*\))'
    r'\s+([\d,]+|-|\(\d[\d,]*\))',
    re.IGNORECASE,
)

_CONSOL_HEADER = re.compile(
    r'Consolidated\s+Account|Income\s+Statement|Balance\s+Sheet|Revenue\s+and\s+Capital',
    re.IGNORECASE,
)


def _parse_statutory_table_text(page_text: str) -> str | None:
    """Extract the statutory financial targets table from inline page text."""
    start = page_text.find("Statutory Financial Target")
    if start == -1:
        return None
    snippet = page_text[start:]
    matches = _STAT_ROWS.findall(snippet)
    if not matches:
        return None
    rows = []
    for label, limit, actual, variance in matches:
        label = re.sub(r"\s+", " ", label).strip()
        rows.append(f"{label}: Limit={limit}, Actual={variance}, Variance={actual}")
    return "Statutory Financial Targets (£000's):\n" + "\n".join(rows)


def _make_chunk_text(df: pd.DataFrame, page: int, table_type: str, page_text: str = "") -> str:
    """Build chunk text from inline page text first (better embedding quality),
    falling back to a CamelCase-fixed markdown table if parsing fails."""
    header = f"TABLE | page={page} | {table_type.replace('_', ' ').title()} | NHS Grampian"

    # Prefer prose extraction for statutory financial targets — Camelot strips
    # spaces from cell text producing concatenated words like 'CoreRevenueResourceLimit'
    # which embeds poorly with MiniLM.
    if table_type == "financial_targets" and page_text:
        prose = _parse_statutory_table_text(page_text)
        if prose:
            return f"{header}\n\n{prose}"

    # Fallback: fix CamelCase concatenation from Camelot, then render markdown
    def _fix_cell(v: str) -> str:
        v = re.sub(r"([a-z])([A-Z])", r"\1 \2", v)   # insert space at CamelCase boundary
        v = re.sub(r"([A-Za-z])(£)", r"\1 \2", v)     # space before £
        v = re.sub(r"([A-Za-z])(\d)", r"\1 \2", v)    # space before digit
        return v

    fixed_df = df.map(_fix_cell)
    md = table_to_markdown(fixed_df)
    facts = build_header_injected_facts(md)
    parts = [header]
    if facts:
        parts.append(facts)
    if md:
        parts.append(md)
    return "\n\n".join(parts)


def _infer_table_type(text: str) -> str:
    t = text.lower()
    if "resource limit" in t or "cash requirement" in t:
        return "financial_targets"
    if "consolidated" in t or "income" in t:
        return "income_statement"
    return "performance_table"


# ---------------------------------------------------------------------------
# Chunk record builder
# ---------------------------------------------------------------------------

def _next_chunk_id(chunks_df: pd.DataFrame, page: int, suffix: int = 0) -> str:
    candidate = f"table_p{page:04d}" if suffix == 0 else f"table_p{page:04d}_{suffix}"
    existing = set(chunks_df["chunk_id"].dropna().astype(str))
    while candidate in existing:
        suffix += 1
        candidate = f"table_p{page:04d}_{suffix}"
    return candidate


def _build_chunk_record(
    *,
    doc_id: str,
    page: int,
    df: pd.DataFrame,
    table_type: str,
    chunk_id: str,
    existing_chunks: pd.DataFrame,
    page_text: str = "",
) -> dict:
    chunk_text = _make_chunk_text(df, page, table_type, page_text=page_text)
    tokens = _count_tokens(chunk_text)

    # Copy scalar metadata from an existing chunk in the same doc
    ref = existing_chunks.iloc[0]

    return {
        "doc_id": doc_id,
        "corpus_id": str(ref.get("corpus_id") or doc_id),
        "report_year": str(ref.get("report_year") or ""),
        "report_year_source": str(ref.get("report_year_source") or "filename"),
        "period_end_date": ref.get("period_end_date"),
        "run_date_utc": datetime.now(timezone.utc).isoformat(),
        "chunk_id": chunk_id,
        "chunk_id_global": f"{doc_id}:{chunk_id}",
        "part": "Performance Report",
        "section_title": "Financial Performance and Position",
        "page_start": page,
        "page_end": page,
        "pages": np.array([page]),
        "page_list": np.array([{"element": page}]),
        "chunk_text": chunk_text,
        "chunk_tokens": tokens,
        "word_count": len(chunk_text.split()),
        "is_table_like": True,
        "many_numbers": True,
        "is_table": True,
        "table_type": table_type,
        "table_ref": chunk_id,
        "subsection_title": "Statutory Financial Targets",
        "table_chunk_kind": "full_table",
        "row_start_idx": None,
        "row_end_idx": None,
        "table_word_budget_target": None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(dry_run: bool = False) -> None:
    total_added = 0

    for doc_id, pdf_path in DOC_PDF_MAP.items():
        print(f"\n{'='*60}")
        print(f"Doc: {doc_id}")

        if not pdf_path.exists():
            print(f"  PDF not found: {pdf_path} — skipping")
            continue

        pages = _affected_pages(doc_id)
        if not pages:
            print("  No affected pages detected")
            continue

        print(f"  Affected pages: {pages}")

        chunks_path = DATA_BASE / doc_id / "chunks.parquet"
        chunks_df = pd.read_parquet(chunks_path)
        new_records: list[dict] = []

        for page in pages:
            if _already_backfilled(doc_id, page):
                print(f"  p{page}: already has is_table chunk — skipping")
                continue

            print(f"  p{page}: running Camelot …")
            df = _extract_table_camelot(pdf_path, page)

            if df is None:
                print(f"  p{page}: Camelot failed — skipping")
                continue

            # Infer table type from page text
            pages_df = pd.read_parquet(DATA_BASE / doc_id / "pages.parquet")
            page_text = str(pages_df[pages_df["page"] == page].iloc[0]["clean_text"])
            table_type = _infer_table_type(page_text)

            chunk_id = _next_chunk_id(chunks_df, page)
            record = _build_chunk_record(
                doc_id=doc_id,
                page=page,
                df=df,
                table_type=table_type,
                chunk_id=chunk_id,
                existing_chunks=chunks_df,
                page_text=page_text,
            )
            new_records.append(record)
            print(f"  p{page}: built chunk {chunk_id} ({record['chunk_tokens']} tokens, type={table_type})")

        if not new_records:
            print(f"  No new chunks to add for {doc_id}")
            continue

        if dry_run:
            print(f"  [dry-run] Would append {len(new_records)} chunk(s) to {chunks_path}")
            for r in new_records:
                print(f"    {r['chunk_id']}: {r['chunk_tokens']} tokens")
            continue

        # Append and save
        new_df = pd.DataFrame(new_records)
        # Align columns to existing schema
        for col in chunks_df.columns:
            if col not in new_df.columns:
                new_df[col] = None
        new_df = new_df[[c for c in chunks_df.columns if c in new_df.columns]]
        combined = pd.concat([chunks_df, new_df], ignore_index=True)
        combined.to_parquet(chunks_path, index=False)
        print(f"  Saved {chunks_path} ({len(combined)} total chunks, +{len(new_records)} new)")
        total_added += len(new_records)

    if dry_run or total_added == 0:
        print("\nDry-run complete" if dry_run else "\nNo new chunks added.")
        return

    # Rebuild FAISS index for all docs
    print(f"\n{'='*60}")
    print(f"Rebuilding FAISS indexes for all docs ({total_added} new chunks total) …")
    result = subprocess.run(
        [PYTHON, str(REPO / "scripts" / "build_index.py")],
        capture_output=False,
    )
    if result.returncode != 0:
        print("ERROR: build_index.py failed — check output above")
        sys.exit(1)
    print("Index rebuild complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill front-section table chunks")
    parser.add_argument("--dry-run", action="store_true", help="Report what would change without writing")
    args = parser.parse_args()
    run(dry_run=args.dry_run)
