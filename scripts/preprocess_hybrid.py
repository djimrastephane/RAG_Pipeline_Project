"""
Preprocess thesis PDFs into page-, section-, and chunk-level artifacts for the
final hybrid retrieval pipeline.

This script is the main entrypoint for document preprocessing. It runs page
extraction, cleaning, OCR fallback checks, section inference, table handling,
and chunk generation, then writes the canonical per-document outputs used by
indexing and retrieval evaluation.
"""

from __future__ import annotations

from typing import Optional

# This script orchestrates the hybrid preprocessing pipeline.
# Core logic lives in src/rag_pdf/ modules for clarity and testability.

import argparse
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# noinspection DuplicatedCode
try:
    import pymupdf as fitz  # Preferred import name for requirements inspection compatibility.
except Exception as e:
    raise RuntimeError(
        "Failed to import PyMuPDF.\n"
        "Fix: pip uninstall -y fitz frontend && pip install -U pymupdf\n"
    ) from e

import pdfplumber
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
src_path = repo_root / "src"
if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from rag_pdf.boilerplate import remove_repeated_header_footer_lines, strip_by_coordinates
from rag_pdf.chunking import (
    chunk_text_by_tokens,
    count_tokens,
    get_encoder,
    require_encoder,
    split_text_for_segment_aware_chunking_with_patterns,
)
from rag_pdf.config import PreprocessConfig, RegionConfig, TableExtractConfig
from rag_pdf.extract_page import OCR_AVAILABLE, extract_page_struct_hybrid, extract_page_with_ocr
from rag_pdf.headings import (
    is_part_label,
    is_section_anchor_line,
    looks_like_heading_text_only,
    looks_like_lettered_subsection,
    select_heading_candidates,
)
from rag_pdf.metrics import StepTimer, safe_json_dump
from rag_pdf.ocr_quality import evaluate_ocr_quality
from rag_pdf.region_classify import classify_region
from rag_pdf.region_segment import segment_page_into_regions
from rag_pdf.schemas import build_page_list_struct, make_chunk_id_global
from rag_pdf.sections import build_sections_from_pages, find_section_for_page
from rag_pdf.table_detect import (
    classify_page_content,
    contains_many_numbers,
    detect_table_type,
    is_table_like_from_raw_lines,
    is_graphics_table_like,
    is_column_alignment_table_like,
)
from rag_pdf.table_extract import process_table_pages
from rag_pdf.text_normalize import (
    extract_report_metadata_from_pdf,
    extract_report_year_from_filename,
    normalize_line,
    normalize_page_text,
    now_utc_iso,
)
from runtime_env import collect_runtime_provenance, critical_environment_checks


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    alpha = sum(c.isalpha() for c in text)
    return alpha / max(len(text), 1)


def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(c.isdigit() for c in text)
    return digits / max(len(text), 1)


def _env_or_default(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val else default


def _env_flag(name: str, default: str = "0") -> bool:
    return _env_or_default(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_common_header_footer_line(text: str, common_header: set[str], common_footer: set[str]) -> bool:
    return (text in common_header or text in common_footer) and not is_section_anchor_line(text)


# noinspection DuplicatedCode
def _extract_top_lines(lines_all: list[dict], k: int) -> list[dict]:
    if not lines_all:
        return []
    sorted_lines = sorted(
        lines_all,
        key=lambda l: (float(l.get("y0", 0.0)), float(l.get("x0", 0.0))),
    )
    top: list[dict] = []
    for ln in sorted_lines[:k]:
        txt = str(ln.get("text", "")).strip()
        if not txt:
            continue
        top.append(
            {
                "text": txt,
                "y0": float(ln.get("y0", 0.0)),
                "y1": float(ln.get("y1", 0.0)),
            }
        )
    return top


def parse_args() -> argparse.Namespace:
    default_pdf_path = (
        repo_root
        / "Data/Annual Accounts NHS Grampian/Preliminary_Test/Grampian-2022-2023.pdf"
    )
    default_out_root = repo_root / "data_processed"
    parser = argparse.ArgumentParser(
        description="Run hybrid preprocessing with optional OCR fallback."
    )
    parser.add_argument(
        "--pdf-path",
        default=_env_or_default("PDF_PATH", str(default_pdf_path)),
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--out-root",
        default=_env_or_default("OUT_ROOT", str(default_out_root)),
        help="Output root directory.",
    )
    parser.add_argument(
        "--chunk-size-tokens",
        type=int,
        default=int(_env_or_default("CHUNK_SIZE_TOKENS", "224")),
        help="Target chunk size in tokens for text chunking.",
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=int(_env_or_default("CHUNK_OVERLAP_TOKENS", "56")),
        help="Chunk overlap in tokens for text chunking.",
    )
    parser.add_argument(
        "--cross-page-sentence-overlap",
        action="store_true",
        default=_env_flag("CROSS_PAGE_SENTENCE_OVERLAP", "0"),
        help="Add one cross-page text chunk when a page ends mid-sentence and the next page continues it.",
    )
    parser.add_argument(
        "--cross-page-overlap-max-chars",
        type=int,
        default=int(_env_or_default("CROSS_PAGE_OVERLAP_MAX_CHARS", "320")),
        help="Max chars taken from each side of a page boundary for cross-page overlap chunks.",
    )
    parser.add_argument(
        "--segment-aware-chunking",
        action="store_true",
        default=_env_flag("SEGMENT_AWARE_CHUNKING", "1"),
        help="Enable segment-aware splitting before token chunking.",
    )
    parser.add_argument(
        "--whole-doc-markdown-mode",
        action="store_true",
        default=_env_flag("WHOLE_DOC_MARKDOWN_MODE", "0"),
        help="Build markdown-style page text (with optional table injection) before chunking.",
    )
    parser.add_argument(
        "--markdown-header-carry-forward",
        action="store_true",
        default=_env_flag("MARKDOWN_HEADER_CARRY_FORWARD", "1"),
        help="Prepend section/subsection markdown headers to each chunk in markdown mode.",
    )
    parser.add_argument(
        "--markdown-table-injection",
        action="store_true",
        default=_env_flag("MARKDOWN_TABLE_INJECTION", "1"),
        help="Inject table summary/facts/markdown into page markdown in markdown mode.",
    )
    parser.add_argument(
        "--fallback-min-chars",
        type=int,
        default=int(_env_or_default("FALLBACK_MIN_CHARS", "80")),
        help="Minimum extracted chars before triggering extractor fallback quality checks.",
    )
    parser.add_argument(
        "--table-chunking",
        default=_env_or_default("TABLE_CHUNKING", "baseline"),
        choices=["baseline", "row_preserving", "two_stage", "row_blocks"],
        help="Table chunk construction strategy (affects only table chunks).",
    )
    parser.add_argument(
        "--region-diagnostics",
        action="store_true",
        default=_env_flag("REGION_DIAGNOSTICS", "0"),
        help="Emit region-level segmentation/classification diagnostics without changing main routing.",
    )
    parser.add_argument(
        "--table-page-backup-text-chunks",
        action="store_true",
        default=_env_flag("TABLE_PAGE_BACKUP_TEXT_CHUNKS", "0"),
        help="Also chunk clean page text for pages classified as table (fallback safety net).",
    )
    parser.add_argument(
        "--table-extract-return-all-tables",
        action="store_true",
        default=_env_flag("TABLE_EXTRACT_RETURN_ALL_TABLES", "0"),
        help="Keep all valid tables found on a page instead of only the single best table.",
    )
    parser.add_argument(
        "--table-extract-secondary-bottom-pass",
        action="store_true",
        default=_env_flag("TABLE_EXTRACT_SECONDARY_BOTTOM_PASS", "0"),
        help="Run an additional bottom-region stream pass for likely multi-table pages.",
    )
    parser.add_argument(
        "--require-tiktoken",
        action="store_true",
        default=_env_flag("REQUIRE_TIKTOKEN", "1"),
        help="Fail fast if tiktoken is unavailable instead of falling back to word-based estimation.",
    )
    return parser.parse_args()


def _clean_heading_for_markdown(text: Optional[str]) -> str:
    s = str(text or "").strip()
    if not s or s.lower() == "unknown":
        return ""
    return re.sub(r"\s+", " ", s)


def _build_table_injection_block(table_rows: list[dict]) -> str:
    blocks: list[str] = []
    for row in table_rows:
        summary = str(row.get("table_summary") or "").strip()
        header_inj = str(row.get("table_header_injection") or "").strip()
        md = str(row.get("table_markdown") or "").strip()
        parts: list[str] = []
        if summary:
            parts.append(f"> Table summary: {summary}")
        if header_inj:
            facts = "\n".join(f"> {ln}" for ln in header_inj.splitlines() if str(ln).strip())
            if facts:
                parts.append("> Table facts:\n" + facts)
        if md:
            parts.append(md)
        if parts:
            blocks.append("\n\n".join(parts))
    return "\n\n".join(blocks).strip()


def _build_markdown_heading_prefix(
    part: Optional[str],
    section: Optional[str],
    subsection: Optional[str],
) -> list[str]:
    prefix: list[str] = []
    prt = _clean_heading_for_markdown(part)
    sec = _clean_heading_for_markdown(section)
    sub = _clean_heading_for_markdown(subsection)
    if prt:
        prefix.append(f"# {prt}")
    if sec:
        prefix.append(f"## {sec}")
    if sub:
        prefix.append(f"### {sub}")
    return prefix


def _compose_page_markdown_text(
    page_no: int,
    base_text: str,
    part: Optional[str],
    section: Optional[str],
    subsection: Optional[str],
    table_rows: list[dict],
    inject_tables: bool,
) -> str:
    lines: list[str] = _build_markdown_heading_prefix(part, section, subsection)
    lines.append(f"#### Page {page_no}")
    body = str(base_text or "").strip()
    if body:
        lines.append(body)
    if inject_tables and table_rows:
        block = _build_table_injection_block(table_rows)
        if block:
            lines.append(block)
    return "\n\n".join([ln for ln in lines if str(ln).strip()]).strip()


def _prepend_header_context(
    chunk_text: str,
    part: Optional[str],
    section: Optional[str],
    subsection: Optional[str],
) -> str:
    """
    Ensure chunk carries section context in Markdown mode.
    """
    text = str(chunk_text or "").strip()
    if not text:
        return text
    if text.startswith("#"):
        return text
    prefix = _build_markdown_heading_prefix(part, section, subsection)
    if not prefix:
        return text
    return "\n".join(prefix) + "\n\n" + text


def _set_module_cfg_attrs(module, cfg: PreprocessConfig, attr_names: tuple[str, ...]) -> None:
    for attr in attr_names:
        setattr(module, attr, getattr(cfg, attr))


def _apply_config_overrides(cfg: PreprocessConfig) -> None:
    import rag_pdf.boilerplate as boilerplate_mod
    import rag_pdf.extract_page as extract_page_mod
    import rag_pdf.headings as headings_mod
    import rag_pdf.region_segment as region_segment_mod
    import rag_pdf.table_camelot as table_camelot_mod
    import rag_pdf.table_chunking as table_chunking_mod
    import rag_pdf.table_detect as table_detect_mod
    import rag_pdf.table_extract as table_extract_mod
    import rag_pdf.table_markdown as table_markdown_mod

    _set_module_cfg_attrs(
        boilerplate_mod,
        cfg,
        (
            "TOP_STRIP_FRAC",
            "BOTTOM_STRIP_FRAC",
            "LEFT_STRIP_FRAC",
            "RIGHT_STRIP_FRAC",
            "HEADER_FOOTER_REPEAT_FRAC",
            "TOP_LINE_K",
            "BOT_LINE_K",
        ),
    )
    _set_module_cfg_attrs(
        headings_mod,
        cfg,
        (
            "HEADING_MAX_CHARS",
            "HEADING_MIN_CHARS",
            "HEADING_FONT_BOOST_FRAC",
        ),
    )
    _set_module_cfg_attrs(
        extract_page_mod,
        cfg,
        (
            "PRIMARY_EXTRACTOR",
            "FALLBACK_MIN_CHARS",
            "FALLBACK_ON_BAD_TEXT",
            "FALLBACK_ON_EXCEPTION",
        ),
    )
    _set_module_cfg_attrs(
        table_detect_mod,
        cfg,
        (
            "TABLE_DIGIT_RATIO",
            "TABLE_SPACE_RATIO",
            "TABLE_MIN_LINES",
        ),
    )
    setattr(table_detect_mod, "TABLE_DETECT_CFG", cfg.TABLE_DETECT)
    _set_module_cfg_attrs(
        table_extract_mod,
        cfg,
        (),
    )
    setattr(table_camelot_mod, "TABLE_EXTRACT_CFG", cfg.TABLE_EXTRACT)
    setattr(table_chunking_mod, "TABLE_EXTRACT_CFG", cfg.TABLE_EXTRACT)
    setattr(table_extract_mod, "TABLE_EXTRACT_CFG", cfg.TABLE_EXTRACT)
    setattr(table_markdown_mod, "TABLE_EXTRACT_CFG", cfg.TABLE_EXTRACT)
    setattr(region_segment_mod, "REGION_CFG", cfg.REGION)


def _attach_section_columns(table_chunks_df: pd.DataFrame, sections_df: pd.DataFrame) -> None:
    if len(table_chunks_df) == 0:
        return
    mapped = [
        find_section_for_page(sections_df, int(p))
        for p in table_chunks_df["page_start"].tolist()
    ]
    parts, sections, subsections = zip(*mapped)
    table_chunks_df["part"] = list(parts)
    table_chunks_df["section_title"] = list(sections)
    table_chunks_df["subsection_title"] = list(subsections)


def _build_page_chunks(
    text: str,
    cfg: PreprocessConfig,
    enc,
) -> tuple[list[tuple[int, str, str, str, bool]], bool]:
    if cfg.SEGMENT_AWARE_CHUNKING:
        segments = split_text_for_segment_aware_chunking_with_patterns(
            text,
            insert_patterns=tuple(cfg.SEGMENT_BOUNDARY_INSERT_PATTERNS),
            boundary_match_patterns=tuple(cfg.SEGMENT_BOUNDARY_MATCH_PATTERNS),
            boundary_search_patterns=tuple(cfg.SEGMENT_BOUNDARY_SEARCH_PATTERNS),
            uppercase_heading_pattern=str(cfg.SEGMENT_UPPERCASE_HEADING_PATTERN),
            uppercase_heading_max_words=int(cfg.SEGMENT_UPPERCASE_HEADING_MAX_WORDS),
        )
        segment_aware_applied = len(segments) > 1
    else:
        segments = [("segment_000", text)]
        segment_aware_applied = False

    page_chunks: list[tuple[int, str, str, str, bool]] = []
    for seg_idx, segment in enumerate(segments):
        seg_chunks = chunk_text_by_tokens(
            segment.text,
            cfg.CHUNK_SIZE_TOKENS,
            cfg.CHUNK_OVERLAP_TOKENS,
            enc,
        )
        for c in seg_chunks:
            page_chunks.append((seg_idx, segment.title, segment.boundary_type, c, bool(segment.segment_has_search_hit)))
    return page_chunks, segment_aware_applied


def _normalize_chunk_text_for_boundary(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _page_ends_mid_sentence(text: str) -> bool:
    tail = _normalize_chunk_text_for_boundary(text).rstrip()
    if not tail:
        return False
    tail = tail.rstrip("\"'”’)]}")
    if not tail:
        return False
    return tail[-1] not in ".!?:;"


def _extract_trailing_sentence_window(text: str, max_chars: int) -> str:
    normalized = _normalize_chunk_text_for_boundary(text)
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    start = max(0, len(normalized) - max_chars)
    boundary_positions = [
        normalized.rfind(mark, 0, start)
        for mark in (". ", "! ", "? ", "; ", ": ")
    ]
    boundary = max(boundary_positions)
    if boundary >= 0:
        start = boundary + 2
    return normalized[start:].strip()


def _extract_leading_sentence_window(text: str, max_chars: int) -> str:
    normalized = _normalize_chunk_text_for_boundary(text)
    if not normalized:
        return ""
    normalized = re.sub(r"^[A-Z][A-Z0-9 ,/&()\-]{6,}\s+", "", normalized).strip()
    if not normalized:
        return ""
    window = normalized[:max_chars].strip()
    if not window:
        return ""
    for idx, ch in enumerate(window):
        if ch in ".!?":
            return window[: idx + 1].strip()
    return window


def _make_text_chunk_record(
    *,
    doc_id: str,
    corpus_id: str,
    report_year,
    report_year_source: str,
    period_end_date,
    run_date_utc: str,
    page_no: int,
    chunk_idx: int,
    seg_idx: int,
    seg_title: str,
    seg_boundary_type: str,
    seg_has_search_hit: bool,
    chunk_text: str,
    part,
    section,
    subsection,
    cfg: PreprocessConfig,
    enc,
    pages: Optional[list[int]] = None,
    chunk_id_local: Optional[str] = None,
) -> Optional[dict]:
    wc = len(chunk_text.split())
    if wc < cfg.MIN_CHUNK_WORDS:
        return None
    chunk_id_local = chunk_id_local or f"p{page_no:04d}_{chunk_idx:03d}"
    pages = [int(p) for p in (pages or [page_no])]
    page_list_struct = build_page_list_struct(pages)
    return {
        "doc_id": doc_id,
        "corpus_id": corpus_id,
        "report_year": report_year,
        "report_year_source": report_year_source,
        "period_end_date": period_end_date,
        "run_date_utc": run_date_utc,
        "chunk_id": chunk_id_local,
        "chunk_id_global": make_chunk_id_global(doc_id, chunk_id_local),
        "part": part,
        "section_title": section,
        "subsection_title": subsection,
        "page_start": min(pages),
        "page_end": max(pages),
        "pages": pages,
        "page_list": page_list_struct,
        "chunk_text": chunk_text,
        "chunk_tokens": count_tokens(chunk_text, enc),
        "word_count": wc,
        "segment_title": seg_title,
        "segment_id": f"s{seg_idx:02d}",
        "segment_boundary_type": str(seg_boundary_type or "CONTINUATION"),
        "segment_has_search_hit": bool(seg_has_search_hit),
        "segment_aware": bool(cfg.SEGMENT_AWARE_CHUNKING),
        "is_table_like": False,
        "many_numbers": contains_many_numbers(chunk_text),
        "is_table": False,
        "table_type": None,
        "table_ref": None,
    }


def _append_text_chunks_for_page(
    *,
    text_chunks: list[dict],
    doc_id: str,
    corpus_id: str,
    report_year,
    report_year_source: str,
    period_end_date,
    run_date_utc: str,
    page_no: int,
    text: str,
    part,
    section,
    subsection,
    cfg: PreprocessConfig,
    enc,
) -> tuple[bool, int]:
    page_chunks, segment_aware_applied = _build_page_chunks(text, cfg, enc)
    created = 0
    for j, (seg_idx, seg_title, seg_boundary_type, ctext, seg_has_search_hit) in enumerate(page_chunks):
        ctext_final = ctext
        if cfg.WHOLE_DOC_MARKDOWN_MODE and cfg.MARKDOWN_HEADER_CARRY_FORWARD:
            ctext_final = _prepend_header_context(
                chunk_text=ctext,
                part=part,
                section=section,
                subsection=subsection,
            )
        chunk_record = _make_text_chunk_record(
            doc_id=doc_id,
            corpus_id=corpus_id,
            report_year=report_year,
            report_year_source=report_year_source,
            period_end_date=period_end_date,
            run_date_utc=run_date_utc,
            page_no=page_no,
            chunk_idx=j,
            seg_idx=seg_idx,
            seg_title=seg_title,
            seg_boundary_type=seg_boundary_type,
            seg_has_search_hit=seg_has_search_hit,
            chunk_text=ctext_final,
            part=part,
            section=section,
            subsection=subsection,
            cfg=cfg,
            enc=enc,
        )
        if chunk_record is not None:
            text_chunks.append(chunk_record)
            created += 1
    return segment_aware_applied, created


def _append_cross_page_overlap_chunk(
    *,
    text_chunks: list[dict],
    doc_id: str,
    corpus_id: str,
    report_year,
    report_year_source: str,
    period_end_date,
    run_date_utc: str,
    page_no: int,
    next_page_no: int,
    current_text: str,
    next_text: str,
    chunk_idx: int,
    part,
    section,
    subsection,
    cfg: PreprocessConfig,
    enc,
) -> bool:
    if not cfg.CROSS_PAGE_SENTENCE_OVERLAP or cfg.WHOLE_DOC_MARKDOWN_MODE:
        return False
    if next_page_no <= page_no:
        return False
    if not _page_ends_mid_sentence(current_text):
        return False

    trailing = _extract_trailing_sentence_window(current_text, cfg.CROSS_PAGE_OVERLAP_MAX_CHARS)
    leading = _extract_leading_sentence_window(next_text, cfg.CROSS_PAGE_OVERLAP_MAX_CHARS)
    if not trailing or not leading:
        return False

    overlap_text = f"{trailing} {leading}".strip()
    if not overlap_text or normalize_page_text(overlap_text) == normalize_page_text(trailing):
        return False

    chunk_record = _make_text_chunk_record(
        doc_id=doc_id,
        corpus_id=corpus_id,
        report_year=report_year,
        report_year_source=report_year_source,
        period_end_date=period_end_date,
        run_date_utc=run_date_utc,
        page_no=page_no,
        chunk_idx=chunk_idx,
        seg_idx=99,
        seg_title="cross_page_sentence_overlap",
        seg_boundary_type="CROSS_PAGE_CONTINUATION",
        seg_has_search_hit=False,
        chunk_text=overlap_text,
        part=part,
        section=section,
        subsection=subsection,
        cfg=cfg,
        enc=enc,
        pages=[page_no, next_page_no],
        chunk_id_local=f"p{page_no:04d}_x{chunk_idx:03d}",
    )
    if chunk_record is None:
        return False
    text_chunks.append(chunk_record)
    return True


def main() -> None:
    """
    Execute hybrid text + table preprocessing pipeline.

    Pipeline stages:
    0. Extract metadata from cover pages
    1. Page extraction with hybrid loader (PyMuPDF + pdfplumber fallback)
    2. Coordinate-based boilerplate stripping (orientation-aware)
    3. Repetition-based header/footer removal
    4. Page classification (text vs. table)
    5. Section inference from headings
    6. Fork processing:
       - Text pages → standard chunking
       - Table pages → dual representation (summary + structured)
    7. Write outputs (parquet + CSV)
    """
    args = parse_args()
    cfg = PreprocessConfig(
        PDF_PATH=Path(args.pdf_path),
        OUT_ROOT=Path(args.out_root),
        CORPUS_ID=None,
        CHUNK_SIZE_TOKENS=int(args.chunk_size_tokens),
        CHUNK_OVERLAP_TOKENS=int(args.chunk_overlap_tokens),
        CROSS_PAGE_SENTENCE_OVERLAP=bool(args.cross_page_sentence_overlap),
        CROSS_PAGE_OVERLAP_MAX_CHARS=int(args.cross_page_overlap_max_chars),
        SEGMENT_AWARE_CHUNKING=bool(args.segment_aware_chunking),
        WHOLE_DOC_MARKDOWN_MODE=bool(args.whole_doc_markdown_mode),
        MARKDOWN_HEADER_CARRY_FORWARD=bool(args.markdown_header_carry_forward),
        MARKDOWN_TABLE_INJECTION=bool(args.markdown_table_injection),
        TABLE_CHUNKING_STRATEGY=str(args.table_chunking),
        TABLE_PAGE_BACKUP_TEXT_CHUNKS=bool(args.table_page_backup_text_chunks),
        TABLE_EXTRACT_RETURN_ALL_TABLES=bool(args.table_extract_return_all_tables),
        TABLE_EXTRACT_SECONDARY_BOTTOM_PASS=bool(args.table_extract_secondary_bottom_pass),
        TOP_STRIP_FRAC=0.08,
        BOTTOM_STRIP_FRAC=0.08,
        LEFT_STRIP_FRAC=0.08,
        RIGHT_STRIP_FRAC=0.08,
        HEADER_FOOTER_REPEAT_FRAC=0.40,
        TOP_LINE_K=5,
        BOT_LINE_K=5,
        HEADING_MAX_CHARS=110,
        HEADING_MIN_CHARS=4,
        HEADING_FONT_BOOST_FRAC=0.85,
        MIN_CHUNK_WORDS=20,
        PRIMARY_EXTRACTOR="pymupdf",
        FALLBACK_MIN_CHARS=int(args.fallback_min_chars),
        FALLBACK_ON_BAD_TEXT=True,
        FALLBACK_ON_EXCEPTION=True,
        TABLE_DIGIT_RATIO=0.15,
        TABLE_SPACE_RATIO=0.3,
        TABLE_MIN_LINES=1,
        TABLE_EXTRACT=TableExtractConfig(
            CAMELOT_LATTICE_ACCURACY_THRESHOLD=70,
            TABLE_SUMMARY_MAX_ROWS=5,
            TABLE_SUMMARY_WORD_TARGET=140,
            TABLE_ROW_CHUNK_WORD_TARGET=320,
            TABLE_ROW_CHUNK_WORD_HARD_MAX=450,
            TABLE_ROW_CHUNK_MAX_ROWS=10,
            TABLE_LOCAL_FACTS_MAX=24,
            TABLE_SUMMARY_KEY_ROWS_MAX=5,
        ),
        REGION=RegionConfig(
            ENABLE_DIAGNOSTICS=bool(args.region_diagnostics),
        ),
        OCR_MIN_ALPHA_RATIO=0.3,
        OCR_MIN_DIGIT_RATIO=0.6,
        OCR_QUALITY_MIN_CHARS=200,
        OCR_QUALITY_MIN_ALPHA_WORDS=30,
        OCR_QUALITY_MAX_SYMBOL_RATIO=0.35,
        OCR_QUALITY_REPEAT_TOKEN_MAX_COUNT=20,
        OCR_QUALITY_REPEAT_TOKEN_MAX_LEN=4,
        OCR_QUALITY_MIN_NON_EMPTY_LINES=4,
        OCR_QUALITY_REJECT_MIN_FLAGS=2,
    )

    _apply_config_overrides(cfg)

    timer = StepTimer()
    t_doc_start = time.perf_counter()

    if not cfg.PDF_PATH.exists():
        raise FileNotFoundError(f"PDF not found: {cfg.PDF_PATH}")

    doc_id = cfg.PDF_PATH.stem
    run_date_utc = now_utc_iso()
    run_utc = run_date_utc
    enc = require_encoder() if bool(args.require_tiktoken) else get_encoder()
    corpus_id = cfg.CORPUS_ID or doc_id

    print(f"\n{'=' * 60}")
    print(f"Processing: {doc_id}")
    print(f"{'=' * 60}\n")

    doc = fitz.open(cfg.PDF_PATH)
    timer.mark("Open PDF (PyMuPDF)")

    # noinspection DuplicatedCode
    with pdfplumber.open(str(cfg.PDF_PATH)) as pdf_plumber:
        # noinspection DuplicatedCode
        timer.mark("Open PDF (PDFPlumber)")

        # Extract cover metadata
        pdf_meta = extract_report_metadata_from_pdf(doc, max_pages=2)
        report_year_from_pdf = pdf_meta.get("report_year_from_pdf")
        report_year_from_filename = extract_report_year_from_filename(doc_id)

        report_year = report_year_from_pdf or report_year_from_filename
        report_year_source = "pdf_cover" if report_year_from_pdf else "filename"
        period_end_date = pdf_meta.get("period_end_date")

        print(f"Report Year: {report_year} (source: {report_year_source})")
        print(f"Period End: {period_end_date or 'Not detected'}\n")

        timer.mark("Step 0: cover metadata extraction")

        # Extract all pages
        pages_text_lines = {}
        page_heading_candidates = {}
        page_top_lines = {}
        page_extractor_used = {}
        page_extractor_notes = {}

        qa_removed_top = defaultdict(list)
        qa_removed_bottom = defaultdict(list)

        print("Extracting pages...")
        page_structs = {}
        time_text_extract_total = 0.0
        time_coord_strip_total = 0.0
        time_ocr_raw_total = 0.0
        ocr_raw_pages_detected = 0
        ocr_raw_pages_accepted = 0

        for i in range(doc.page_count):
            if (i + 1) % 20 == 0:
                print(f"  Page {i + 1}/{doc.page_count}")

            page_no = i + 1

            t_extract_start = time.perf_counter()
            s, used, note = extract_page_struct_hybrid(
                doc,
                pdf_plumber,
                i,
                pdf_path=str(cfg.PDF_PATH),
            )
            time_text_extract_total += time.perf_counter() - t_extract_start
            page_structs[page_no] = s
            page_extractor_used[page_no] = used
            page_extractor_notes[page_no] = note
            if "ocr_raw_attempted" in note:
                ocr_raw_pages_detected += 1
            if used == "ocr":
                time_ocr_raw_total += time.perf_counter() - t_extract_start
                ocr_raw_pages_accepted += 1

            # Check if raw lines look like a table (before cleanup)
            raw_lines = [ln["text"] for ln in s.get("lines_all", [])]
            is_raw_table = is_table_like_from_raw_lines(raw_lines)
            try:
                drawings = doc.load_page(i).get_drawings()
            except (RuntimeError, ValueError):
                drawings = []
            if is_graphics_table_like(drawings):
                is_raw_table = True
            if is_column_alignment_table_like(s.get("lines_all", [])):
                is_raw_table = True

            t_strip_start = time.perf_counter()
            kept, rem_a, rem_b = strip_by_coordinates(
                s["lines_all"],
                page_height=s["page_height"],
                page_width=s["page_width"],
                rotation=s["rotation"],
            )
            time_coord_strip_total += time.perf_counter() - t_strip_start

            pages_text_lines[page_no] = kept
            page_heading_candidates[page_no] = select_heading_candidates(
                s["lines_all"], s["p95_font"]
            )
            page_top_lines[page_no] = _extract_top_lines(
                s["lines_all"], k=max(cfg.TOP_LINE_K, 10)
            )

            # Store raw table flag for later use
            pages_text_lines[page_no] = (kept, is_raw_table)

            qa_removed_top[page_no] = rem_a
            qa_removed_bottom[page_no] = rem_b

        timer.mark("Step 1: page extraction + coord strip")

        # Remove repeated headers/footers
        pages_text_only = {pno: lines if isinstance(lines, list) else lines[0]
                           for pno, lines in pages_text_lines.items()}
        pages_text_lines2, common_header, common_footer = remove_repeated_header_footer_lines(
            pages_text_only
        )
        timer.mark("Step 2: repeated header/footer strip")
        for page_no, lines in page_top_lines.items():
            filtered_top_lines: list[dict] = []
            for ln in lines:
                txt = normalize_line(str(ln.get("text", "")))
                if not txt:
                    continue
                if _is_common_header_footer_line(txt, common_header, common_footer):
                    continue
                filtered_top_lines.append(ln)
            page_top_lines[page_no] = filtered_top_lines
        for page_no in range(1, doc.page_count + 1):
            cleaned_lines = pages_text_lines2.get(page_no, [])
            cleaned_set = {
                normalize_line(l) for l in cleaned_lines if normalize_line(l)
            }
            raw_candidates = page_heading_candidates.get(page_no, [])
            filtered_headings: list[str] = []
            for cand in raw_candidates:
                norm = normalize_line(str(cand))
                if not norm:
                    continue
                if _is_common_header_footer_line(norm, common_header, common_footer):
                    continue
                if cleaned_set and norm not in cleaned_set:
                    continue
                filtered_headings.append(cand)
            if not filtered_headings:
                for line in cleaned_lines[:25]:
                    if (
                        looks_like_heading_text_only(line)
                        or looks_like_lettered_subsection(line)
                    ) and not is_part_label(line):
                        filtered_headings = [line]
                        break
            page_heading_candidates[page_no] = filtered_headings

        # Build pages dataframe with classification
        print("\nClassifying pages...")
        pages_records = []
        region_records = []
        text_pages = []
        table_pages = []
        ocr_short_pages_triggered = 0
        ocr_short_pages_accepted = 0
        ocr_attempts = 0
        ocr_too_short = 0
        ocr_rejected_quality = 0
        ocr_debug_logged = 0
        ocr_force_table = {}

        for i in range(doc.page_count):
            page_no = i + 1
            raw = "\n".join(pages_text_lines2.get(page_no, [])).strip()
            clean_text = normalize_page_text(raw)
            raw_text_for_table = raw

            ocr_clean_len = None
            ocr_text_len = None
            ocr_quality_reject = None
            ocr_low_text_density = None
            ocr_high_symbol_ratio = None
            ocr_repeated_garbage = None
            ocr_low_line_count = None
            ocr_symbol_ratio = None
            ocr_alpha_word_count = None
            ocr_non_empty_lines = None
            if OCR_AVAILABLE and len(clean_text) < 50:
                ocr_short_pages_triggered += 1
                ocr_attempts += 1
                pre_ocr_text_len = len(clean_text)
                s_for_ocr = page_structs.get(page_no, {}) or {}
                ocr_text = extract_page_with_ocr(
                    str(cfg.PDF_PATH),
                    page_no - 1,
                    int(s_for_ocr.get("rotation", 0) or 0),
                )
                ocr_clean = normalize_page_text(ocr_text)
                ocr_text_len = len(ocr_text)
                ocr_clean_len = len(ocr_clean)
                ocr_alpha = _alpha_ratio(ocr_clean)
                ocr_digits = _digit_ratio(ocr_clean)
                ocr_quality = evaluate_ocr_quality(
                    ocr_text,
                    min_chars=int(cfg.OCR_QUALITY_MIN_CHARS),
                    min_alpha_words=int(cfg.OCR_QUALITY_MIN_ALPHA_WORDS),
                    max_symbol_ratio=float(cfg.OCR_QUALITY_MAX_SYMBOL_RATIO),
                    repeat_token_max_count=int(cfg.OCR_QUALITY_REPEAT_TOKEN_MAX_COUNT),
                    repeat_token_max_len=int(cfg.OCR_QUALITY_REPEAT_TOKEN_MAX_LEN),
                    min_non_empty_lines=int(cfg.OCR_QUALITY_MIN_NON_EMPTY_LINES),
                    reject_min_flags=int(cfg.OCR_QUALITY_REJECT_MIN_FLAGS),
                )
                ocr_flags = ocr_quality.get("flags", {})
                ocr_quality_reject = bool(ocr_quality.get("reject_ocr", False))
                ocr_low_text_density = bool(ocr_flags.get("low_text_density", False))
                ocr_high_symbol_ratio = bool(ocr_flags.get("high_symbol_ratio", False))
                ocr_repeated_garbage = bool(ocr_flags.get("repeated_garbage", False))
                ocr_low_line_count = bool(ocr_flags.get("low_line_count", False))
                ocr_symbol_ratio = float(ocr_quality.get("symbol_ratio", 0.0))
                ocr_alpha_word_count = int(ocr_quality.get("alpha_word_count", 0))
                ocr_non_empty_lines = int(ocr_quality.get("non_empty_lines", 0))
                accept_ocr = len(ocr_clean) >= 50 and (
                    ocr_alpha >= cfg.OCR_MIN_ALPHA_RATIO or ocr_digits > cfg.OCR_MIN_DIGIT_RATIO
                ) and (not ocr_quality_reject)
                if accept_ocr:
                    clean_text = ocr_clean
                    raw_text_for_table = ocr_text
                    page_extractor_used[page_no] = "ocr"
                    note = "clean_text_short_used_ocr"
                    if ocr_digits > cfg.OCR_MIN_DIGIT_RATIO:
                        ocr_force_table[page_no] = True
                        note = f"{note};table_like"
                    page_extractor_notes[page_no] = note
                    ocr_short_pages_accepted += 1
                    print(
                        f"[OCR] page {page_no} used (clean_text_short) "
                        f"pre_ocr_text_len={pre_ocr_text_len} "
                        f"post_ocr_text_len={ocr_clean_len}"
                    )
                else:
                    ocr_too_short += 1
                    if ocr_quality_reject:
                        ocr_rejected_quality += 1
                    if ocr_debug_logged < 3:
                        reason = ""
                        if ocr_quality_reject:
                            reason = f" quality_flags={','.join(ocr_quality.get('active_flags', []))}"
                        print(
                            f"[OCR] page {page_no} too short: "
                            f"ocr_len={len(ocr_text)} ocr_clean_len={len(ocr_clean)}"
                            f"{reason}"
                        )
                        ocr_debug_logged += 1

            # Get raw table flag from earlier detection
            page_data = pages_text_lines.get(page_no)
            is_raw_table = False
            if isinstance(page_data, tuple):
                is_raw_table = page_data[1]

            # Classify page content (combine raw check + post-cleanup check)
            classification = classify_page_content(clean_text)

            # Override if raw structure indicated table
            if is_raw_table and not classification["is_table"]:
                classification["is_table"] = True
                classification["confidence"] = "medium"
                if not classification["table_type"]:
                    classification["table_type"] = detect_table_type(clean_text)

            if ocr_force_table.get(page_no) and not classification["is_table"]:
                classification["is_table"] = True
                classification["confidence"] = "medium"
                if not classification["table_type"]:
                    classification["table_type"] = detect_table_type(clean_text)

            s = page_structs.get(page_no, {})
            if bool(cfg.REGION.ENABLE_DIAGNOSTICS):
                regions = segment_page_into_regions(
                    page_no=page_no,
                    lines_all=s.get("lines_all", []),
                )
                for region in regions:
                    region_cls = classify_region(region)
                    region_records.append({
                        "doc_id": doc_id,
                        "corpus_id": corpus_id,
                        "report_year": report_year,
                        "report_year_source": report_year_source,
                        "period_end_date": period_end_date,
                        "run_date_utc": run_date_utc,
                        "page": page_no,
                        "region_id": region.region_id,
                        "region_index": region.region_index,
                        "x0": region.x0,
                        "y0": region.y0,
                        "x1": region.x1,
                        "y1": region.y1,
                        "width": region.width,
                        "height": region.height,
                        "line_count": region.line_count,
                        "text": region.text,
                        "is_table": bool(region_cls.get("is_table", False)),
                        "is_text": bool(region_cls.get("is_text", False)),
                        "is_raw_table": bool(region_cls.get("is_raw_table", False)),
                        "table_type": region_cls.get("table_type"),
                        "confidence": region_cls.get("confidence"),
                    })

            pages_records.append({
                "doc_id": doc_id,
                "corpus_id": corpus_id,
                "report_year": report_year,
                "report_year_source": report_year_source,
                "period_end_date": period_end_date,
                "run_date_utc": run_date_utc,
                "page": page_no,
                "clean_text": clean_text,
                "heading_candidates": page_heading_candidates.get(page_no, []),
                "top_lines": page_top_lines.get(page_no, []),
                "extractor": page_extractor_used.get(page_no, "unknown"),
                "extractor_notes": page_extractor_notes.get(page_no, ""),
                "ocr_text_len": ocr_text_len,
                "ocr_clean_text_len": ocr_clean_len,
                "ocr_quality_reject": ocr_quality_reject,
                "ocr_low_text_density": ocr_low_text_density,
                "ocr_high_symbol_ratio": ocr_high_symbol_ratio,
                "ocr_repeated_garbage": ocr_repeated_garbage,
                "ocr_low_line_count": ocr_low_line_count,
                "ocr_symbol_ratio": ocr_symbol_ratio,
                "ocr_alpha_word_count": ocr_alpha_word_count,
                "ocr_non_empty_lines": ocr_non_empty_lines,
                "is_table": classification["is_table"],
                "table_type": classification["table_type"],
                "classification_confidence": classification["confidence"],
                "rotation": s.get("rotation", 0),
                "page_width": s.get("page_width", 0.0),
                "page_height": s.get("page_height", 0.0),
            })

            # Split into text vs. table pages
            if classification["is_table"]:
                table_pages.append({
                    "page": page_no,
                    "text": clean_text,
                    "raw_text": raw_text_for_table,
                    "table_type": classification["table_type"],
                    "extractor": page_extractor_used.get(page_no, "unknown"),
                    "rotation": int(s.get("rotation", 0) or 0),
                    "page_width": float(s.get("page_width", 0.0) or 0.0),
                    "page_height": float(s.get("page_height", 0.0) or 0.0),
                    "is_table": True,
                })
            else:
                text_pages.append({
                    "page": page_no,
                    "text": clean_text,
                })

        pages_df = pd.DataFrame(pages_records)

        print(f"  Text pages: {len(text_pages)}")
        print(f"  Table pages: {len(table_pages)}")
        print(f"  OCR short pages: {ocr_short_pages_triggered}")
        print(f"  OCR used pages: {ocr_short_pages_accepted}")
        print(f"  OCR attempts: {ocr_attempts}")
        print(f"  OCR too short: {ocr_too_short}")
        print(f"  OCR rejected by quality flags: {ocr_rejected_quality}")

        timer.mark("Step 3: pages dataframe + classification")

        # Build sections (ToC priors + heuristic overrides)
        sections_result = build_sections_from_pages(pages_df, return_diagnostics=True)
        if isinstance(sections_result, tuple):
            sections_df, section_diag = sections_result
        else:
            sections_df = sections_result
            section_diag = {}
        timer.mark("Step 4: section inference")

        table_chunks_df = pd.DataFrame()
        structured_tables_df = pd.DataFrame()
        table_facts_df = pd.DataFrame()
        _rejected_ocr_table_pages: list[dict] = []
        rejected_ocr_table_pages: list[dict] = []
        table_processed_early = False

        # Markdown mode needs table markdown/summary available before text chunking.
        if cfg.WHOLE_DOC_MARKDOWN_MODE:
            print("\nExtracting tables (early, for markdown injection)...")
            table_chunks_df, structured_tables_df, table_facts_df, _rejected_ocr_table_pages = process_table_pages(
                table_pages,
                cfg.PDF_PATH,
                pdf_plumber,
                doc_id,
                corpus_id,
                report_year,
                period_end_date,
                report_year_source,
                run_date_utc,
                enc,
                chunk_size_tokens=cfg.CHUNK_SIZE_TOKENS,
                table_chunking_strategy=cfg.TABLE_CHUNKING_STRATEGY,
                return_all_tables=bool(cfg.TABLE_EXTRACT_RETURN_ALL_TABLES),
                enable_secondary_bottom_pass=bool(cfg.TABLE_EXTRACT_SECONDARY_BOTTOM_PASS),
            )
            _attach_section_columns(table_chunks_df, sections_df)
            table_processed_early = True

        # Process TEXT pages → standard chunking
        print("\nChunking text pages...")
        text_chunks = []
        segment_aware_applied_pages = 0
        cross_page_overlap_chunks = 0

        table_rows_by_page: dict[int, list[dict]] = {}
        if cfg.WHOLE_DOC_MARKDOWN_MODE and cfg.MARKDOWN_TABLE_INJECTION and len(structured_tables_df) > 0:
            for _, row in structured_tables_df.iterrows():
                try:
                    pno = int(row.get("page_no") or row.get("page") or 0)
                except (TypeError, ValueError):
                    pno = 0
                if pno <= 0:
                    continue
                table_rows_by_page.setdefault(pno, []).append({
                    "table_summary": row.get("table_summary"),
                    "table_header_injection": row.get("table_header_injection"),
                    "table_markdown": row.get("table_markdown"),
                })

        if cfg.WHOLE_DOC_MARKDOWN_MODE:
            text_page_records = [
                {"page": int(r.get("page")), "text": str(r.get("clean_text") or "")}
                for _, r in pages_df.sort_values("page").iterrows()
            ]
        else:
            text_page_records = text_pages

        for idx, page_record in enumerate(text_page_records):
            page_no = int(page_record["page"])
            raw_text = str(page_record.get("text") or "")
            if not raw_text:
                continue

            part, section, subsection = find_section_for_page(sections_df, page_no)
            text = raw_text
            if cfg.WHOLE_DOC_MARKDOWN_MODE:
                text = _compose_page_markdown_text(
                    page_no=page_no,
                    base_text=text,
                    part=part,
                    section=section,
                    subsection=subsection,
                    table_rows=table_rows_by_page.get(page_no, []),
                    inject_tables=bool(cfg.MARKDOWN_TABLE_INJECTION),
                )

            segment_aware_applied, chunk_count = _append_text_chunks_for_page(
                text_chunks=text_chunks,
                doc_id=doc_id,
                corpus_id=corpus_id,
                report_year=report_year,
                report_year_source=report_year_source,
                period_end_date=period_end_date,
                run_date_utc=run_date_utc,
                page_no=page_no,
                text=text,
                part=part,
                section=section,
                subsection=subsection,
                cfg=cfg,
                enc=enc,
            )
            if segment_aware_applied:
                segment_aware_applied_pages += 1
            if cfg.CROSS_PAGE_SENTENCE_OVERLAP and not cfg.WHOLE_DOC_MARKDOWN_MODE and chunk_count >= 0:
                next_record = text_page_records[idx + 1] if idx + 1 < len(text_page_records) else None
                if next_record is not None:
                    next_page_no = int(next_record["page"])
                    next_raw_text = str(next_record.get("text") or "")
                    if next_raw_text and _append_cross_page_overlap_chunk(
                        text_chunks=text_chunks,
                        doc_id=doc_id,
                        corpus_id=corpus_id,
                        report_year=report_year,
                        report_year_source=report_year_source,
                        period_end_date=period_end_date,
                        run_date_utc=run_date_utc,
                        page_no=page_no,
                        next_page_no=next_page_no,
                        current_text=raw_text,
                        next_text=next_raw_text,
                        chunk_idx=chunk_count,
                        part=part,
                        section=section,
                        subsection=subsection,
                        cfg=cfg,
                        enc=enc,
                    ):
                        cross_page_overlap_chunks += 1

        text_chunks_df = pd.DataFrame(text_chunks)
        print(f"  Created {len(text_chunks_df)} text chunks")
        if cfg.SEGMENT_AWARE_CHUNKING:
            print(f"  Segment-aware pages (multi-segment): {segment_aware_applied_pages}")
        if cfg.CROSS_PAGE_SENTENCE_OVERLAP and not cfg.WHOLE_DOC_MARKDOWN_MODE:
            print(f"  Cross-page overlap chunks: {cross_page_overlap_chunks}")

        timer.mark("Step 5: text chunking")

        # Process TABLE pages → dual representation
        if not table_processed_early:
            print("\nExtracting tables...")
            table_chunks_df, structured_tables_df, table_facts_df, rejected_ocr_table_pages = process_table_pages(
                table_pages,
                cfg.PDF_PATH,
                pdf_plumber,
                doc_id,
                corpus_id,
                report_year,
                period_end_date,
                report_year_source,
                run_date_utc,
                enc,
                chunk_size_tokens=cfg.CHUNK_SIZE_TOKENS,
                table_chunking_strategy=cfg.TABLE_CHUNKING_STRATEGY,
                return_all_tables=bool(cfg.TABLE_EXTRACT_RETURN_ALL_TABLES),
                enable_secondary_bottom_pass=bool(cfg.TABLE_EXTRACT_SECONDARY_BOTTOM_PASS),
            )

            # OCR-table pages that failed fallback acceptance should be handled as normal text.
            if rejected_ocr_table_pages:
                print(f"  OCR-table fallback rejected {len(rejected_ocr_table_pages)} page(s); chunking as text")
                for page_record in rejected_ocr_table_pages:
                    page_no = page_record["page"]
                    text = page_record["text"]
                    if not text:
                        continue

                    part, section, subsection = find_section_for_page(sections_df, page_no)
                    _append_text_chunks_for_page(
                        text_chunks=text_chunks,
                        doc_id=doc_id,
                        corpus_id=corpus_id,
                        report_year=report_year,
                        report_year_source=report_year_source,
                        period_end_date=period_end_date,
                        run_date_utc=run_date_utc,
                        page_no=page_no,
                        text=text,
                        part=part,
                        section=section,
                        subsection=subsection,
                        cfg=cfg,
                        enc=enc,
                    )
                text_chunks_df = pd.DataFrame(text_chunks)

        if cfg.TABLE_PAGE_BACKUP_TEXT_CHUNKS and not cfg.WHOLE_DOC_MARKDOWN_MODE:
            rejected_pages: set[int] = set()
            for rec in (_rejected_ocr_table_pages or []):
                try:
                    rejected_pages.add(int(rec.get("page")))
                except Exception:
                    continue
            for rec in (rejected_ocr_table_pages or []):
                try:
                    rejected_pages.add(int(rec.get("page")))
                except Exception:
                    continue

            table_backup_pages = 0
            for tpage in table_pages:
                page_no = int(tpage.get("page", 0) or 0)
                if page_no <= 0 or page_no in rejected_pages:
                    continue
                text = str(tpage.get("text") or "").strip()
                if not text:
                    continue
                part, section, subsection = find_section_for_page(sections_df, page_no)
                _append_text_chunks_for_page(
                    text_chunks=text_chunks,
                    doc_id=doc_id,
                    corpus_id=corpus_id,
                    report_year=report_year,
                    report_year_source=report_year_source,
                    period_end_date=period_end_date,
                    run_date_utc=run_date_utc,
                    page_no=page_no,
                    text=text,
                    part=part,
                    section=section,
                    subsection=subsection,
                    cfg=cfg,
                    enc=enc,
                )
                table_backup_pages += 1
            if table_backup_pages:
                print(f"  Added backup text chunks for {table_backup_pages} table page(s)")
            text_chunks_df = pd.DataFrame(text_chunks)

        _attach_section_columns(table_chunks_df, sections_df)

        print(f"  Extracted {len(structured_tables_df)} tables")
        print(f"  Created {len(table_chunks_df)} table chunks")

        timer.mark("Step 6: table extraction + summarization")

        # Merge text and table chunks
        all_chunks_df = pd.concat([text_chunks_df, table_chunks_df], ignore_index=True)
        all_chunks_df = all_chunks_df.sort_values(["page_start", "chunk_id"]).reset_index(drop=True)

        print(
            f"\nTotal chunks: {len(all_chunks_df)} "
            f"({len(text_chunks_df)} text + {len(table_chunks_df)} table)"
        )

        # Validate chunk page spans
        if len(all_chunks_df) > 0:
            bad_span = all_chunks_df[all_chunks_df["page_start"] != all_chunks_df["page_end"]]
            if len(bad_span) > 0 and not cfg.CROSS_PAGE_SENTENCE_OVERLAP:
                raise ValueError(
                    f"Found {len(bad_span)} chunks spanning multiple pages. "
                    "Pipeline requires page-bounded chunks for accurate citations."
                )
            if len(bad_span) > 0 and cfg.CROSS_PAGE_SENTENCE_OVERLAP:
                print(f"  Retained {len(bad_span)} multi-page text chunks for cross-page sentence continuity")

        timer.mark("Step 7: chunk merging + validation")

        # Write outputs
        out_dir = cfg.OUT_ROOT / doc_id
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nWriting outputs to: {out_dir}")

        pages_df.to_parquet(out_dir / "pages.parquet", index=False)
        sections_df.to_parquet(out_dir / "sections.parquet", index=False)
        sections_df.to_csv(out_dir / "sections.csv", index=False)
        toc_df = section_diag.get("toc_df") if isinstance(section_diag, dict) else None
        if isinstance(toc_df, pd.DataFrame) and len(toc_df) > 0:
            toc_df.to_parquet(out_dir / "toc.parquet", index=False)
            toc_df.to_csv(out_dir / "toc.csv", index=False)
        rejected_sub_df = (
            section_diag.get("rejected_subsections_df") if isinstance(section_diag, dict) else None
        )
        if isinstance(rejected_sub_df, pd.DataFrame):
            rejected_sub_df.to_csv(out_dir / "subsection_rejected_candidates.csv", index=False)
        all_chunks_df.to_parquet(out_dir / "chunks.parquet", index=False)
        ocr_pages_df = pages_df.loc[
            pages_df["extractor"] == "ocr",
            [
                "page",
                "extractor_notes",
                "ocr_text_len",
                "ocr_clean_text_len",
                "ocr_quality_reject",
                "ocr_low_text_density",
                "ocr_high_symbol_ratio",
                "ocr_repeated_garbage",
                "ocr_low_line_count",
                "ocr_symbol_ratio",
                "ocr_alpha_word_count",
                "ocr_non_empty_lines",
                "clean_text",
            ],
        ].copy()
        ocr_pages_df = ocr_pages_df.rename(
            columns={"ocr_clean_text_len": "clean_text_len"}
        )
        ocr_pages_df["clean_text_len"] = ocr_pages_df["clean_text"].fillna("").str.len()
        ocr_pages_df = ocr_pages_df.drop(columns=["clean_text"])
        ocr_pages_df.to_csv(out_dir / "ocr_pages.csv", index=False)
        if region_records:
            regions_df = pd.DataFrame(region_records)
            regions_df.to_parquet(out_dir / "regions.parquet", index=False)
            regions_df.to_csv(out_dir / "regions.csv", index=False)

        # Write structured tables
        if len(structured_tables_df) > 0:
            structured_tables_df.to_parquet(out_dir / "tables_structured.parquet", index=False)
        if len(table_facts_df) > 0:
            table_facts_df.to_parquet(out_dir / "table_facts.parquet", index=False)
        if len(table_chunks_df) > 0:
            diag_cols = [
                "doc_id",
                "chunk_id",
                "table_ref",
                "table_type",
                "table_chunk_kind",
                "row_start_idx",
                "row_end_idx",
                "word_count",
                "page_start",
            ]
            available_diag_cols = [col for col in diag_cols if col in table_chunks_df.columns]
            if available_diag_cols:
                table_chunks_df.loc[:, available_diag_cols].to_csv(
                    out_dir / "table_chunk_diagnostics.csv",
                    index=False,
                )

        timer.mark("Step 8: parquet writes")

        # Generate metrics
        try:
            git_commit_short = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(repo_root),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except (subprocess.SubprocessError, OSError):
            git_commit_short = None
        embedding_model = os.getenv("EMBED_MODEL_NAME") or "sentence-transformers/all-MiniLM-L6-v2"
        tokenizer_backend = "tiktoken" if enc is not None else "word_fallback"
        time_total_wall = time.perf_counter() - t_doc_start
        table_word_counts = (
            pd.to_numeric(table_chunks_df.get("word_count"), errors="coerce").dropna()
            if "word_count" in table_chunks_df.columns
            else pd.Series(dtype=float)
        )
        table_chunk_kind_series = (
            table_chunks_df["table_chunk_kind"].astype(str)
            if "table_chunk_kind" in table_chunks_df.columns
            else pd.Series(dtype=str)
        )
        metrics = {
            "schema_version": "3.0_hybrid",
            "doc_id": doc_id,
            "corpus_id": corpus_id,
            "report_year": report_year,
            "period_end_date": period_end_date,
            "run_utc": run_utc,
            "git_commit_short": git_commit_short,
            "embedding_model": embedding_model,
            "runtime": collect_runtime_provenance(),
            "critical_environment_checks": critical_environment_checks(),
            "counts": {
                "pages_total": len(pages_df),
                "pages_text": len(text_pages),
                "pages_table": len(table_pages),
                "sections_detected": len(sections_df),
                "chunks_total": len(all_chunks_df),
                "chunks_text": len(text_chunks_df),
                "chunks_table": len(table_chunks_df),
                "tables_extracted": len(structured_tables_df),
                "table_facts": len(table_facts_df),
                "table_summary_chunks_total": int((table_chunk_kind_series == "summary").sum()),
                "table_row_chunks_total": int((table_chunk_kind_series == "row_block").sum()),
                "regions": len(region_records),
                "ocr_raw_pages_detected": ocr_raw_pages_detected,
                "ocr_raw_pages_accepted": ocr_raw_pages_accepted,
                "ocr_short_pages_triggered": ocr_short_pages_triggered,
                "ocr_short_pages_accepted": ocr_short_pages_accepted,
                "ocr_rejected_quality": ocr_rejected_quality,
                "toc_detected": bool(section_diag.get("toc_detected", False)) if isinstance(section_diag, dict) else False,
                "toc_pages_count": int(section_diag.get("toc_pages_count", 0)) if isinstance(section_diag, dict) else 0,
                "toc_items_count": int(section_diag.get("toc_items_count", 0)) if isinstance(section_diag, dict) else 0,
                "toc_offset_support_count": int(section_diag.get("toc_offset_support_count", 0)) if isinstance(section_diag, dict) else 0,
                "subsection_reject_count": int(section_diag.get("subsection_reject_count", 0)) if isinstance(section_diag, dict) else 0,
            },
            "derived": {
                "chunks_per_page": (
                    len(all_chunks_df) / max(len(pages_df), 1)
                ),
                "tables_per_100_pages": (
                    len(structured_tables_df) / max(len(pages_df), 1) * 100.0
                ),
                "ocr_raw_acceptance_rate": (
                    ocr_raw_pages_accepted / max(ocr_raw_pages_detected, 1)
                ),
                "ocr_short_acceptance_rate": (
                    ocr_short_pages_accepted / max(ocr_short_pages_triggered, 1)
                ),
                "ocr_quality_reject_rate": (
                    ocr_rejected_quality / max(ocr_attempts, 1)
                ),
                "table_chunk_words_mean": (
                    float(table_word_counts.mean()) if len(table_word_counts) > 0 else 0.0
                ),
                "table_chunk_words_median": (
                    float(table_word_counts.median()) if len(table_word_counts) > 0 else 0.0
                ),
                "table_chunk_words_p95": (
                    float(table_word_counts.quantile(0.95)) if len(table_word_counts) > 0 else 0.0
                ),
                "table_chunk_words_p99": (
                    float(table_word_counts.quantile(0.99)) if len(table_word_counts) > 0 else 0.0
                ),
                "table_chunk_words_max": (
                    float(table_word_counts.max()) if len(table_word_counts) > 0 else 0.0
                ),
                "table_chunks_gt_300": (
                    int((table_word_counts > 300).sum()) if len(table_word_counts) > 0 else 0
                ),
                "table_chunks_gt_400": (
                    int((table_word_counts > 400).sum()) if len(table_word_counts) > 0 else 0
                ),
                "toc_coverage_pct": float(section_diag.get("toc_coverage_pct", 0.0))
                if isinstance(section_diag, dict)
                else 0.0,
                "toc_override_rate": float(section_diag.get("toc_override_rate", 0.0))
                if isinstance(section_diag, dict)
                else 0.0,
                "subsection_unknown_pct_before": float(section_diag.get("subsection_unknown_pct_before", 0.0))
                if isinstance(section_diag, dict)
                else 0.0,
                "subsection_unknown_pct_after": float(section_diag.get("subsection_unknown_pct_after", 0.0))
                if isinstance(section_diag, dict)
                else 0.0,
            },
            "timing": {
                "time_unit": "seconds",
                "time_text_extract_total": round(time_text_extract_total, 6),
                "time_coord_strip_total": round(time_coord_strip_total, 6),
                "time_ocr_raw_total": round(time_ocr_raw_total, 6),
                "time_total_wall": round(time_total_wall, 6),
            },
            "params": {
                "chunk_size_tokens": cfg.CHUNK_SIZE_TOKENS,
                "chunk_overlap_tokens": cfg.CHUNK_OVERLAP_TOKENS,
                "cross_page_sentence_overlap": bool(cfg.CROSS_PAGE_SENTENCE_OVERLAP),
                "cross_page_overlap_max_chars": int(cfg.CROSS_PAGE_OVERLAP_MAX_CHARS),
                "tokenizer_backend": tokenizer_backend,
                "tokenizer_exact_counting": bool(enc is not None),
                "require_tiktoken": bool(args.require_tiktoken),
                "top_strip_frac": cfg.TOP_STRIP_FRAC,
                "bottom_strip_frac": cfg.BOTTOM_STRIP_FRAC,
                "left_strip_frac": cfg.LEFT_STRIP_FRAC,
                "right_strip_frac": cfg.RIGHT_STRIP_FRAC,
                "header_footer_repeat_frac": cfg.HEADER_FOOTER_REPEAT_FRAC,
                "min_chunk_words": cfg.MIN_CHUNK_WORDS,
                "primary_extractor": cfg.PRIMARY_EXTRACTOR,
                "segment_aware_chunking": bool(cfg.SEGMENT_AWARE_CHUNKING),
                "whole_doc_markdown_mode": bool(cfg.WHOLE_DOC_MARKDOWN_MODE),
                "markdown_header_carry_forward": bool(cfg.MARKDOWN_HEADER_CARRY_FORWARD),
                "markdown_table_injection": bool(cfg.MARKDOWN_TABLE_INJECTION),
                "table_chunking": str(cfg.TABLE_CHUNKING_STRATEGY),
                "table_summary_word_target": int(cfg.TABLE_EXTRACT.TABLE_SUMMARY_WORD_TARGET),
                "table_row_chunk_word_target": int(cfg.TABLE_EXTRACT.TABLE_ROW_CHUNK_WORD_TARGET),
                "table_row_chunk_word_hard_max": int(cfg.TABLE_EXTRACT.TABLE_ROW_CHUNK_WORD_HARD_MAX),
                "table_row_chunk_max_rows": int(cfg.TABLE_EXTRACT.TABLE_ROW_CHUNK_MAX_ROWS),
                "table_local_facts_max": int(cfg.TABLE_EXTRACT.TABLE_LOCAL_FACTS_MAX),
                "table_page_backup_text_chunks": bool(cfg.TABLE_PAGE_BACKUP_TEXT_CHUNKS),
                "table_extract_return_all_tables": bool(cfg.TABLE_EXTRACT_RETURN_ALL_TABLES),
                "table_extract_secondary_bottom_pass": bool(cfg.TABLE_EXTRACT_SECONDARY_BOTTOM_PASS),
                "toc_confidence_threshold": 0.6,
                "toc_allow_cross_major_override": False,
                "toc_subsection_token_overlap_threshold": 0.6,
                "toc_subsection_sequence_threshold": 0.75,
            },
            "toc": {
                "toc_detected": bool(section_diag.get("toc_detected", False)) if isinstance(section_diag, dict) else False,
                "toc_pages_count": int(section_diag.get("toc_pages_count", 0)) if isinstance(section_diag, dict) else 0,
                "toc_items_count": int(section_diag.get("toc_items_count", 0)) if isinstance(section_diag, dict) else 0,
                "toc_offset": int(section_diag.get("toc_offset", 0)) if isinstance(section_diag, dict) else 0,
                "toc_offset_support_count": int(section_diag.get("toc_offset_support_count", 0)) if isinstance(section_diag, dict) else 0,
                "toc_offset_confidence": float(section_diag.get("toc_offset_confidence", 0.0)) if isinstance(section_diag, dict) else 0.0,
                "toc_coverage_pct": float(section_diag.get("toc_coverage_pct", 0.0)) if isinstance(section_diag, dict) else 0.0,
                "toc_override_rate": float(section_diag.get("toc_override_rate", 0.0)) if isinstance(section_diag, dict) else 0.0,
                "subsection_reject_count": int(section_diag.get("subsection_reject_count", 0)) if isinstance(section_diag, dict) else 0,
                "subsection_unknown_pct_before": float(section_diag.get("subsection_unknown_pct_before", 0.0)) if isinstance(section_diag, dict) else 0.0,
                "subsection_unknown_pct_after": float(section_diag.get("subsection_unknown_pct_after", 0.0)) if isinstance(section_diag, dict) else 0.0,
            },
            "table_types_detected": (
                structured_tables_df["table_type"].value_counts().to_dict()
                if len(structured_tables_df) > 0
                else {}
            ),
        }

        safe_json_dump(metrics, out_dir / "metrics.json")
        print(f"  - tokenizer_backend: {tokenizer_backend}")

        print(f"\n{'=' * 60}")
        print("PROCESSING COMPLETE")
        print(f"{'=' * 60}")
        print(f"\nOutputs written to: {out_dir}")
        print(f"  - pages.parquet: {len(pages_df)} pages")
        print(f"  - sections.parquet: {len(sections_df)} sections")
        print(f"  - chunks.parquet: {len(all_chunks_df)} chunks (text + table summaries)")
        if region_records:
            print(f"  - regions.parquet: {len(region_records)} regions")
        if len(structured_tables_df) > 0:
            print(f"  - tables_structured.parquet: {len(structured_tables_df)} tables")
        if len(table_facts_df) > 0:
            print(f"  - table_facts.parquet: {len(table_facts_df)} facts")
        print("  - metrics.json: Pipeline statistics")

        timer.mark("Step 9: metrics + completion")

    doc.close()
    timer.mark("Close documents")
    timer.report()


if __name__ == "__main__":
    main()
