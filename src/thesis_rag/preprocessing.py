from __future__ import annotations

import math
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pdfplumber

from rag_pdf.boilerplate import strip_by_coordinates
from rag_pdf.chunking import split_text_for_segment_aware_chunking_with_patterns
from rag_pdf.config import DEFAULT_CONFIG as LEGACY_CONFIG
from rag_pdf.headings import select_heading_candidates
from rag_pdf.schemas import make_chunk_id_global
from rag_pdf.sections import build_sections_from_pages, find_section_for_page
from rag_pdf.table_extract import process_table_pages
from rag_pdf.table_detect import (
    classify_page_content,
    detect_table_type,
    is_column_alignment_table_like,
    is_graphics_table_like,
    is_table_like_from_raw_lines,
)
from rag_pdf.text_normalize import normalize_line, normalize_page_text

from .chunking import chunk_text, count_tokens, get_encoder
from .ocr import page_needs_ocr
from .schemas import ChunkRecord, ChunkingConfig, OCRConfig, PageRecord

LOGGER = logging.getLogger(__name__)


def remove_repeated_headers_and_footers(
    page_lines: dict[int, list[str]],
    top_k: int = 5,
    bottom_k: int = 5,
    repeat_fraction: float = 0.40,
) -> tuple[dict[int, list[str]], set[str], set[str]]:
    top_lines: list[str] = []
    bottom_lines: list[str] = []
    for lines in page_lines.values():
        normalized = [normalize_line(item) for item in lines if normalize_line(item)]
        top_lines.extend(normalized[:top_k])
        bottom_lines.extend(normalized[-bottom_k:])
    threshold = max(1, math.ceil(len(page_lines) * repeat_fraction))
    common_header = {text for text, count in Counter(top_lines).items() if count >= threshold}
    common_footer = {text for text, count in Counter(bottom_lines).items() if count >= threshold}
    cleaned: dict[int, list[str]] = {}
    for page_number, lines in page_lines.items():
        normalized = [normalize_line(item) for item in lines if normalize_line(item)]
        kept: list[str] = []
        for index, line in enumerate(normalized):
            if index < top_k and line in common_header:
                continue
            if index >= len(normalized) - bottom_k and line in common_footer:
                continue
            kept.append(line)
        cleaned[page_number] = kept
    return cleaned, common_header, common_footer


def build_page_records(
    doc_id: str,
    page_structs: list[tuple[int, dict, str, str]],
    ocr_config: OCRConfig,
) -> list[PageRecord]:
    line_map: dict[int, list[str]] = {}
    stripped_map: dict[int, tuple[list[str], list[str]]] = {}
    metadata: dict[int, tuple[str, str, bool]] = {}
    heading_candidates_by_page: dict[int, list[str]] = {}
    top_lines_by_page: dict[int, list[dict[str, Any]]] = {}
    for page_number, page_struct, extractor_used, quality_note in page_structs:
        kept, removed_header, removed_footer = strip_by_coordinates(
            page_struct.get("lines_all", []),
            page_height=float(page_struct.get("page_height", 0.0)),
            page_width=float(page_struct.get("page_width", 0.0)),
            rotation=int(page_struct.get("rotation", 0)),
        )
        line_map[page_number] = kept
        stripped_map[page_number] = (removed_header, removed_footer)
        raw_text = "\n".join(line.get("text", "") for line in page_struct.get("lines_all", []))
        metadata[page_number] = (extractor_used, quality_note, _accepted_ocr_used(extractor_used, quality_note))
        heading_candidates_by_page[page_number] = select_heading_candidates(
            page_struct.get("lines_all", []),
            float(page_struct.get("p95_font", 0.0) or 0.0),
        )
        top_lines_by_page[page_number] = _extract_top_lines(page_struct.get("lines_all", []), k=max(LEGACY_CONFIG.TOP_LINE_K, 10))
    table_flags = _classify_table_pages(page_structs, cleaned_map=None)
    cleaned_map, common_header, common_footer = remove_repeated_headers_and_footers(line_map)
    table_flags = _classify_table_pages(page_structs, cleaned_map=cleaned_map)
    LOGGER.info("Repeated boilerplate detected: headers=%s footers=%s", len(common_header), len(common_footer))
    records: list[PageRecord] = []
    for page_number in sorted(cleaned_map):
        extractor_used, quality_note, ocr_used = metadata[page_number]
        removed_header, removed_footer = stripped_map[page_number]
        clean_text = normalize_page_text("\n".join(cleaned_map[page_number]))
        raw_text = normalize_page_text("\n".join(line_map[page_number]))
        is_table, table_type = table_flags.get(page_number, (False, None))
        records.append(
            PageRecord(
                page_id=f"{doc_id}:p{page_number:04d}",
                doc_id=doc_id,
                page_number=page_number,
                raw_text=raw_text,
                clean_text=clean_text,
                extractor_used=extractor_used,
                quality_note=quality_note,
                ocr_used=ocr_used,
                is_table=is_table,
                table_type=table_type,
                heading_candidates=heading_candidates_by_page.get(page_number, []),
                top_lines=top_lines_by_page.get(page_number, []),
                header_lines_removed=removed_header,
                footer_lines_removed=removed_footer,
                rotation=int(page_struct.get("rotation", 0) or 0),
                page_width=float(page_struct.get("page_width", 0.0) or 0.0),
                page_height=float(page_struct.get("page_height", 0.0) or 0.0),
            )
        )
    _validate_pages(records)
    return records


def build_chunk_records(
    doc_id: str,
    pages: list[PageRecord],
    config: ChunkingConfig,
    source_pdf_path: str | Path | None = None,
) -> list[ChunkRecord]:
    encoder = get_encoder()
    chunks: list[ChunkRecord] = []
    seen_chunk_ids: set[str] = set()
    sections_df = _build_sections_dataframe(pages)
    sorted_pages = sorted(pages, key=lambda item: item.page_number)
    table_chunks_by_page, rejected_table_pages = _build_legacy_table_chunks(
        doc_id=doc_id,
        pages=sorted_pages,
        config=config,
        encoder=encoder,
        sections_df=sections_df,
        source_pdf_path=source_pdf_path,
    )
    text_pages = [page for page in sorted_pages if not page.is_table or page.page_number in rejected_table_pages]
    next_text_page_by_number = {
        text_pages[index].page_number: text_pages[index + 1]
        for index in range(len(text_pages) - 1)
    }
    for page in sorted_pages:
        if page.page_number < 1:
            raise ValueError(f"Invalid page number for {page.page_id}: {page.page_number}")
        if page.is_table and page.page_number not in rejected_table_pages:
            table_chunks = table_chunks_by_page.get(page.page_number, [])
            if table_chunks:
                for chunk in table_chunks:
                    if chunk.chunk_id in seen_chunk_ids:
                        raise ValueError(f"Duplicate chunk id detected: {chunk.chunk_id}")
                    seen_chunk_ids.add(chunk.chunk_id)
                    chunks.append(chunk)
            else:
                _part, section_title, subsection_title = find_section_for_page(sections_df, page.page_number)
                chunk_id = f"table_p{page.page_number:04d}"
                if chunk_id in seen_chunk_ids:
                    raise ValueError(f"Duplicate chunk id detected: {chunk_id}")
                seen_chunk_ids.add(chunk_id)
                chunks.append(
                    ChunkRecord(
                        chunk_id=chunk_id,
                        doc_id=doc_id,
                        page_number=page.page_number,
                        chunk_index=0,
                        text=page.clean_text,
                        token_count=count_tokens(page.clean_text, encoder),
                        word_count=len(page.clean_text.split()),
                        chunk_id_global=make_chunk_id_global(doc_id, chunk_id),
                        page_start=page.page_number,
                        page_end=page.page_number,
                        pages=[page.page_number],
                        part=_part,
                        section_title=section_title,
                        subsection_title=subsection_title,
                        is_table=True,
                        table_type=page.table_type,
                    )
                )
            continue
        _part, section_title, subsection_title = find_section_for_page(sections_df, page.page_number)
        token_chunks = _build_page_chunks(page.clean_text, config, encoder)
        for index, token_chunk in enumerate(token_chunks):
            word_count = len(token_chunk.text.split())
            if word_count < config.min_chunk_words:
                continue
            chunk_id = f"{doc_id}:p{page.page_number:04d}:c{index:03d}"
            if chunk_id in seen_chunk_ids:
                raise ValueError(f"Duplicate chunk id detected: {chunk_id}")
            seen_chunk_ids.add(chunk_id)
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page_number=page.page_number,
                    chunk_index=index,
                    text=token_chunk.text,
                    token_count=token_chunk.token_count,
                    word_count=word_count,
                    chunk_id_global=make_chunk_id_global(doc_id, chunk_id),
                    page_start=page.page_number,
                    page_end=page.page_number,
                    pages=[page.page_number],
                    part=_part,
                    section_title=section_title,
                    subsection_title=subsection_title,
                )
            )
        next_page = next_text_page_by_number.get(page.page_number)
        if next_page is not None:
            overlap_text = _build_cross_page_overlap_text(
                current_text=page.raw_text,
                next_text=next_page.raw_text,
                max_chars=320,
            )
            if overlap_text:
                word_count = len(overlap_text.split())
                if word_count >= config.min_chunk_words:
                    chunk_id = f"{doc_id}:p{page.page_number:04d}:x{len(token_chunks):03d}"
                    if chunk_id in seen_chunk_ids:
                        raise ValueError(f"Duplicate chunk id detected: {chunk_id}")
                    seen_chunk_ids.add(chunk_id)
                    chunks.append(
                        ChunkRecord(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            page_number=page.page_number,
                            chunk_index=len(token_chunks),
                            text=overlap_text,
                            token_count=count_tokens(overlap_text, encoder),
                            word_count=word_count,
                            chunk_id_global=make_chunk_id_global(doc_id, chunk_id),
                            page_start=page.page_number,
                            page_end=next_page.page_number,
                            pages=[page.page_number, next_page.page_number],
                            part=_part,
                            section_title=section_title,
                            subsection_title=subsection_title,
                        )
                    )
    if not chunks:
        raise ValueError(f"No chunks produced for {doc_id}")
    return chunks


def _build_legacy_table_chunks(
    *,
    doc_id: str,
    pages: list[PageRecord],
    config: ChunkingConfig,
    encoder: Any,
    sections_df: pd.DataFrame,
    source_pdf_path: str | Path | None,
) -> tuple[dict[int, list[ChunkRecord]], set[int]]:
    table_pages = [page for page in pages if page.is_table]
    if not table_pages or source_pdf_path is None:
        return {}, set()

    pdf_path = Path(source_pdf_path)
    table_page_payloads = [
        {
            "page": page.page_number,
            "text": page.clean_text,
            "raw_text": page.raw_text,
            "table_type": page.table_type,
            "extractor": page.extractor_used,
            "rotation": page.rotation,
            "page_width": page.page_width,
            "page_height": page.page_height,
            "is_table": page.is_table,
        }
        for page in table_pages
    ]
    with pdfplumber.open(pdf_path) as plumber_doc:
        table_chunks_df, _structured_tables_df, _table_facts_df, rejected_ocr_table_pages = process_table_pages(
            table_page_payloads,
            pdf_path,
            plumber_doc,
            doc_id,
            doc_id,
            None,
            None,
            None,
            "refactor-parity",
            encoder,
            chunk_size_tokens=config.chunk_size_tokens,
            table_chunking_strategy=str(config.table_chunking_strategy or "baseline"),
            return_all_tables=False,
            enable_secondary_bottom_pass=False,
        )

    chunks_by_page: dict[int, list[ChunkRecord]] = {}
    if len(table_chunks_df) > 0:
        for row in table_chunks_df.to_dict(orient="records"):
            page_no = int(row.get("page_start") or row.get("page_no") or row.get("page") or 0)
            if page_no < 1:
                raise ValueError(f"Legacy table chunk emitted invalid page number: {page_no}")
            chunk_text = str(row.get("chunk_text") or "").strip()
            if not chunk_text:
                continue
            raw_pages = row.get("pages")
            chunk_pages = list(raw_pages) if isinstance(raw_pages, list) else [page_no]
            _part, section_title, subsection_title = find_section_for_page(sections_df, page_no)
            chunks_by_page.setdefault(page_no, []).append(
                ChunkRecord(
                    chunk_id=str(row["chunk_id"]),
                    doc_id=doc_id,
                    page_number=page_no,
                    chunk_index=len(chunks_by_page.get(page_no, [])),
                    text=chunk_text,
                    token_count=int(row.get("chunk_tokens") or count_tokens(chunk_text, encoder)),
                    word_count=int(row.get("word_count") or len(chunk_text.split())),
                    chunk_id_global=str(row.get("chunk_id_global") or make_chunk_id_global(doc_id, str(row["chunk_id"]))),
                    page_start=page_no,
                    page_end=int(row.get("page_end") or page_no),
                    pages=chunk_pages,
                    part=_part,
                    section_title=section_title,
                    subsection_title=subsection_title,
                    is_table=True,
                    table_type=str(row.get("table_type")) if row.get("table_type") is not None else None,
                    table_chunk_kind=str(row.get("table_chunk_kind")) if row.get("table_chunk_kind") is not None else None,
                )
            )
    rejected_pages: set[int] = set()
    for record in rejected_ocr_table_pages:
        try:
            page_no = int(record.get("page") or 0)
        except (TypeError, ValueError):
            continue
        if page_no > 0:
            rejected_pages.add(page_no)
    return chunks_by_page, rejected_pages


def _validate_pages(pages: list[PageRecord]) -> None:
    seen: set[tuple[str, int]] = set()
    for page in pages:
        key = (page.doc_id, page.page_number)
        if key in seen:
            raise ValueError(f"Duplicate page record detected for {page.doc_id} page {page.page_number}")
        if page.page_number < 1:
            raise ValueError(f"Invalid page number: {page.page_number}")
        seen.add(key)


def _accepted_ocr_used(extractor_used: str, quality_note: str) -> bool:
    note = str(quality_note or "")
    extractor = str(extractor_used or "").strip().lower()
    return extractor == "ocr" or "ocr_raw_used" in note


def _classify_table_pages(
    page_structs: list[tuple[int, dict, str, str]],
    cleaned_map: dict[int, list[str]] | None,
) -> dict[int, tuple[bool, str | None]]:
    table_flags: dict[int, tuple[bool, str | None]] = {}
    for page_number, page_struct, _extractor_used, _quality_note in page_structs:
        raw_lines = [line.get("text", "") for line in page_struct.get("lines_all", [])]
        raw_table = is_table_like_from_raw_lines(raw_lines)
        drawings = page_struct.get("drawings", [])
        if is_graphics_table_like(drawings):
            raw_table = True
        if is_column_alignment_table_like(page_struct.get("lines_all", [])):
            raw_table = True
        cleaned_lines = cleaned_map.get(page_number, raw_lines) if cleaned_map is not None else raw_lines
        clean_text = normalize_page_text("\n".join(cleaned_lines))
        classification = classify_page_content(clean_text)
        is_table = bool(classification.get("is_table", False))
        table_type = classification.get("table_type")
        if raw_table and not is_table:
            is_table = True
            table_type = table_type or detect_table_type(clean_text)
        table_flags[page_number] = (is_table, table_type)
    return table_flags


def _normalize_boundary_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _page_ends_mid_sentence(text: str) -> bool:
    tail = _normalize_boundary_text(text).rstrip().rstrip("\"'”’)]}")
    return bool(tail) and tail[-1] not in ".!?:;"


def _extract_trailing_sentence_window(text: str, max_chars: int) -> str:
    normalized = _normalize_boundary_text(text)
    if not normalized:
        return ""
    if len(normalized) <= max_chars:
        return normalized
    start = max(0, len(normalized) - max_chars)
    boundary = max(normalized.rfind(mark, 0, start) for mark in (". ", "! ", "? ", "; ", ": "))
    if boundary >= 0:
        start = boundary + 2
    return normalized[start:].strip()


def _extract_leading_sentence_window(text: str, max_chars: int) -> str:
    normalized = _normalize_boundary_text(text)
    if not normalized:
        return ""
    normalized = re.sub(r"^[A-Z][A-Z0-9 ,/&()\\-]{6,}\s+", "", normalized).strip()
    if not normalized:
        return ""
    window = normalized[:max_chars].strip()
    if not window:
        return ""
    for index, char in enumerate(window):
        if char in ".!?":
            return window[: index + 1].strip()
    return window


def _build_cross_page_overlap_text(current_text: str, next_text: str, max_chars: int) -> str:
    if not _page_ends_mid_sentence(current_text):
        return ""
    trailing = _extract_trailing_sentence_window(current_text, max_chars)
    leading = _extract_leading_sentence_window(next_text, max_chars)
    if not trailing or not leading:
        return ""
    overlap_text = f"{trailing} {leading}".strip()
    if _normalize_boundary_text(overlap_text) == _normalize_boundary_text(trailing):
        return ""
    return overlap_text


def _build_page_chunks(text: str, config: ChunkingConfig, encoder) -> list:
    segments = split_text_for_segment_aware_chunking_with_patterns(
        text,
        insert_patterns=tuple(LEGACY_CONFIG.SEGMENT_BOUNDARY_INSERT_PATTERNS),
        boundary_match_patterns=tuple(LEGACY_CONFIG.SEGMENT_BOUNDARY_MATCH_PATTERNS),
        boundary_search_patterns=tuple(LEGACY_CONFIG.SEGMENT_BOUNDARY_SEARCH_PATTERNS),
        uppercase_heading_pattern=str(LEGACY_CONFIG.SEGMENT_UPPERCASE_HEADING_PATTERN),
        uppercase_heading_max_words=int(LEGACY_CONFIG.SEGMENT_UPPERCASE_HEADING_MAX_WORDS),
    )
    chunks = []
    for segment in segments:
        chunks.extend(
            chunk_text(
                segment.text,
                chunk_size=config.chunk_size_tokens,
                overlap=config.chunk_overlap_tokens,
                encoder=encoder,
            )
        )
    return chunks


def _extract_top_lines(lines_all: list[dict], k: int) -> list[dict]:
    if not lines_all:
        return []
    sorted_lines = sorted(
        lines_all,
        key=lambda line: (float(line.get("y0", 0.0)), float(line.get("x0", 0.0))),
    )
    top_lines: list[dict] = []
    for line in sorted_lines[:k]:
        text = str(line.get("text", "")).strip()
        if not text:
            continue
        top_lines.append(
            {
                "text": text,
                "y0": float(line.get("y0", 0.0)),
                "y1": float(line.get("y1", 0.0)),
            }
        )
    return top_lines


def _build_sections_dataframe(pages: list[PageRecord]) -> pd.DataFrame:
    rows = []
    for page in pages:
        rows.append(
            {
                "doc_id": page.doc_id,
                "report_year": None,
                "period_end_date": None,
                "report_year_source": None,
                "run_date_utc": None,
                "page": page.page_number,
                "clean_text": page.clean_text,
                "heading_candidates": page.heading_candidates,
                "top_lines": page.top_lines,
            }
        )
    return build_sections_from_pages(pd.DataFrame(rows))
