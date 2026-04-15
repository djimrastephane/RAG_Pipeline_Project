from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber
import pymupdf as fitz

from rag_pdf.extract_page import extract_page_struct_hybrid

from .schemas import DocumentRecord

LOGGER = logging.getLogger(__name__)


def discover_documents(data_dir: Path) -> list[DocumentRecord]:
    pdf_paths = sorted(data_dir.glob("*.pdf"))
    return [DocumentRecord(doc_id=path.stem, pdf_path=str(path)) for path in pdf_paths]


def extract_page_structures(document: DocumentRecord) -> list[tuple[int, dict, str, str]]:
    pdf_path = Path(document.pdf_path)
    with fitz.open(pdf_path) as fitz_doc, pdfplumber.open(pdf_path) as plumber_doc:
        pages: list[tuple[int, dict, str, str]] = []
        for page_index in range(fitz_doc.page_count):
            page_struct, extractor_used, quality_note = extract_page_struct_hybrid(
                fitz_doc,
                plumber_doc,
                page_index,
                pdf_path=str(pdf_path),
            )
            try:
                page_struct["drawings"] = fitz_doc.load_page(page_index).get_drawings()
            except Exception:
                page_struct["drawings"] = []
            pages.append((page_index + 1, page_struct, extractor_used, quality_note))
    LOGGER.info("Extracted %s pages from %s", len(pages), document.doc_id)
    return pages
