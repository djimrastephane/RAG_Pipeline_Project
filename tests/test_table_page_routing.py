from thesis_rag.preprocessing import build_chunk_records
from thesis_rag.schemas import ChunkRecord, PageRecord


class _Cfg:
    chunk_size_tokens = 224
    chunk_overlap_tokens = 56
    min_chunk_words = 1


def test_table_pages_emit_single_table_chunk_and_skip_text_chunking() -> None:
    pages = [
        PageRecord(
            page_id="doc:p0001",
            doc_id="doc",
            page_number=1,
            raw_text="narrative page ends mid sentence",
            clean_text="narrative page ends mid sentence",
            extractor_used="pymupdf",
            quality_note="ok",
            ocr_used=False,
            is_table=False,
        ),
        PageRecord(
            page_id="doc:p0002",
            doc_id="doc",
            page_number=2,
            raw_text="table row one table row two",
            clean_text="table row one table row two",
            extractor_used="pymupdf",
            quality_note="ok",
            ocr_used=False,
            is_table=True,
        ),
    ]
    chunks = build_chunk_records("doc", pages, _Cfg())
    assert any(chunk.chunk_id == "table_p0002" for chunk in chunks)
    assert not any(chunk.page_number == 2 and chunk.chunk_id.startswith("doc:p0002:c") for chunk in chunks)


def test_table_pages_use_legacy_table_chunk_text_when_available(monkeypatch) -> None:
    pages = [
        PageRecord(
            page_id="doc:p0002",
            doc_id="doc",
            page_number=2,
            raw_text="raw table text",
            clean_text="flat fallback text",
            extractor_used="pymupdf",
            quality_note="ok",
            ocr_used=False,
            is_table=True,
            table_type="cash_flow",
        ),
    ]

    def _fake_table_chunks(**_kwargs):
        return (
            {
                2: [
                    ChunkRecord(
                        chunk_id="table_p0002",
                        doc_id="doc",
                        page_number=2,
                        chunk_index=0,
                        text="Financial table on page 2. Structured summary.",
                        token_count=7,
                        word_count=7,
                        page_start=2,
                        page_end=2,
                        pages=[2],
                        section_title="SECTION",
                        subsection_title="SUBSECTION",
                        is_table=True,
                        table_type="cash_flow",
                        table_chunk_kind="full_table",
                    )
                ]
            },
            set(),
        )

    monkeypatch.setattr("thesis_rag.preprocessing._build_legacy_table_chunks", _fake_table_chunks)

    chunks = build_chunk_records("doc", pages, _Cfg(), source_pdf_path="dummy.pdf")

    assert len(chunks) == 1
    assert chunks[0].text == "Financial table on page 2. Structured summary."
    assert chunks[0].table_chunk_kind == "full_table"
    assert chunks[0].table_type == "cash_flow"
