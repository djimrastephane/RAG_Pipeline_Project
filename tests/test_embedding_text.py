from thesis_rag.embedding import build_embedding_text
from thesis_rag.schemas import ChunkRecord


def test_build_embedding_text_matches_legacy_heading_prefix_rule() -> None:
    chunk = ChunkRecord(
        chunk_id="c1",
        doc_id="doc",
        page_number=1,
        chunk_index=0,
        text="body",
        token_count=1,
        word_count=1,
        section_title="SECTION",
        subsection_title="SUBSECTION",
        is_table=False,
    )
    assert build_embedding_text(chunk) == "SECTION\nSUBSECTION\nbody"

    table_chunk = ChunkRecord(
        chunk_id="c2",
        doc_id="doc",
        page_number=1,
        chunk_index=0,
        text="body",
        token_count=1,
        word_count=1,
        section_title="SECTION",
        subsection_title="SUBSECTION",
        is_table=True,
    )
    assert build_embedding_text(table_chunk) == "SECTION\nbody"
