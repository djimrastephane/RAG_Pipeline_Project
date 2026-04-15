from thesis_rag.chunking import chunk_text, get_encoder


def test_token_chunking_with_overlap_is_stable() -> None:
    encoder = get_encoder()
    text = " ".join(f"token{i}" for i in range(120))
    chunks = chunk_text(text, chunk_size=30, overlap=10, encoder=encoder)
    assert len(chunks) >= 4
    assert chunks[0].token_count <= 30
    assert chunks[1].token_count <= 30
    overlap_words = chunks[0].text.split()[-5:]
    assert overlap_words
    assert " ".join(overlap_words) in chunks[1].text
