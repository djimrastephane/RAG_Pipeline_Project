from thesis_rag.preprocessing import _accepted_ocr_used, _build_cross_page_overlap_text


def test_accepted_ocr_used_tracks_legacy_note() -> None:
    assert _accepted_ocr_used("ocr", "ok") is True
    assert _accepted_ocr_used("pymupdf", "fallback;ocr_raw_used") is True
    assert _accepted_ocr_used("pymupdf", "ok") is False


def test_cross_page_overlap_text_only_when_sentence_continues() -> None:
    text = _build_cross_page_overlap_text(
        "This page ends mid sentence and continues",
        "On the next page where the sentence ends.",
        320,
    )
    assert "continues On the next page" in text
    assert _build_cross_page_overlap_text("This page ends cleanly.", "Next page starts here.", 320) == ""
