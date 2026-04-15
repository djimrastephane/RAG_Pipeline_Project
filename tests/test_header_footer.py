from thesis_rag.preprocessing import remove_repeated_headers_and_footers


def test_repeated_header_footer_lines_are_removed() -> None:
    pages = {
        1: ["Annual Report", "Body one", "Page 1"],
        2: ["Annual Report", "Body two", "Page 2"],
        3: ["Annual Report", "Body three", "Page 3"],
    }
    cleaned, headers, footers = remove_repeated_headers_and_footers(pages, top_k=1, bottom_k=1, repeat_fraction=0.6)
    assert "Annual Report" in headers
    assert cleaned[1] == ["Body one", "Page 1"]
    assert not footers
