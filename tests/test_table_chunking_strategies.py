from __future__ import annotations

import sys
from pathlib import Path
import unittest

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag_pdf.table_extract import _build_table_chunk_texts


class TableChunkingStrategyTests(unittest.TestCase):
    def test_row_preserving_does_not_split_rows(self) -> None:
        df = pd.DataFrame(
            [
                ["Payables", "100", "120"],
                ["Receivables", "30", "28"],
                ["Cash", "55", "61"],
            ],
            columns=["Line", "2024", "2025"],
        )
        chunks = _build_table_chunk_texts(
            strategy="row_preserving",
            page_no=10,
            table_summary="Financial table on page 10",
            raw_table=df,
            header_injected_facts="",
            table_markdown="",
            chunk_size_tokens=20,
            enc=None,
        )
        rows = ["Payables | 100 | 120", "Receivables | 30 | 28", "Cash | 55 | 61"]
        joined = "\n".join(chunks)
        for r in rows:
            self.assertIn(r, joined)
        # Row text should be intact (not split into fragments).
        self.assertNotIn("Payables | 100", joined.replace("Payables | 100 | 120", ""))

    def test_two_stage_has_header_and_repeated_headers_in_body(self) -> None:
        df = pd.DataFrame(
            [
                ["Staff costs", "100", "110"],
                ["Agency", "20", "15"],
                ["Other", "10", "9"],
            ],
            columns=["Category", "2024", "2025"],
        )
        chunks = _build_table_chunk_texts(
            strategy="two_stage",
            page_no=83,
            table_summary="Staff costs table",
            raw_table=df,
            header_injected_facts="",
            table_markdown="",
            chunk_size_tokens=30,
            enc=None,
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertIn("TABLE | page=83", chunks[0])
        self.assertIn("COLUMNS:", chunks[0])
        for body in chunks[1:]:
            self.assertIn("COLUMNS:", body)


if __name__ == "__main__":
    unittest.main()
