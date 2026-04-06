from __future__ import annotations

import pandas as pd


def results_to_dataframe(results: list[dict]) -> pd.DataFrame:
    """Convert API search results list into a flat DataFrame for display."""
    rows = []
    for r in results:
        rows.append(
            {
                "rank": r.get("rank"),
                "chunk_id": r.get("chunk_id"),
                "pages": ", ".join(str(p) for p in r.get("pages", [])),
                "score": r.get("score"),
                "rrf_score": r.get("rrf_score"),
                "dense_rank": r.get("dense_rank"),
                "bm25_rank": r.get("bm25_rank"),
                "dense_raw_score": r.get("dense_raw_score"),
                "bm25_raw_score": r.get("bm25_raw_score"),
                "table_chunk_kind": r.get("table_chunk_kind"),
                "row_start_idx": r.get("row_start_idx"),
                "row_end_idx": r.get("row_end_idx"),
                "section_title": r.get("section_title", ""),
                "subsection_title": r.get("subsection_title", ""),
                "hit_expected_page": r.get("hit_expected_page", False),
                "snippet": r.get("snippet", ""),
            }
        )
    return pd.DataFrame(rows)
