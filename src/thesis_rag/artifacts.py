from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import pandas as pd

from .schemas import ChunkRecord, PageRecord, QueryRecord, RetrievalHit
from .utils import read_json, write_json, write_jsonl


def save_pages(pages: list[PageRecord], out_dir: Path) -> Path:
    path = out_dir / "pages.jsonl"
    write_jsonl(path, [page.to_dict() for page in pages])
    pd.DataFrame([page.to_dict() for page in pages]).to_parquet(out_dir / "pages.parquet", index=False)
    return path


def save_chunks(chunks: list[ChunkRecord], out_dir: Path) -> Path:
    path = out_dir / "chunks.jsonl"
    records = [chunk.to_dict() for chunk in chunks]
    write_jsonl(path, records)
    pd.DataFrame(records).to_parquet(out_dir / "chunks.parquet", index=False)
    return path


def load_chunks(path: Path) -> list[ChunkRecord]:
    frame = pd.read_parquet(path)
    return [ChunkRecord(**row) for row in frame.to_dict(orient="records")]


def load_pages(path: Path) -> list[PageRecord]:
    frame = pd.read_parquet(path)
    return [PageRecord(**row) for row in frame.to_dict(orient="records")]


def save_queries(queries: list[QueryRecord], out_path: Path) -> None:
    write_json(out_path, {"queries": [query.to_dict() for query in queries]})


def load_queries(path: Path) -> list[QueryRecord]:
    payload = read_json(path)
    rows = payload["queries"] if isinstance(payload, dict) and "queries" in payload else payload
    queries: list[QueryRecord] = []
    for row in rows:
        queries.append(
            QueryRecord(
                query_id=row["query_id"],
                query_text=row.get("query_text") or row.get("question") or row.get("query"),
                doc_id=row["doc_id"],
                gold_pages=list(row.get("gold_pages") or row.get("expected_pages") or []),
                expected_answer=row.get("expected_answer"),
                difficulty=row.get("difficulty"),
                evidence_layout=row.get("evidence_layout"),
                expected_section=row.get("expected_section"),
                expected_subsection=row.get("expected_subsection"),
            )
        )
    return queries


def save_hits(hits: Iterable[RetrievalHit], out_path: Path) -> None:
    records = [hit.to_dict() for hit in hits]
    write_jsonl(out_path, records)
    pd.DataFrame(records).to_csv(out_path.with_suffix(".csv"), index=False)


def save_manifest(manifest: dict, out_path: Path) -> None:
    write_json(out_path, manifest)
