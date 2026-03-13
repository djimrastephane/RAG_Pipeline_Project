from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ..chunking import get_encoder


class StorageService:
    """Read/write document artifacts for the retrieval UI."""
    DOC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,119}$")

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        # Default to main pipeline artifacts so UI reflects current pipeline state.
        # Override via UI_DATA_ROOT if needed.
        data_root_name = os.getenv("UI_DATA_ROOT", "data_processed")
        self.data_root = repo_root / data_root_name
        self.data_root.mkdir(parents=True, exist_ok=True)
        self._token_encoder = get_encoder()

    def _tokenize_for_display(self, text: str) -> list[str]:
        """Tokenize text for chunk-overlap inspection, preserving readable spacing."""
        value = str(text or "")
        if not value:
            return []
        if self._token_encoder is not None:
            try:
                token_ids = self._token_encoder.encode(value)
                return [self._token_encoder.decode([tid]) for tid in token_ids]
            except Exception:
                pass
        return re.findall(r"\S+\s*|\s+", value)

    @staticmethod
    def _shared_suffix_prefix_len(left: list[str], right: list[str]) -> int:
        """Return longest exact token overlap between left suffix and right prefix."""
        max_k = min(len(left), len(right))
        left_norm = [str(tok).strip() for tok in left]
        right_norm = [str(tok).strip() for tok in right]
        for k in range(max_k, 0, -1):
            if left_norm[-k:] == right_norm[:k]:
                return k
        return 0

    def doc_dir(self, doc_id: str) -> Path:
        """Return processed data directory for a document id."""
        raw = str(doc_id or "").strip()
        if not raw:
            raise ValueError("doc_id is required.")
        if not self.DOC_ID_RE.fullmatch(raw):
            raise ValueError("Invalid doc_id format.")
        root = self.data_root.resolve()
        target = (self.data_root / raw).resolve()
        try:
            target.relative_to(root)
        except Exception as e:
            raise ValueError("Invalid doc_id path.") from e
        return target

    def list_docs(self) -> list[str]:
        """List available UI-processed document ids."""
        if not self.data_root.exists():
            return []
        return sorted([p.name for p in self.data_root.iterdir() if p.is_dir()])

    @staticmethod
    def _clean_title(text: str, max_len: int = 140) -> str:
        """Normalize title text for compact display."""
        value = re.sub(r"\s+", " ", str(text or "")).strip()
        if len(value) <= max_len:
            return value
        return value[: max_len - 1].rstrip() + "…"

    def _read_doc_title(self, doc_id: str) -> str:
        """Best-effort extraction of a user-facing document title."""
        doc_dir = self.doc_dir(doc_id)
        metrics_path = doc_dir / "metrics.json"
        pages_path = doc_dir / "pages.parquet"
        report_year: Optional[str] = None

        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                raw_year = metrics.get("report_year")
                if raw_year:
                    report_year = str(raw_year).strip()
            except Exception:
                report_year = None

        title = str(doc_id)
        if pages_path.exists():
            try:
                pages_df = pd.read_parquet(pages_path, columns=["heading_candidates", "clean_text"])
                if len(pages_df):
                    row = pages_df.iloc[0]
                    heading_candidates = row.get("heading_candidates")
                    lines: list[str] = []
                    if isinstance(heading_candidates, (list, tuple)):
                        lines = [str(x).strip() for x in heading_candidates if str(x).strip()]
                    elif heading_candidates is not None:
                        try:
                            lines = [str(x).strip() for x in heading_candidates if str(x).strip()]
                        except Exception:
                            lines = []

                    if lines:
                        org = lines[0]
                        annual = next((ln for ln in lines if "ANNUAL REPORT" in ln.upper()), None)
                        if annual and annual.upper() != org.upper():
                            title = f"{org} — {annual}"
                        else:
                            title = org
                    else:
                        clean_text = str(row.get("clean_text") or "").strip()
                        if clean_text:
                            title = clean_text.split(".")[0]
            except Exception:
                title = str(doc_id)

        title = self._clean_title(title)
        if report_year and report_year not in title:
            title = self._clean_title(f"{title} ({report_year})")
        return title

    def list_docs_with_titles(self) -> list[dict[str, str]]:
        """List documents with display title for UI selectors."""
        docs = self.list_docs()
        out: list[dict[str, str]] = []
        for doc_id in docs:
            out.append({"doc_id": doc_id, "title": self._read_doc_title(doc_id)})
        return out

    def read_eval_items(self, doc_id: str) -> list[dict[str, Any]]:
        """Read eval_set.json for a document, or return empty list."""
        path = self.doc_dir(doc_id) / "eval_set.json"
        if not path.exists():
            return []
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        if isinstance(obj, dict):
            queries = obj.get("queries")
            if isinstance(queries, list):
                return [x for x in queries if isinstance(x, dict)]
        return []

    def read_doc_stats(self, doc_id: str) -> dict[str, Any]:
        """Compute page/chunk counts from parquet artifacts."""
        doc_dir = self.doc_dir(doc_id)
        pages_path = doc_dir / "pages.parquet"
        chunks_path = doc_dir / "chunks.parquet"
        if not pages_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Missing required artifacts under {doc_dir}")

        pages_df = pd.read_parquet(pages_path)
        chunks_df = pd.read_parquet(chunks_path)
        table_chunk_count = int(chunks_df["is_table"].fillna(False).sum()) if "is_table" in chunks_df.columns else 0
        tables_structured_path = doc_dir / "tables_structured.parquet"
        table_count = 0
        if tables_structured_path.exists():
            try:
                table_count = int(len(pd.read_parquet(tables_structured_path)))
            except Exception:
                table_count = 0

        chunk_size_tokens: Optional[int] = None
        chunk_overlap_tokens: Optional[int] = None
        segment_aware_chunking: Optional[bool] = None
        metrics_path = doc_dir / "metrics.json"
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                params = metrics.get("params", {}) if isinstance(metrics, dict) else {}
                if isinstance(params, dict):
                    raw_chunk_size = params.get("chunk_size_tokens")
                    raw_chunk_overlap = params.get("chunk_overlap_tokens")
                    if raw_chunk_size is not None:
                        chunk_size_tokens = int(raw_chunk_size)
                    if raw_chunk_overlap is not None:
                        chunk_overlap_tokens = int(raw_chunk_overlap)
                    raw_segment_aware = params.get("segment_aware_chunking")
                    if raw_segment_aware is not None:
                        segment_aware_chunking = bool(raw_segment_aware)
            except Exception:
                # Keep stats endpoint resilient even when metrics.json is malformed.
                chunk_size_tokens = None
                chunk_overlap_tokens = None
                segment_aware_chunking = None

        return {
            "doc_id": doc_id,
            "data_dir": str(doc_dir),
            "page_count": int(len(pages_df)),
            "chunk_count": int(len(chunks_df)),
            "table_chunk_count": int(table_chunk_count),
            "table_count": int(table_count),
            "chunk_size_tokens": chunk_size_tokens,
            "chunk_overlap_tokens": chunk_overlap_tokens,
            "segment_aware_chunking": segment_aware_chunking,
            "has_eval_set": bool((doc_dir / "eval_set.json").exists()),
            "has_pipeline_log": bool((doc_dir / "ui_pipeline.log").exists()),
            "has_tables_structured": bool(tables_structured_path.exists()),
        }

    def read_doc_log_tail(self, doc_id: str, last_n: int = 200) -> dict[str, Any]:
        """Return last N lines from the UI pipeline log for one document."""
        path = self.doc_dir(doc_id) / "ui_pipeline.log"
        if not path.exists():
            return {"doc_id": doc_id, "has_log": False, "log_tail": ""}
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-max(1, int(last_n)) :])
        return {"doc_id": doc_id, "has_log": True, "log_tail": tail}

    def read_tables_structured(
        self,
        doc_id: str,
        limit: int = 200,
        page: Optional[int] = None,
    ) -> dict[str, Any]:
        """Read extracted tables for a document from tables_structured.parquet."""
        doc_dir = self.doc_dir(doc_id)
        path = doc_dir / "tables_structured.parquet"
        if not path.exists():
            return {"doc_id": doc_id, "has_tables": False, "items": [], "total": 0}

        df = pd.read_parquet(path)
        if page is not None and "page" in df.columns:
            df = df[df["page"].astype("Int64") == int(page)]

        total = int(len(df))
        if limit > 0:
            df = df.head(int(limit))

        cols = [
            "page",
            "table_id",
            "table_type",
            "rows",
            "cols",
            "extraction_method",
            "table_summary",
            "table_markdown",
        ]
        present = [c for c in cols if c in df.columns]
        items = df[present].fillna("").to_dict(orient="records")
        return {
            "doc_id": doc_id,
            "has_tables": True,
            "items": items,
            "total": total,
        }

    def read_page_chunk_inspector(self, doc_id: str, page: int) -> dict[str, Any]:
        """Return one page with chunk tokenization and inferred overlap metadata."""
        doc_dir = self.doc_dir(doc_id)
        page_no = int(page)
        pages_path = doc_dir / "pages.parquet"
        chunks_path = doc_dir / "chunks.parquet"
        metrics_path = doc_dir / "metrics.json"
        if not pages_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(f"Missing required artifacts under {doc_dir}")

        pages_df = pd.read_parquet(pages_path)
        page_df = pages_df[pages_df["page"].astype("Int64") == page_no]
        if page_df.empty:
            raise FileNotFoundError(f"Unknown page {page_no} for {doc_id}")
        page_row = page_df.iloc[0]

        chunks_df = pd.read_parquet(chunks_path)
        page_chunks_df = chunks_df[
            (chunks_df["page_start"].astype("Int64") <= page_no)
            & (chunks_df["page_end"].astype("Int64") >= page_no)
        ].copy()
        if "chunk_id" in page_chunks_df.columns:
            page_chunks_df = page_chunks_df.sort_values(["chunk_id"]).reset_index(drop=True)
        else:
            page_chunks_df = page_chunks_df.reset_index(drop=True)

        params: dict[str, Any] = {}
        if metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                raw_params = metrics.get("params", {}) if isinstance(metrics, dict) else {}
                if isinstance(raw_params, dict):
                    params = raw_params
            except Exception:
                params = {}

        chunk_rows: list[dict[str, Any]] = []
        token_lists: list[list[str]] = []
        for _, row in page_chunks_df.iterrows():
            token_list = self._tokenize_for_display(str(row.get("chunk_text") or ""))
            token_lists.append(token_list)
            chunk_rows.append(
                {
                    "chunk_id": str(row.get("chunk_id") or ""),
                    "chunk_id_global": str(row.get("chunk_id_global") or ""),
                    "segment_title": str(row.get("segment_title") or ""),
                    "section_title": str(row.get("section_title") or ""),
                    "subsection_title": str(row.get("subsection_title") or ""),
                    "is_table": bool(row.get("is_table", False)),
                    "table_type": row.get("table_type"),
                    "chunk_tokens": int(row.get("chunk_tokens") or len(token_list)),
                    "token_display_count": int(len(token_list)),
                    "chunk_text": str(row.get("chunk_text") or ""),
                    "tokens": token_list,
                }
            )

        overlaps_next: list[int] = []
        for i in range(len(token_lists)):
            if i + 1 >= len(token_lists):
                overlaps_next.append(0)
            else:
                overlaps_next.append(self._shared_suffix_prefix_len(token_lists[i], token_lists[i + 1]))

        inferred_start = 0
        for i, row in enumerate(chunk_rows):
            token_count = len(token_lists[i])
            prev_overlap = overlaps_next[i - 1] if i > 0 else 0
            if i == 0:
                inferred_start = 0
            else:
                inferred_start = max(0, inferred_start + len(token_lists[i - 1]) - prev_overlap)
            inferred_end = inferred_start + token_count
            row["overlap_prev_tokens"] = int(prev_overlap)
            row["overlap_next_tokens"] = int(overlaps_next[i])
            row["token_start"] = int(inferred_start)
            row["token_end"] = int(inferred_end)

        page_text = str(page_row.get("clean_text") or "")
        page_tokens = self._tokenize_for_display(page_text)
        live_tokenizer_backend = "tiktoken" if self._token_encoder is not None else "display_fallback"
        artifact_tokenizer_backend = str(
            params.get("tokenizer_backend") or live_tokenizer_backend
        )
        return {
            "doc_id": doc_id,
            "page": page_no,
            "page_text": page_text,
            "page_token_count": int(len(page_tokens)),
            "page_is_table": bool(page_row.get("is_table", False)),
            "page_table_type": page_row.get("table_type"),
            "chunk_count": int(len(chunk_rows)),
            "artifact_tokenizer_backend": artifact_tokenizer_backend,
            "artifact_tokenizer_exact_counting": bool(
                params.get("tokenizer_exact_counting", self._token_encoder is not None)
            ),
            "inspector_tokenizer_backend": live_tokenizer_backend,
            "inspector_tokenizer_exact_counting": bool(self._token_encoder is not None),
            "chunk_size_tokens": params.get("chunk_size_tokens"),
            "chunk_overlap_tokens": params.get("chunk_overlap_tokens"),
            "segment_aware_chunking": params.get("segment_aware_chunking"),
            "chunks": chunk_rows,
        }
