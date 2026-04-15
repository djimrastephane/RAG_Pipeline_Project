from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Optional

import numpy as np
import pandas as pd


TRUST_CANONICAL_MAP: dict[str, str] = {
    "ayrshire and arran": "NHS Ayrshire & Arran",
    "borders": "NHS Borders",
    "dumfries & galloway": "NHS Dumfries & Galloway",
    "dumfries and galloway": "NHS Dumfries & Galloway",
    "fife": "NHS Fife",
    "forth valley": "NHS Forth Valley",
    "grampian": "NHS Grampian",
    "greater glasgow & clyde": "NHS Greater Glasgow & Clyde",
    "greater glasgow and clyde": "NHS Greater Glasgow & Clyde",
    "highland": "NHS Highland",
    "lanarkshire": "NHS Lanarkshire",
    "lothian": "NHS Lothian",
    "orkney": "NHS Orkney",
    "shetland": "NHS Shetland",
    "tayside": "NHS Tayside",
    "western isles": "NHS Western Isles",
}


def file_signature(path: Path) -> tuple[bool, int, int]:
    """Return a minimal stable signature tuple for cache invalidation."""
    if not path.exists():
        return (False, 0, 0)
    st = path.stat()
    return (True, int(st.st_size), int(st.st_mtime_ns))


def doc_artifact_signature(data_dir: Path) -> tuple[Any, ...]:
    """Signature for per-document retrieval artifacts."""
    return (
        file_signature(data_dir / "faiss.index"),
        file_signature(data_dir / "chunk_meta.parquet"),
        file_signature(data_dir / "chunks.parquet"),
        file_signature(data_dir / "eval_set.json"),
    )


def global_artifact_signature(data_root: Path) -> tuple[Any, ...]:
    """Signature for all global-ready docs under data_root."""
    docs = [p for p in sorted(data_root.iterdir()) if p.is_dir()]
    sig_items: list[tuple[str, tuple[Any, ...]]] = []
    for d in docs:
        sig_items.append(
            (
                d.name,
                (
                    file_signature(d / "embeddings.npy"),
                    file_signature(d / "chunk_meta.parquet"),
                    file_signature(d / "chunks.parquet"),
                ),
            )
        )
    return tuple(sig_items)


def trust_from_doc_id(doc_id: str) -> str:
    """
    Derive canonical trust id from document id.

    Examples:
    - 'Grampian-2023-2024' -> 'NHS Grampian'
    - 'NHS Shetland-2022-2023' -> 'NHS Shetland'
    """
    d = str(doc_id or "").strip()
    if not d:
        return ""
    base = d.split("-", 1)[0].strip()
    norm = base.lower().replace("nhs ", "").replace("_", " ")
    norm = re.sub(r"\s+", " ", norm).strip()
    return TRUST_CANONICAL_MAP.get(norm, f"NHS {base.replace('_', ' ').strip()}")


def read_eval_items(eval_path: Path) -> list[dict[str, Any]]:
    if not eval_path.exists():
        return []
    raw_eval = json.loads(eval_path.read_text(encoding="utf-8"))
    if isinstance(raw_eval, list):
        return [x for x in raw_eval if isinstance(x, dict)]
    if isinstance(raw_eval, dict):
        queries = raw_eval.get("queries")
        if isinstance(queries, list):
            return [x for x in queries if isinstance(x, dict)]
    return []


def apply_filters(meta: pd.DataFrame, filters: Optional[dict[str, Any]]) -> np.ndarray:
    """
    Build a boolean mask from metadata filters.

    Supported filter keys:
    - doc_id (str)
    - trust_id (str)
    - year (int)
    - is_table (bool)
    - section_contains (str)
    - subsection_contains (str)
    """
    if meta.empty:
        return np.zeros((0,), dtype=bool)
    mask = np.ones((len(meta),), dtype=bool)
    if not filters:
        return mask

    doc_id = str(filters.get("doc_id") or "").strip()
    if doc_id and "doc_id" in meta.columns:
        mask &= (meta["doc_id"].astype(str).values == doc_id)

    trust_id = str(filters.get("trust_id") or "").strip().lower()
    if trust_id and "trust_id" in meta.columns:
        mask &= (meta["trust_id"].astype(str).str.lower().values == trust_id)

    year_val = filters.get("year")
    if year_val is not None:
        if "year" in meta.columns:
            try:
                y = int(year_val)
                mask &= (pd.to_numeric(meta["year"], errors="coerce").fillna(-1).astype(int).values == y)
            except Exception:
                pass
        elif "report_year" in meta.columns:
            try:
                y = int(year_val)
                mask &= (pd.to_numeric(meta["report_year"], errors="coerce").fillna(-1).astype(int).values == y)
            except Exception:
                pass

    is_table = filters.get("is_table")
    if is_table is not None and "is_table" in meta.columns:
        want = bool(is_table)
        cur = meta["is_table"].fillna(False).astype(bool).values
        mask &= (cur == want)

    sec = str(filters.get("section_contains") or "").strip()
    if sec and "section_title" in meta.columns:
        mask &= meta["section_title"].fillna("").astype(str).str.contains(sec, case=False, regex=False).values

    sub = str(filters.get("subsection_contains") or "").strip()
    if sub and "subsection_title" in meta.columns:
        mask &= meta["subsection_title"].fillna("").astype(str).str.contains(sub, case=False, regex=False).values

    return mask

