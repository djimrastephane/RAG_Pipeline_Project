"""
build_index.py

Build embeddings + FAISS index from chunks.parquet produced by preprocess_pdf_rag.py.

Batch mode:
- Scans BASE_DATA_DIR for document folders.
- For each folder containing chunks.parquet, builds:
  - faiss.index
  - embeddings.npy
  - chunk_meta.parquet
  - updates metrics.json with embedding/index details

Key requirements supported by this script
- Metadata row order matches FAISS row order.
  This guarantees FAISS row id i maps to chunk_meta row i.

- Multi-document safety.
  If chunks.parquet includes chunk_id_global, chunk_meta.parquet will include it.
  Retrieval evaluation can then log globally unique chunk ids.

- Page-level grounding for evaluation.
  chunk_meta.parquet will include a canonical pages field (list[int]).
  It will also include page_start and page_end.
  page_list is kept as a structured compatibility field when available.

Outputs used later
- Retrieval: embed query, search FAISS, map returned ids to chunk_id_global or chunk_id plus pages/page_start/page_end
- Evaluation: compute Recall@k, MRR, and failure attribution using expected_pages vs retrieved pages
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd

import faiss
from transformers import logging as hf_logging
from sentence_transformers import SentenceTransformer


# =============================================================================
# CONFIG
# =============================================================================
BASE_DATA_DIR = Path(
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed"
)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_WITH_SUBSECTION = os.getenv("EMBED_WITH_SUBSECTION", "1") != "0"

FAISS_INDEX_NAME = "faiss.index"
EMB_NPY_NAME = "embeddings.npy"
META_PARQUET_NAME = "chunk_meta.parquet"
CHUNKS_FILENAME = "chunks.parquet"
METRICS_FILENAME = "metrics.json"

TOPK_DEFAULT = 5

# Artifact scan guardrail
ARTIFACT_SCAN_MAX_CHUNKS = 2000  # set to None to scan all chunks
FAIL_ON_ARTIFACTS = True

# If True, overwrite existing index artifacts
OVERWRITE_EXISTING = True


# =============================================================================
# ARTIFACT CONSTANTS
# =============================================================================
ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\ufeff"}
LIGATURE_GLYPHS = {"\ufb00", "\ufb01", "\ufb02", "\ufb03", "\ufb04"}


# =============================================================================
# HELPERS
# =============================================================================
def utc_now_iso() -> str:
    """
    Returns current UTC time in ISO-8601 format.
    Used to timestamp metrics for traceability.
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2 normalisation.

    Why:
    - FAISS IndexFlatIP performs inner-product search.
    - If vectors are L2 normalised, inner product equals cosine similarity.
    """
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (norms + eps)


def write_json(path: Path, obj: dict) -> None:
    """
    Writes a JSON file with UTF-8 encoding and creates directories if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> dict:
    """
    Reads a JSON file if it exists, else returns an empty dict.
    """
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _env_or_default(name: str, default: str) -> str:
    val = os.getenv(name)
    return val if val else default


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build embeddings + FAISS index from chunks.parquet."
    )
    parser.add_argument(
        "--data-dir",
        default=_env_or_default("DATA_DIR", str(BASE_DATA_DIR)),
        help="Base directory containing per-document folders.",
    )
    parser.add_argument(
        "--model",
        default=_env_or_default("EMBED_MODEL_NAME", EMBED_MODEL_NAME),
        help="Sentence-transformers model name or local path.",
    )
    return parser.parse_args()


def describe_array(x: np.ndarray) -> dict[str, Any]:
    """
    Simple numeric summary for sanity checks and metrics logging.
    """
    return {
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def find_text_artifacts(text: str) -> dict[str, int]:
    """
    Detect common PDF extraction artifacts that should not reach the embedding stage.

    Returns counts:
    - U+00AD soft hyphen
    - '￾' common extraction artifact
    - zero-width characters
    - typographic ligature glyphs
    """
    return {
        "soft_hyphen_u00ad": text.count("\u00ad"),
        "artifact_fff0": text.count("￾"),
        "zero_width_total": sum(text.count(ch) for ch in ZERO_WIDTH),
        "ligature_total": sum(text.count(ch) for ch in LIGATURE_GLYPHS),
    }


def _clean_heading_value(val: Any) -> str:
    if val is None:
        return ""
    if isinstance(val, float) and pd.isna(val):
        return ""
    s = str(val).strip()
    if not s or s.lower() == "unknown":
        return ""
    return s


def build_embedding_text(row: pd.Series) -> str:
    section = _clean_heading_value(row.get("section_title"))
    subsection = _clean_heading_value(row.get("subsection_title"))
    chunk_text = str(row.get("chunk_text", "") or "")
    is_table = bool(row.get("is_table", False))

    parts: list[str] = []
    if section:
        parts.append(section)
    if EMBED_WITH_SUBSECTION and not is_table and subsection:
        parts.append(subsection)
    parts.append(chunk_text)
    return "\n".join(parts)


def _to_int_if_whole(x: Any) -> Optional[int]:
    """
    Convert a value into an integer when it represents a whole number.

    Purpose:
        Parquet readers and transforms can change numeric types. Page fields may
        appear as int, float (2.0), or str ("2"). This helper normalises such
        values so downstream logic can treat page numbers as plain integers.

    Args:
        x (Any): Candidate numeric value.

    Returns:
        Optional[int]: Integer when conversion is safe and exact, else None.
    """
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        if math.isfinite(x) and float(x).is_integer():
            return int(x)
        return None
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        if s.isdigit():
            return int(s)
        try:
            f = float(s)
            if math.isfinite(f) and f.is_integer():
                return int(f)
        except Exception:
            return None
    return None


def normalize_pages_value(v: Any) -> list[int]:
    """
    Normalise a pages-like value into list[int].

    Supported inputs:
    - list[int]
    - tuple[int]
    - numpy array
    - scalar
    - stringified lists
    - list of dicts like [{"element": 2}, {"element": 3}]
    - stringified list of dicts like '[{"element":2}]'

    Returns:
        list[int]: Sorted unique pages, empty list if not parseable.
    """
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []

    if isinstance(v, (list, tuple)):
        out: list[int] = []
        for x in v:
            if x is None or (isinstance(x, float) and pd.isna(x)):
                continue
            if isinstance(x, dict):
                if "element" in x:
                    iv = _to_int_if_whole(x.get("element"))
                    if iv is not None:
                        out.append(iv)
                continue
            iv = _to_int_if_whole(x)
            if iv is not None:
                out.append(iv)
        return sorted(set(out))

    if hasattr(v, "tolist"):
        try:
            vv = v.tolist()
            return normalize_pages_value(vv)
        except Exception:
            return []

    if isinstance(v, str):
        s = v.strip()
        if not s:
            return []
        nums = re.findall(r"\d+", s)
        return sorted(set(int(n) for n in nums)) if nums else []

    iv = _to_int_if_whole(v)
    return [iv] if iv is not None else []


def build_pages_from_span(page_start: Any, page_end: Any) -> list[int]:
    """
    Build pages list from page_start/page_end.

    Returns:
        list[int]: Inclusive range between start and end when valid, else [].
    """
    ps = _to_int_if_whole(page_start)
    pe = _to_int_if_whole(page_end)
    if ps is None or pe is None:
        return []
    if ps <= pe:
        return list(range(ps, pe + 1))
    return list(range(pe, ps + 1))


def build_page_list_struct(pages: list[int]) -> list[dict]:
    """
    Build structured page list field for compatibility with earlier viewers.

    Output form:
        [{"element": 2}, {"element": 3}]
    """
    return [{"element": int(p)} for p in pages]


def build_meta_table(chunks: pd.DataFrame) -> pd.DataFrame:
    """
    Build a metadata table aligned to FAISS index rows.

    This function guarantees:
    - chunk_id exists
    - pages exists as list[int]
    - page_start/page_end exist when present in input
    - chunk_id_global is preserved when present in input

    Why:
        Retrieval evaluation needs stable document and page grounding.
        Multi-document indexing needs globally unique chunk ids.

    Output columns are ordered and minimal to keep chunk_meta.parquet lightweight.
    """
    if "chunk_id" not in chunks.columns:
        raise ValueError("chunks.parquet must include 'chunk_id'.")

    preferred_cols = [
        "chunk_id",
        "chunk_id_global",
        "doc_id",
        "corpus_id",
        "report_year",
        "report_year_source",
        "period_end_date",
        "run_date_utc",
        "part",
        "section_title",
        "subsection_title",
        "is_table",
        "page_start",
        "page_end",
        "pages",
        "page_list",
    ]

    cols = [c for c in preferred_cols if c in chunks.columns]
    meta = chunks[cols].copy()

    # Ensure page_start/page_end exist if present in source.
    # If missing, attempt to derive from pages.
    if "pages" in meta.columns:
        meta["pages"] = meta["pages"].apply(normalize_pages_value)
    elif "page_list" in meta.columns:
        # If pages is missing but page_list exists, derive pages from it.
        meta["pages"] = meta["page_list"].apply(normalize_pages_value)
    else:
        # Create pages later from span if possible.
        meta["pages"] = [[] for _ in range(len(meta))]

    # Normalize page_start and page_end where possible.
    if "page_start" in meta.columns and "page_end" in meta.columns:
        meta["page_start"] = meta["page_start"].apply(_to_int_if_whole)
        meta["page_end"] = meta["page_end"].apply(_to_int_if_whole)

        # Fill pages from span when pages is empty.
        def _fill_pages_from_span(row) -> list[int]:
            pages = row["pages"]
            if isinstance(pages, list) and len(pages) > 0:
                return sorted(set(int(p) for p in pages))
            return build_pages_from_span(row.get("page_start"), row.get("page_end"))

        meta["pages"] = meta.apply(_fill_pages_from_span, axis=1)
    else:
        # If page_start/page_end not present, try to derive from pages.
        if "page_start" not in meta.columns:
            meta["page_start"] = meta["pages"].apply(lambda ps: int(min(ps)) if ps else None)
        if "page_end" not in meta.columns:
            meta["page_end"] = meta["pages"].apply(lambda ps: int(max(ps)) if ps else None)

    # Keep page_list as compatibility field.
    # If the input has it but it is nested or stringified, normalise it to list[dict].
    # If input does not have it, create it from pages.
    if "page_list" in meta.columns:
        def _norm_page_list(v: Any, pages: list[int]) -> list[dict]:
            # If v already looks like a list of dicts, keep it.
            if isinstance(v, list) and all(isinstance(x, dict) for x in v):
                if all(("element" in x) for x in v):
                    return [{"element": int(_to_int_if_whole(x.get("element")) or 0)} for x in v if _to_int_if_whole(x.get("element")) is not None]
            # Otherwise, rebuild from pages.
            return build_page_list_struct(pages)

        meta["page_list"] = [
            _norm_page_list(v, pages)
            for v, pages in zip(meta["page_list"].tolist(), meta["pages"].tolist())
        ]
    else:
        meta["page_list"] = meta["pages"].apply(build_page_list_struct)

    # Final column order
    out_cols = [
        "chunk_id",
        "chunk_id_global",
        "doc_id",
        "corpus_id",
        "report_year",
        "report_year_source",
        "period_end_date",
        "run_date_utc",
        "part",
        "section_title",
        "subsection_title",
        "is_table",
        "page_start",
        "page_end",
        "pages",
        "page_list",
    ]
    out_cols = [c for c in out_cols if c in meta.columns]
    meta = meta[out_cols].copy()

    return meta


def should_skip_doc(doc_dir: Path) -> tuple[bool, str]:
    """
    Determine whether a document folder should be skipped.

    Skips when:
    - chunks.parquet is missing
    - overwrite is disabled and index artifacts already exist
    """
    chunks_path = doc_dir / CHUNKS_FILENAME
    if not chunks_path.exists():
        return True, "missing_chunks_parquet"

    if not OVERWRITE_EXISTING:
        idx_path = doc_dir / FAISS_INDEX_NAME
        emb_path = doc_dir / EMB_NPY_NAME
        meta_path = doc_dir / META_PARQUET_NAME
        if idx_path.exists() and emb_path.exists() and meta_path.exists():
            return True, "artifacts_exist_overwrite_false"

    return False, "ok"


def iter_document_dirs(base_dir: Path) -> list[Path]:
    """
    Return all immediate subdirectories of base_dir.
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"BASE_DATA_DIR not found: {base_dir}")
    return sorted([d for d in base_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower())


# =============================================================================
# PER-DOCUMENT BUILD
# =============================================================================
def build_index_for_doc(doc_dir: Path, model: SentenceTransformer) -> None:
    """
    Build embeddings and FAISS index for one processed document folder.

    Guarantees:
    - chunk_meta.parquet includes chunk_id_global when available
    - chunk_meta.parquet includes pages as list[int]
    - embeddings.npy row order matches chunk_meta.parquet row order
    - faiss.index row order matches chunk_meta.parquet row order
    """
    chunks_path = doc_dir / CHUNKS_FILENAME
    metrics_path = doc_dir / METRICS_FILENAME

    chunks = pd.read_parquet(chunks_path)

    required_cols = {"chunk_id", "chunk_text"}
    missing = required_cols - set(chunks.columns)
    if missing:
        raise ValueError(f"{doc_dir.name}: chunks.parquet missing columns: {missing}")

    texts = [build_embedding_text(row) for _, row in chunks.iterrows()]
    if not texts or all(len(t.strip()) == 0 for t in texts):
        raise ValueError(f"{doc_dir.name}: No chunk_text found to embed.")

    # Artifact scan guardrail
    scan_texts = texts if ARTIFACT_SCAN_MAX_CHUNKS is None else texts[: int(ARTIFACT_SCAN_MAX_CHUNKS)]
    artifact_counts = {"soft_hyphen_u00ad": 0, "artifact_fff0": 0, "zero_width_total": 0, "ligature_total": 0}
    for t in scan_texts:
        c = find_text_artifacts(t)
        for k, v in c.items():
            artifact_counts[k] += v

    if any(v > 0 for v in artifact_counts.values()):
        msg = f"{doc_dir.name}: Text artifacts detected in chunk_text. Counts: {artifact_counts}"
        if FAIL_ON_ARTIFACTS:
            raise ValueError(msg)
        print("WARNING:", msg)

    # Metadata aligned with index rows
    meta = build_meta_table(chunks)

    # Load preprocess metrics for traceability
    preprocess_metrics = read_json(metrics_path)
    preprocess_schema = preprocess_metrics.get("schema_version")
    preprocess_norm = preprocess_metrics.get("params", {}).get("final_text_normalization", {})

    # Compute embeddings
    print(f"{doc_dir.name}: Embedding {len(texts)} chunks...")
    emb = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ).astype("float32")
    emb = l2_normalize(emb).astype("float32")
    n, d = emb.shape

    # Build FAISS index
    index = faiss.IndexFlatIP(d)
    index.add(emb)

    # Save artifacts
    index_path = doc_dir / FAISS_INDEX_NAME
    emb_path = doc_dir / EMB_NPY_NAME
    meta_path = doc_dir / META_PARQUET_NAME

    faiss.write_index(index, str(index_path))
    np.save(emb_path, emb)
    meta.to_parquet(meta_path, index=False)

    print(f"{doc_dir.name}: Saved FAISS index: {index_path}")
    print(f"{doc_dir.name}: Saved embeddings: {emb_path}")
    print(f"{doc_dir.name}: Saved metadata: {meta_path}")

    # Update metrics.json
    metrics_update = {
        "embedding": {
            "created_utc": utc_now_iso(),
            "model": EMBED_MODEL_NAME,
            "chunks_embedded": int(n),
            "embedding_dim": int(d),
            "faiss_index_type": "IndexFlatIP",
            "normalised_for_cosine": True,
            "topk_default": TOPK_DEFAULT,
            "embedding_summary": describe_array(emb),
            "chunk_meta_schema": {
                "columns": list(meta.columns),
                "has_chunk_id_global": bool("chunk_id_global" in meta.columns),
                "has_pages": bool("pages" in meta.columns),
                "sample_chunk_id": str(meta["chunk_id"].iloc[0]) if len(meta) else "",
                "sample_chunk_id_global": str(meta["chunk_id_global"].iloc[0]) if "chunk_id_global" in meta.columns and len(meta) else "",
                "sample_pages": meta["pages"].iloc[0] if "pages" in meta.columns and len(meta) else [],
            },
            "artifacts": {
                "faiss_index": str(index_path),
                "embeddings_npy": str(emb_path),
                "chunk_meta_parquet": str(meta_path),
            },
            "preprocess_trace": {
                "schema_version": preprocess_schema,
                "final_text_normalization": preprocess_norm,
                "artifact_counts_sample": artifact_counts,
                "artifact_scan_max_chunks": ARTIFACT_SCAN_MAX_CHUNKS,
                "fail_on_artifacts": FAIL_ON_ARTIFACTS,
            },
        }
    }

    merged = {**preprocess_metrics, **metrics_update}
    write_json(metrics_path, merged)
    print(f"{doc_dir.name}: Updated metrics: {metrics_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    hf_logging.set_verbosity_error()
    args = parse_args()
    global BASE_DATA_DIR, EMBED_MODEL_NAME
    BASE_DATA_DIR = Path(args.data_dir)
    EMBED_MODEL_NAME = args.model
    doc_dirs = iter_document_dirs(BASE_DATA_DIR)
    if not doc_dirs:
        print("No document folders found under:", BASE_DATA_DIR)
        return

    print(f"Loading embedding model once: {EMBED_MODEL_NAME}")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    processed = 0
    skipped = 0
    failed = 0

    for doc_dir in doc_dirs:
        skip, reason = should_skip_doc(doc_dir)
        if skip:
            skipped += 1
            print(f"\n=== SKIP {doc_dir.name} ({reason}) ===")
            continue

        print(f"\n=== BUILD INDEX: {doc_dir.name} ===")
        try:
            build_index_for_doc(doc_dir, model)
            processed += 1
        except Exception as e:
            failed += 1
            print(f"{doc_dir.name}: FAILED: {type(e).__name__}: {e}")

    print("\nDONE")
    print("Base dir:", BASE_DATA_DIR)
    print("Processed:", processed)
    print("Skipped:", skipped)
    print("Failed:", failed)


if __name__ == "__main__":
    main()
