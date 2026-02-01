"""
build_index.py

Build embeddings + FAISS index from chunks.parquet produced by preprocess_pdf_rag.py.

Batch mode (Option 2):
- Automatically scans BASE_DATA_DIR for document folders.
- For each folder containing chunks.parquet, builds:
  - faiss.index
  - embeddings.npy
  - chunk_meta.parquet
  - updates metrics.json with embedding/index details

How this is used later
- Retrieval: embed a query, search the FAISS index, map returned row ids to chunk_id + page_list
- Evaluation: compute Recall@k and citation accuracy using expected_pages vs retrieved page_list
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

import faiss
from sentence_transformers import SentenceTransformer


# =============================================================================
# CONFIG
# =============================================================================
BASE_DATA_DIR = Path(
    "/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed"
)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

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


def build_meta_table(chunks: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a lightweight metadata table aligned to the FAISS index rows.

    Critical for:
    - mapping FAISS row ids -> chunk_id
    - returning page provenance for citations (page_list, page_start)
    """
    preferred_cols = [
        "chunk_id",
        "doc_id",
        "report_year",
        "report_year_source",
        "period_end_date",
        "part",
        "section_title",
        "page_start",
        "page_end",
        "page_list",
    ]

    cols = [c for c in preferred_cols if c in chunks.columns]
    if "chunk_id" not in cols:
        raise ValueError("chunks.parquet must include 'chunk_id'.")

    meta = chunks[cols].copy()

    def _to_page_list(v) -> list[int]:
        """
        Normalises page_list into a python list[int].
        Handles list/tuple, numpy arrays, scalars, and stringified lists.
        """
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return []

        if isinstance(v, (list, tuple)):
            out = []
            for x in v:
                if x is None or (isinstance(x, float) and pd.isna(x)):
                    continue
                out.append(int(x))
            return out

        if hasattr(v, "tolist"):
            try:
                vv = v.tolist()
                if isinstance(vv, list):
                    return [int(x) for x in vv if x is not None]
                return [int(vv)]
            except Exception:
                pass

        if isinstance(v, str):
            s = v.strip()
            if s.startswith("[") and s.endswith("]"):
                nums = re.findall(r"\d+", s)
                return [int(n) for n in nums]
            nums = re.findall(r"\d+", s)
            if len(nums) == 1:
                return [int(nums[0])]
            return []

        try:
            return [int(v)]
        except Exception:
            return []

    if "page_list" in meta.columns:
        meta["page_list"] = meta["page_list"].apply(_to_page_list)

    return meta


def should_skip_doc(doc_dir: Path) -> tuple[bool, str]:
    """
    Determines whether a document folder should be skipped.

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
    Returns all immediate subdirectories of base_dir.
    """
    if not base_dir.exists():
        raise FileNotFoundError(f"BASE_DATA_DIR not found: {base_dir}")
    return sorted([d for d in base_dir.iterdir() if d.is_dir()], key=lambda p: p.name.lower())


# =============================================================================
# PER-DOCUMENT BUILD
# =============================================================================
def build_index_for_doc(doc_dir: Path, model: SentenceTransformer) -> None:
    """
    Build embeddings + FAISS index for one processed document folder.
    """
    chunks_path = doc_dir / CHUNKS_FILENAME
    metrics_path = doc_dir / METRICS_FILENAME

    chunks = pd.read_parquet(chunks_path)

    required_cols = {"chunk_id", "chunk_text"}
    missing = required_cols - set(chunks.columns)
    if missing:
        raise ValueError(f"{doc_dir.name}: chunks.parquet missing columns: {missing}")

    texts = chunks["chunk_text"].fillna("").astype(str).tolist()
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

    # Load preprocess metrics (traceability)
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

    # Build FAISS index (exact baseline)
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