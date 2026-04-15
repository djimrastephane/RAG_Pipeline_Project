from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from .schemas import ChunkRecord, FaissConfig

LOGGER = logging.getLogger(__name__)


def build_faiss_index(vectors: np.ndarray, config: FaissConfig) -> faiss.Index:
    if config.index_type != "IndexFlatIP":
        raise ValueError("Only exact IndexFlatIP is supported for deterministic thesis runs.")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    if index.ntotal != len(vectors):
        raise ValueError(f"FAISS index size {index.ntotal} does not match vector count {len(vectors)}")
    return index


def save_faiss_index(index: faiss.Index, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_path))


def save_embeddings(vectors: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, vectors)


def save_chunk_metadata(chunks: list[ChunkRecord], out_path: Path) -> None:
    frame = pd.DataFrame([chunk.to_dict() for chunk in chunks])
    frame.to_parquet(out_path, index=False)
    if frame["chunk_id"].duplicated().any():
        duplicates = frame.loc[frame["chunk_id"].duplicated(), "chunk_id"].tolist()
        raise ValueError(f"Duplicate chunk ids in metadata: {duplicates[:5]}")
    LOGGER.info("Saved chunk metadata for %s chunks", len(frame))
