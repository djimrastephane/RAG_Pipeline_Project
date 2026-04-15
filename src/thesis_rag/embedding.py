from __future__ import annotations

import logging
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from .schemas import ChunkRecord, EmbeddingConfig
from .utils import l2_normalize

LOGGER = logging.getLogger(__name__)


def load_embedding_model(config: EmbeddingConfig, device: str, cache_dir: str) -> SentenceTransformer:
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    model = SentenceTransformer(config.model_name, device=device, cache_folder=cache_dir)
    return model


def build_embedding_text(chunk: ChunkRecord) -> str:
    parts: list[str] = []
    section = (chunk.section_title or "").strip()
    subsection = (chunk.subsection_title or "").strip()
    if section and section.lower() != "unknown":
        parts.append(section)
    if subsection and subsection.lower() != "unknown" and not chunk.is_table:
        parts.append(subsection)
    parts.append(chunk.text)
    return "\n".join(part for part in parts if part)


def embed_chunks(
    chunks: list[ChunkRecord],
    config: EmbeddingConfig,
    *,
    device: str,
    cache_dir: str,
) -> np.ndarray:
    model = load_embedding_model(config, device=device, cache_dir=cache_dir)
    texts = [build_embedding_text(chunk) for chunk in chunks]
    vectors = model.encode(
        texts,
        batch_size=config.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    if len(vectors) != len(chunks):
        raise ValueError("Embedding count does not match chunk count")
    if vectors.shape[1] != config.expected_dimension:
        raise ValueError(
            f"Unexpected embedding dimension {vectors.shape[1]} for {config.model_name}; "
            f"expected {config.expected_dimension}"
        )
    if config.normalize_embeddings:
        vectors = l2_normalize(vectors.astype(np.float32))
    LOGGER.info("Embedded %s chunks with dimension %s", len(chunks), vectors.shape[1])
    return vectors.astype(np.float32)


def embed_queries(
    query_texts: list[str],
    config: EmbeddingConfig,
    *,
    device: str,
    cache_dir: str,
) -> np.ndarray:
    model = load_embedding_model(config, device=device, cache_dir=cache_dir)
    vectors = model.encode(
        query_texts,
        batch_size=config.batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    if config.normalize_embeddings:
        vectors = l2_normalize(vectors.astype(np.float32))
    return vectors.astype(np.float32)
