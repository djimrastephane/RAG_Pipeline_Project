# How `chunk_id_global` Is Kept Consistent Across `chunks`, `chunk_meta`, and FAISS

## 1) Artifact creation order (pipeline timeline)

1. `preprocess_hybrid.py` writes `data_processed/<doc_id>/chunks.parquet`
2. `scripts/build_index.py` loads `chunks.parquet` in its current row order
3. Embedding text is built row-by-row from that same dataframe
4. `embeddings.npy` is written in that same row order
5. `faiss.index` is built with `index.add(embeddings)` in that same row order
6. `chunk_meta.parquet` is built from the same dataframe (`build_meta_table(chunks)`)
7. `metrics.json` is updated (contains index/build metadata)

## Why this matters
- FAISS id `i` == embedding row `i` == `chunk_meta` row `i` == source row `i` from `chunks.parquet`

---

# Alignment contract (exactly how IDs are preserved)

## `chunk_id_global` consistency rule
- `build_meta_table(chunks)` copies `chunk_id_global` directly from the input `chunks` dataframe.
- No join/merge/shuffle is used between embedding generation and meta writing.
- Therefore, row alignment is positional, not probabilistic.

## Query-time mapping
1. FAISS returns hit row id(s): `idx`
2. System reads `chunk_meta.iloc[idx]`
3. Gets `chunk_id_global` + `pages`
4. Uses that ID to fetch text from chunk-text map built from `chunks.parquet`
5. LLM receives context with `[chunk_id=... pages=...]`
6. Citation validator only accepts citations matching retrieved `chunk_id/page`

## Practical guarantee
- If any artifact is regenerated, regenerate all retrieval-linked artifacts together:
  `embeddings.npy`, `faiss.index`, `chunk_meta.parquet`
- Never reorder `chunk_meta.parquet` manually without rebuilding FAISS.

---

# One-line summary for viva

`chunks.parquet` defines canonical row order -> embeddings and FAISS are built in that order -> `chunk_meta.parquet` is copied in that order -> FAISS row IDs deterministically map back to the same global chunk ID and pages.

---

# Hybrid retrieval (Dense + BM25) before ID resolution

## What happens before `chunk_id` resolution
1. Dense branch: query embedding -> FAISS search -> dense ranked list
2. Lexical branch: BM25 scoring over candidate chunk texts -> BM25 ranked list
3. Fusion: Reciprocal Rank Fusion (RRF) combines both ranked lists
4. Final top-k fused rows are selected
5. Then: `FAISS/BM25 fused row id -> chunk_meta row -> chunk_id_global -> chunks text`

## RRF intuition
- A chunk ranked highly by either branch gets boosted.
- A chunk ranked highly by both branches gets boosted more.
- This improves robustness for finance/table questions where exact tokens matter (BM25) and semantics matter (dense).

## Key point for citations
- Citation grounding still uses the same deterministic mapping:
  fused hit index -> `chunk_meta` -> `chunk_id_global/pages` -> validated citation.
