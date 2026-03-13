% Query Flow: chunks.parquet + chunk_meta.parquet
% RAG Pipeline Illustration

# Query -> Answer (Visual)

## Core path

`User Query`

`-> Query Embedding`

`-> FAISS Search`

`-> Top Row IDs`

`-> chunk_meta.parquet (row lookup)`

`-> chunk_id_global + pages`

`-> chunks.parquet (text lookup by ID)`

`-> Top-k Evidence Context`

`-> LLM`

`-> Final Answer + Citations`

---

# Roles At a Glance

## `chunk_meta.parquet`
- Fast retrieval metadata
- Aligned to FAISS row positions
- Provides: `chunk_id_global`, `pages`, section hints

## `chunks.parquet`
- Full chunk records
- Provides: `chunk_text` and rich attributes
- Supplies text that is injected into the LLM prompt

## Connection
- Join key: `chunk_id_global`
- Fallback: `chunk_id`

---

# Mini Flow Diagram

## Mapping logic

| Retrieval stage | Data used | Output |
|---|---|---|
| FAISS result | `faiss.index` + query embedding | row ids: `[12, 44, 3]` |
| Row grounding | `chunk_meta.parquet` | chunk ids + pages |
| Text fetch | `chunks.parquet` | chunk_text snippets |
| Answering | top-k snippets + pages | cited final answer |

---

# Real `chunk_meta.parquet` Snippet

| chunk_id  | chunk_id_global              | page_start | page_end | pages | section_title      |
|-----------|------------------------------|------------|----------|-------|--------------------|
| p0002_000 | Grampian-2022-2023:p0002_000 | 2          | 2        | 2     | Performance Report |
| p0003_000 | Grampian-2022-2023:p0003_000 | 3          | 3        | 3     | PERFORMANCE REPORT |
| p0003_001 | Grampian-2022-2023:p0003_001 | 3          | 3        | 3     | PERFORMANCE REPORT |

(From `data_processed/Grampian-2022-2023`)

---

# Real `chunks.parquet` Snippet

| chunk_id  | chunk_id_global              | subsection_title | chunk_text (truncated) |
|-----------|------------------------------|------------------|-------------------------|
| p0002_000 | Grampian-2022-2023:p0002_000 | Unknown          | Page Performance Report 2 a) Overview... |
| p0003_000 | Grampian-2022-2023:p0003_000 | A) OVERVIEW      | PERFORMANCE REPORT A OVERVIEW 1. Purpose... |
| p0003_001 | Grampian-2022-2023:p0003_001 | A) OVERVIEW      | everything that people in communities... |

(From `data_processed/Grampian-2022-2023`)

---

# 15-Second Talk Track

1. FAISS returns nearest vector row IDs.
2. Those row IDs directly select `chunk_meta` rows.
3. Meta rows give chunk IDs and pages.
4. Chunk IDs fetch full text from `chunks`.
5. LLM answers using those snippets and cites the pages.

---

# One-Line Takeaway

`FAISS finds candidates, chunk_meta grounds them, chunks supplies text, and the LLM returns a cited answer.`
