# How Query-Time Retrieval Uses `chunks.parquet` and `chunk_meta.parquet`

## Purpose
Explain how a user query is transformed into a cited answer using retrieval artifacts.

---

# Artifact Roles

## `chunks.parquet`
- Full chunk content (`chunk_text`) + rich metadata
- Source of text sent to the LLM context

## `chunk_meta.parquet`
- Lightweight retrieval metadata aligned to vector index rows
- Used to map FAISS hit rows to chunk IDs and page citations

## Link Key
- Primary key: `chunk_id_global`
- Fallback key: `chunk_id`

---

# Query-Time Flow

1. User enters a query.
2. Query is embedded by the embedding model.
3. FAISS searches nearest vectors in `faiss.index`.
4. Returned FAISS row IDs select rows in `chunk_meta.parquet`.
5. From each meta row, read `chunk_id_global` (or `chunk_id`) and `pages`.
6. Use that ID to fetch `chunk_text` from `chunks.parquet`.
7. Build top-k evidence context (text + pages + chunk IDs).
8. LLM generates the final answer with citations.

## Short formula
`FAISS row -> chunk_meta row -> chunk_id -> chunks chunk_text -> LLM answer`

---

# Real Snippet: `chunk_meta.parquet`

(From `data_processed/Grampian-2022-2023`)

| chunk_id  | chunk_id_global              | page_start | page_end | pages | section_title      | is_table |
|-----------|------------------------------|------------|----------|-------|--------------------|----------|
| p0002_000 | Grampian-2022-2023:p0002_000 | 2          | 2        | 2     | Performance Report | False    |
| p0003_000 | Grampian-2022-2023:p0003_000 | 3          | 3        | 3     | PERFORMANCE REPORT | False    |
| p0003_001 | Grampian-2022-2023:p0003_001 | 3          | 3        | 3     | PERFORMANCE REPORT | False    |

---

# Real Snippet: `chunks.parquet`

(From `data_processed/Grampian-2022-2023`)

| chunk_id  | chunk_id_global              | page_start | page_end | section_title      | subsection_title | is_table | chunk_text (truncated)                      |
|-----------|------------------------------|------------|----------|--------------------|------------------|----------|---------------------------------------------|
| p0002_000 | Grampian-2022-2023:p0002_000 | 2          | 2        | PERFORMANCE REPORT | Unknown          | False    | Page Performance Report 2 a) Overview...    |
| p0003_000 | Grampian-2022-2023:p0003_000 | 3          | 3        | PERFORMANCE REPORT | A) OVERVIEW      | False    | PERFORMANCE REPORT A OVERVIEW 1. Purpose... |
| p0003_001 | Grampian-2022-2023:p0003_001 | 3          | 3        | PERFORMANCE REPORT | A) OVERVIEW      | False    | everything that people in communities...    |

---

# Example Walkthrough

## Suppose FAISS returns row `44`
- `chunk_meta.parquet` row `44` contains:
  - `chunk_id_global = Grampian-2022-2023:p0102_001`
  - `pages = [102]`
- Retrieval uses `chunk_id_global` to find matching row in `chunks.parquet`.
- The matched `chunk_text` is inserted into LLM context.
- Final response cites page `102` and the source chunk ID.

---

# Takeaway

`FAISS finds rows, chunk_meta grounds them, chunks provides text, and the LLM produces the answer with citations.`
