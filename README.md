# RAG_Pipeline_Project

A lightweight, local Retrieval-Augmented Generation (RAG) pipeline for PDF reports. It covers:

- PDF preprocessing (cleaning, header/footer removal, sectioning, chunking)
- Embedding + FAISS index build
- Retrieval evaluation against a labeled question set

The scripts are designed for reproducible, page-accurate retrieval on large reports (e.g., NHS annual reports).

## Project Structure

- `Data/` — raw PDFs
- `data_processed/` — per-document outputs from preprocessing and indexing
- `preprocess_pdf_rag.py` — extract text, clean, detect sections, chunk, and write metrics
- `build_index.py` — embed chunks and build a FAISS index
- `retrieval_eval.py` — evaluate retrieval with an `eval_set.json`
- `faiss_smoke.py`, `st_smoke.py`, `st_smoke_safe.py` — quick dependency checks
- `inspect_section.py`, `make_charts.py`, `make_titles_before_after.py` — analysis/visualization helpers
- `figures/` — charts/figures produced by analysis scripts
- `RAG_NHS.tex`, `RAG_Poster.tex` — thesis/ poster sources and compiled outputs

## Requirements

Python 3.10+ recommended. Core dependencies by script:

- `preprocess_pdf_rag.py`: `pymupdf`, `pandas`, `pyarrow` (optional: `tiktoken`)
- `build_index.py`: `faiss-cpu`, `sentence-transformers`, `pandas`, `pyarrow`, `numpy`
- `retrieval_eval.py`: `faiss-cpu`, `sentence-transformers`, `pandas`, `pyarrow`, `numpy`

Example setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pymupdf pandas pyarrow tiktoken faiss-cpu sentence-transformers numpy
```

## Configuration

Key constants to adjust before running each script:

- `preprocess_pdf_rag.py`
  - Paths: `PDF_PATH`, `DOC_ID`, `OUT_ROOT`
  - Chunking: `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`
  - Header/footer removal: `TOP_STRIP_FRAC`, `BOTTOM_STRIP_FRAC`, `HEADER_FOOTER_REPEAT_FRAC`, `TOP_LINE_K`, `BOT_LINE_K`
  - Heading detection: `HEADING_MAX_CHARS`, `HEADING_MIN_CHARS`, `HEADING_FONT_BOOST_FRAC`
  - Filters: `MIN_CHUNK_WORDS`

- `build_index.py`
  - Paths: `DATA_DIR`, `CHUNKS_PATH`, `METRICS_PATH`
  - Embeddings/index: `EMBED_MODEL_NAME`, `FAISS_INDEX_NAME`, `EMB_NPY_NAME`, `META_PARQUET_NAME`
  - Retrieval sanity check: `TOPK_DEFAULT`

- `retrieval_eval.py`
  - Paths: `DATA_DIR`, `INDEX_PATH`, `META_PATH`, `EVAL_SET_PATH`
  - Embeddings: `EMBED_MODEL_NAME`
  - Metrics/output: `K_LIST`, `RESULTS_JSON`, `METRICS_JSON`, `SUMMARY_CSV`

- `inspect_section.py`
  - Paths: `PROJECT_ROOT`, `DOC_FOLDER`, `sections_path`, `chunks_path`

- `make_charts.py`
  - Paths: `PROJECT_ROOT`, `DOC_ID`, `DATA_DIR`, `CHUNKS_PATH`, `SECTIONS_PATH`, `FIG_DIR`
  - Matplotlib: `MPLBACKEND`

- `make_titles_before_after.py`
  - Paths: `PROJECT_ROOT`, `DOC_ID`, `DATA_DIR`, `SECTIONS_PATH`, `FIG_DIR`, `OUT_BEFORE`, `OUT_AFTER`
  - Title filtering: `TOP_N`, `STOP_TITLES`, `DATE_TITLE_RE`

## Quickstart

1) Point to your PDF

Edit `PDF_PATH` and `OUT_ROOT` in `preprocess_pdf_rag.py`.

2) Preprocess the PDF

```bash
python preprocess_pdf_rag.py
```

Outputs per document (under `data_processed/<DOC_ID>/`):

- `pages.parquet`
- `sections.parquet`
- `chunks.parquet`
- `metrics.json`
- `qa_report.json`
- `sample_chunks.md`

3) Build embeddings + FAISS index

Edit `DATA_DIR` in `build_index.py` to point at the folder containing `chunks.parquet`, then run:

```bash
python build_index.py
```

Outputs:

- `faiss.index`
- `embeddings.npy`
- `chunk_meta.parquet`
- `metrics.json` (updated)

4) Evaluate retrieval

Create an `eval_set.json` in the same `DATA_DIR` used above, then run:

```bash
python retrieval_eval.py
```

Outputs:

- `retrieval_results.json`
- `retrieval_metrics.json`
- `retrieval_summary.csv`

## Notes

- Paths in scripts are currently absolute; update them to match your environment.
- If you see `fitz` import errors, uninstall the `fitz` package and install `pymupdf`.
- The FAISS index uses inner product on L2-normalized vectors to approximate cosine similarity.

## Example eval_set.json

```json
[
  {
    "query_id": "Q001",
    "question": "What is the reporting period end date?",
    "expected_pages": [1],
    "answer_type": "date"
  },
  {
    "query_id": "Q002",
    "question": "What is the total staff costs figure?",
    "expected_pages": [120, 121],
    "answer_type": "number"
  }
]
```

## License

Add a license if you plan to share this project publicly.
