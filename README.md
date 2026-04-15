# RAG_Pipeline_Project

A lightweight, local Retrieval-Augmented Generation (RAG) pipeline for PDF reports. It covers:

- PDF preprocessing (cleaning, header/footer removal, sectioning, chunking)
- Embedding + FAISS index build
- Retrieval evaluation against a labeled question set

## Thesis Refactor Path

The repository now also includes a thesis-focused refactor under `src/thesis_rag/`. This path is designed for:

- typed pipeline stages and YAML configuration
- deterministic CPU-first retrieval runs
- isolated `runs/<timestamp>_*` outputs with saved config, logs, and metrics
- explicit dense, sparse, fused, and page-level evaluation artifacts
- unit tests around chunking, page mapping, fusion, and metrics

Thin CLI entrypoints:

```bash
python scripts/preprocess.py --config configs/thesis_rag.yaml
python scripts/index.py --config configs/thesis_rag.yaml --chunks-path runs/<preprocess_run>/<DOC_ID>/chunks.parquet
python scripts/retrieve.py --config configs/thesis_rag.yaml --chunk-metadata-path runs/<index_run>/chunk_metadata.parquet --faiss-index-path runs/<index_run>/faiss.index --query-set-path data/eval_set.json
python scripts/evaluate.py --config configs/thesis_rag.yaml --dense-hits-path runs/<retrieve_run>/dense_page_hits.jsonl --sparse-hits-path runs/<retrieve_run>/bm25_page_hits.jsonl --hybrid-hits-path runs/<retrieve_run>/hybrid_page_hits.jsonl --query-set-path data/eval_set.json
```

The scripts are designed for reproducible, page-accurate retrieval on large reports (e.g., NHS annual reports).

Parity status against the legacy pipeline has been checked on a fixed benchmark subset for `Grampian-2022-2023`.

- Preprocessing counts are matched.
- BM25 top-10 parity is matched on the checked subset.
- Hybrid top-10 parity is matched on the checked subset.
- Page-level retrieval metrics are matched on the checked subset.
- One dense-only exact-order exception remains (`Q_2023_FIN_07`), despite identical candidate chunk vectors and matched eval-set provenance.

See [docs/experiments/thesis_refactor_parity_note.md](docs/experiments/thesis_refactor_parity_note.md) and the final parity run [runs/parity_validation/manual_2026-04-15_grampian_subset10_fix10b](runs/parity_validation/manual_2026-04-15_grampian_subset10_fix10b).

Accepted switch criteria for the thesis refactor were: fixed-subset parity checks against the legacy pipeline, exact BM25 and hybrid top-10 agreement on that subset, matched page-level `Hit@1`, `Hit@3`, and `MRR`, a documented forensic review of the one remaining dense-only discrepancy (`Q_2023_FIN_07`), and a clean end-to-end smoke test of `preprocess.py`, `index.py`, `retrieve.py`, and `evaluate.py` in the pinned `rag-pipeline` environment. On that basis, `src/thesis_rag/` is the maintained pipeline for thesis experiments, with the legacy path retained only as an archived reference baseline.

## Project Structure

- `src/` — core library code
- `scripts/` — runnable pipeline, evaluation, and analysis entrypoints
- `docs/` — thesis notes, architecture diagrams, slide sources, and supporting figures
- `configs/` — tracked config files and examples
- `config/` — local runtime config overrides (legacy, typically untracked)
- `Data/` — raw PDFs (ignored by git)
- `data_processed/` — per-document outputs from preprocessing and indexing (ignored by git)
- `data_variants/` — alternate processed corpora for ablations, refreshes, and comparison roots
- `preprocess_hybrid.py` — thin runner for the hybrid preprocessing pipeline
- `qa/` — validation and QA utilities for preprocessing output
- `results/` / `runs/` — experiment outputs and reports
- `results/ablations/` — promoted ablation summaries, final selections, and comparison artifacts
- `figures/` — local scratch/output area for analysis charts (ignored by git)

Documentation layout:

- `docs/architecture/` — pipeline maps, flowcharts, and slide artifacts
- `docs/experiments/` — ablation notes and retrieval tuning references
- `docs/ui/` — UI/API notes
- `docs/figures/` — thesis/supporting figures tracked with the repository

## Requirements

Python 3.10+ recommended. For the final thesis pipeline, use the checked-in Conda environment in [`environment.yml`](environment.yml) rather than mixing ad hoc installs into a global interpreter. Core dependencies by script:

- `scripts/preprocess_pdf_rag.py`: `pymupdf`, `pandas`, `pyarrow` (optional: `tiktoken`)
- `scripts/build_index.py`: `faiss-cpu`, `sentence-transformers`, `pandas`, `pyarrow`, `numpy`
- `scripts/evaluate_pipeline.py`: `faiss-cpu`, `sentence-transformers`, `pandas`, `pyarrow`, `numpy`
- `scripts/retrieval_eval.py`: `faiss-cpu`, `sentence-transformers`, `pandas`, `pyarrow`, `numpy` (dense-first baseline)
- `scripts/benchmark_table_extractors.py`: `pymupdf`, `pdfplumber`, `pandas` (optional: `docling`, benchmark-only)
- OCR fallback: `pytesseract`, `pdf2image`, system `tesseract`, and `poppler` (for `pdftoppm`)

Canonical setup:

```bash
conda env create -f environment.yml
conda activate rag-pipeline
python scripts/check_environment.py --strict
```

Examiner-facing quickstart:

```bash
conda activate rag-pipeline
python scripts/check_examiner_path.py
python scripts/check_environment.py --strict
python scripts/check_pipeline_reproducibility.py --runs 2 --out-json results/reproducibility/examiner_repro_check.json
```

See `docs/EXAMINER_QUICKSTART.md` for the short verification path and the optional demo path.

Fallback setup if you are not using Conda:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/check_environment.py --strict
```

Environment policy:

- Do not rely on the global Anaconda base environment for final preprocessing or evaluation runs.
- Run `python scripts/check_environment.py --strict` before rebuilding canonical `data_processed` roots.
- `metrics.json` and retrieval output `run_info` now record runtime provenance and dependency status so runs can be audited later.
- Repo-launched Python processes set `MPLCONFIGDIR` to a writable project-local cache under `.cache/matplotlib` to avoid Matplotlib cache permission issues.

Known warnings on macOS:

- `RequestsDependencyWarning` from `requests` may appear in this environment; it has been non-blocking in the validated smoke runs.
- `CryptographyDeprecationWarning` from `pypdf` may appear during PDF-related stages; it has also been non-blocking in the validated smoke runs.
- On macOS, the thesis CLI sets `KMP_DUPLICATE_LIB_OK=TRUE` for the FAISS/Torch stages to avoid duplicate OpenMP runtime aborts in this environment.

Windows setup notes:

- The refactored pipeline is designed to be cross-platform, but Windows should be treated as supported only after a local smoke test with `preprocess.py`, `index.py`, `retrieve.py`, and `evaluate.py`.
- `faiss-cpu` publishes Windows wheels, so dense retrieval is feasible on Windows with the pinned Python 3.11 environment.
- OCR and table extraction depend on external binaries. On Windows, make sure these executables are installed and either available on `PATH` or referenced explicitly in your environment:
  - Tesseract OCR: commonly `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - Ghostscript console binary: commonly `C:\Program Files\gs\gs<version>\bin\gswin64c.exe`
  - Poppler tools for `pdf2image`: commonly `C:\poppler\Library\bin\pdftoppm.exe` or `C:\poppler\bin\pdftoppm.exe`
- If `pytesseract` cannot find Tesseract, set `pytesseract.pytesseract.tesseract_cmd` to the full `tesseract.exe` path. The `pytesseract` project documents this usage pattern directly.
- If `pdf2image` cannot find Poppler, add the Poppler `bin` directory to `PATH` or pass `poppler_path` explicitly. The `pdf2image` documentation notes this requirement for Windows.
- For Ghostscript, the command-line executable to validate is `gswin64c.exe`; Ghostscript documents this as the normal Windows command prompt binary.
- After installing those tools, run `python scripts/check_environment.py --strict` before attempting a full pipeline run on Windows.

Core runtime policy:

- The active pipeline does not require Docling.
- Docling is used only for optional A/B benchmarking in `scripts/benchmark_table_extractors.py`.
- `umap-learn` is not part of the canonical thesis runtime; install it separately only if you need `scripts/export_wizmap_umap.py` or other embedding-visualization analysis helpers.

## Configuration

Key constants to adjust before running each script:

- `scripts/preprocess_pdf_rag.py`
  - Paths: `PDF_PATH`, `DOC_ID`, `OUT_ROOT`
  - Chunking: `CHUNK_SIZE_TOKENS`, `CHUNK_OVERLAP_TOKENS`
  - Header/footer removal: `TOP_STRIP_FRAC`, `BOTTOM_STRIP_FRAC`, `HEADER_FOOTER_REPEAT_FRAC`, `TOP_LINE_K`, `BOT_LINE_K`
  - Heading detection: `HEADING_MAX_CHARS`, `HEADING_MIN_CHARS`, `HEADING_FONT_BOOST_FRAC`
  - Filters: `MIN_CHUNK_WORDS`

- `scripts/build_index.py`
  - Paths: `DATA_DIR`, `CHUNKS_PATH`, `METRICS_PATH`
  - Embeddings/index: `EMBED_MODEL_NAME`, `FAISS_INDEX_NAME`, `EMB_NPY_NAME`, `META_PARQUET_NAME`
  - Retrieval sanity check: `TOPK_DEFAULT`

- `scripts/evaluate_pipeline.py`
  - Canonical evaluation entrypoint for the production pipeline
  - Runs hybrid Dense+BM25 RRF evaluation and then builds reports
- `scripts/retrieval_eval.py`
  - Dense-first baseline evaluator kept for ablations/comparisons
- `scripts/build_global_indexes.py`
  - Multi-document artifacts: global dense index + lexical scope manifest
  - Paths: `--data-root`, `--out-dir`

- `scripts/preprocess_hybrid.py`
  - OCR thresholds: `OCR_MIN_ALPHA_RATIO`, `OCR_MIN_DIGIT_RATIO`
- `scripts/report_retrieval_metrics.py`
  - Outputs: `retrieval_report.csv`, `retrieval_queries_report.csv`, `retrieval_failure_summary.csv`
- `scripts/run_full_pipeline.py`
  - Full pipeline: preprocess -> build index -> retrieval eval -> reports

## Quickstart

1) Validate the environment

```bash
python scripts/check_environment.py --strict
```

2) Point to your PDF

Edit `PDF_PATH` and `OUT_ROOT` in `scripts/preprocess_pdf_rag.py`.

3) Preprocess the PDF

```bash
python preprocess_hybrid.py
```

Outputs per document (under `data_processed/<DOC_ID>/`):

- `pages.parquet`
- `sections.parquet`
- `chunks.parquet`
- `metrics.json`
- `qa_report.json`
- `sample_chunks.md`
- `ocr_pages.csv` (pages processed with OCR, if enabled)
- Table chunks include a Markdown rendering of detected tables to preserve structure for retrieval.
- `table_facts.parquet` (canonical row/column/value facts derived from extracted tables)

4) Build embeddings + FAISS index

Edit `DATA_DIR` in `scripts/build_index.py` to point at the folder containing `chunks.parquet`, then run:

```bash
python scripts/build_index.py
```

Outputs:

- `faiss.index`
- `embeddings.npy`
- `chunk_meta.parquet`
- `metrics.json` (updated)

Optional: build global (multi-document) retrieval artifacts

```bash
python scripts/build_global_indexes.py --data-root data_processed --out-dir data_processed/_global
```

Outputs:
- `data_processed/_global/global_dense.faiss`
- `data_processed/_global/global_meta.parquet`
- `data_processed/_global/lexical_manifest.json`
- `data_processed/_global/lexical_corpus.parquet`

5) Evaluate retrieval

Create an `eval_set.json` in the same `DATA_DIR` used above, then run:

```bash
python scripts/evaluate_pipeline.py --data-dir data_processed/<DOC_ID>
```

Outputs:

- `retrieval_results_hybrid.json`
- `retrieval_metrics_hybrid.json`
- `retrieval_summary_hybrid.csv`
- `retrieval_report.csv`
- `retrieval_queries_report.csv`
- `retrieval_failure_summary.csv`

When `eval_set.json` includes `expected_answer`, `retrieval_results_hybrid.json` now also includes:

- `answer_correct` (true/false/null when not scored)
- `answer_status` (`correct`, `partial`, `incorrect`, `not_scored`)

And `retrieval_metrics_hybrid.json` includes an `answer_scoring` block with aggregate answer accuracy.

If you explicitly want the older dense-only baseline for comparison, run:

```bash
python scripts/retrieval_eval.py
```

6) Build reports (including failure summary)

```bash
python scripts/report_retrieval_metrics.py
```

Outputs:

- `retrieval_report.csv` (metrics by doc/k)
- `retrieval_queries_report.csv` (per-query details)
- `retrieval_failure_summary.csv` (one-row failure counts)

## Demo UI

Start the API:

```bash
bash scripts/run_api_demo.sh
```

Start the current Streamlit UI:

```bash
bash scripts/run_streamlit_demo.sh current
```

Start the preserved legacy Streamlit UI:

```bash
bash scripts/run_streamlit_demo.sh legacy
```

## Failure Taxonomy (Evaluation)

The pipeline tags each query with a single failure type at k=1 to separate retrieval vs generation errors:

Retrieval-stage failures (FP1–FP3):
- `FP1_MISSING_CONTENT` — expected pages are not present in the index.
- `FP2_MISSED_TOP_RANK` — expected pages exist but are not retrieved at k=1.
- `FP3_NOT_IN_CONTEXT` — expected pages are retrieved, but the expected answer is not found in the retrieved context.

Generation-stage failures (FP4–FP7):
- `FP4_NOT_EXTRACTED` — answer appears in context but extraction returns nothing useful.
- `FP5_WRONG_FORMAT` — extracted answer does not match the required type (number/date/list).
- `FP6_INCORRECT_SPECIFICITY` — extracted answer is the wrong value despite the right context.
- `FP7_INCOMPLETE` — extracted answer partially matches the expected answer.

Success cases are labeled `HIT`. The per-query report includes both `failure_type` and `failure_stage`.

## Full Pipeline Runner

Runs preprocess -> build index -> retrieval eval -> reports in one command.

```bash
python scripts/run_full_pipeline.py \
  --pdf-dir "/path/to/pdfs" \
  --out-root data_processed \
  --model models/all-MiniLM-L6-v2
```

## Notes

- Paths in scripts are currently absolute; update them to match your environment.
- If `python scripts/check_environment.py --strict` fails on `camelot` or `gs`, do not trust table-aware preprocessing outputs until that is fixed.
- If you see `fitz` import errors, uninstall the `fitz` package and install `pymupdf`.
- The FAISS index uses inner product on L2-normalized vectors to approximate cosine similarity.
- Search API supports metadata filtering and scope control:
  - `retrieval_scope`: `doc | trust | global`
  - `lexical_scope`: `doc | trust | global`
  - filters: `doc_id`, `trust_id`, `year`, `is_table`, `section_contains`, `subsection_contains`

## Canonical Trust IDs

Trust-scoped retrieval uses canonical NHS board names derived from `doc_id`.
Current canonical list:

- NHS Ayrshire & Arran
- NHS Borders
- NHS Dumfries & Galloway
- NHS Fife
- NHS Forth Valley
- NHS Grampian
- NHS Greater Glasgow & Clyde
- NHS Highland
- NHS Lanarkshire
- NHS Lothian
- NHS Orkney
- NHS Shetland
- NHS Tayside
- NHS Western Isles
- If `scripts/build_index.py` or `scripts/retrieval_eval.py` crashes with a SIGSEGV, run with:
  `OMP_NUM_THREADS=1 FAISS_NO_AVX2=1`.
- Git ignores `Data/`, `data_processed/`, `figures/`, `tmp/`, and all `*.pdf` outputs by default.
- OCR requires `tesseract` on PATH; for Homebrew installs this is typically `/opt/homebrew/bin/tesseract`.
- `pdf2image` requires Poppler (`pdftoppm`) on PATH; for Homebrew installs this is typically `/opt/homebrew/bin/pdftoppm`.
- When loading `sentence-transformers/all-MiniLM-L6-v2`, you may see an `UNEXPECTED embeddings.position_ids` warning; it is harmless and can be ignored.

## Example eval_set.json

```json
{
  "_meta": {
    "doc_id": "Grampian-2023-2024"
  },
  "queries": [
    {
      "query_id": "Q_2024_FIN_01",
      "question": "What was the underlying deficit against the Core Revenue Resource Limit for 2023/24?",
      "expected_pages": [27],
      "expected_answer": "-24,703 (£000)",
      "answer_type": "number",
      "doc_id": "Grampian-2023-2024",
      "year": 2024
    }
  ]
}
```

## OCR Setup (Optional)

If you want OCR fallback for image-based pages, install the Python deps and system binaries:

```bash
pip install pytesseract pdf2image
brew install tesseract poppler
```

`ocr_pages.csv` columns:

- `page` — page number (1-based)
- `extractor_notes` — OCR usage reason
- `ocr_text_len` — raw OCR text length
- `clean_text_len` — final normalized text length

OCR behavior:

- Trigger: `clean_text` length < 50 characters.
- Accept: OCR result is used if normalized OCR text length >= 50.
- Tracking: pages using OCR are tagged with `extractor=ocr`.
- Debug: set `OCR_DEBUG=1` to print OCR errors during processing.

OCR metrics (metrics.json):

- `counts.ocr_raw_pages_detected` — OCR attempted in raw extraction stage
- `counts.ocr_raw_pages_accepted` — raw OCR accepted and used in extraction
- `counts.ocr_short_pages_triggered` — clean_text < 50 triggered OCR
- `counts.ocr_short_pages_accepted` — clean_text OCR accepted
- `derived.ocr_raw_acceptance_rate` — accepted / detected (raw OCR)
- `derived.ocr_short_acceptance_rate` — accepted / triggered (short-page OCR)
- `counts.sections_detected` — total sections inferred

Provenance fields (metrics.json):

- `run_utc` — ISO-8601 UTC timestamp for the run
- `git_commit_short` — short git commit hash, if available
- `embedding_model` — model name/path if provided in environment

Derived fields (metrics.json):

- `derived.chunks_per_page`
- `derived.tables_per_100_pages`

CLI + env options:

- `scripts/preprocess_hybrid.py`
  - CLI: `--pdf-path`, `--out-root`
  - Env: `PDF_PATH`, `OUT_ROOT`
- `scripts/build_index.py`
  - CLI: `--data-dir`, `--model`
  - Env: `DATA_DIR`, `EMBED_MODEL_NAME`
- `scripts/retrieval_eval.py`
  - CLI: `--data-dir`, `--model`, `--k-list`
  - Env: `DATA_DIR`, `EMBED_MODEL_NAME`, `K_LIST`

## Table Extractor A/B Benchmark

Benchmark the current extractor (Camelot/pdfplumber path) against Docling on detected table-like pages.

```bash
python scripts/benchmark_table_extractors.py \
  --pdf-path "Data/Annual Accounts NHS Grampian/Preliminary_Test/Grampian-2022-2023.pdf" \
  --max-table-pages 12
```

Outputs are written to `data_processed/benchmarks/`:

- `table_extract_benchmark_per_page_<timestamp>.csv`
- `table_extract_benchmark_summary_<timestamp>.csv`
- `table_extract_benchmark_run_<timestamp>.json`

Notes:

- Docling is benchmark-only and not part of the core pipeline/runtime requirements.
- If Docling is not installed, the script still runs and records `docling_not_available`.
- To enable Docling comparison only for this benchmark, install it in your environment before running:

```bash
pip install docling
```

## Table Facts Backfill

For existing processed folders that already have `tables_structured.parquet`, you can generate canonical facts without rerunning full preprocessing:

```bash
python scripts/backfill_table_facts.py --data-dir data_processed/Grampian-2024-2025
```

## Question Router (Modular QA)

`scripts/retrieval_eval.py` now uses an intent router to dispatch extraction:

- Router module: `src/rag_pdf/question_router.py`
- Current routed families:
  - `table_metric_*` (uses `table_facts.parquet` first)
    - includes milestone metrics, `staff_costs`, and `emissions` intents
  - governance intents (legacy regex path)
  - `unknown` fallback

To add new question classes later:

1. Add a new intent in `route_question(...)` in `src/rag_pdf/question_router.py`.
2. Add/extend extraction logic in `scripts/retrieval_eval.py` for that intent.
3. Re-run eval and inspect `route_intent` / `route_confidence` in `retrieval_results.json`.

## Retrieval Tuning / A-B Ablation

Use the ablation runner to compare:

- top-k settings
- chunking settings (optional rebuild mode)
- lexical/table rerank weights
- deterministic query rewrites
- sparse retrievers (`hybrid` dense+BM25 and `splade_hybrid` dense+SPLADE)

Default hybrid fusion parameters are documented in:
- `docs/experiments/HYBRID_RRF_DEFAULTS.md`

Optional local built-in cross-encoder reranker:
- `docs/experiments/LOCAL_CROSS_ENCODER_RERANKER.md`

Config file:

- `configs/retrieval_tuning.yaml`
- `configs/retrieval_tuning_splade_template.yaml` (SPLADE-ready template)

Run all configured experiments:

```bash
python scripts/run_retrieval_ablation.py --config configs/retrieval_tuning.yaml
```

Run SPLADE ablation template:

```bash
python scripts/run_retrieval_ablation.py \
  --config configs/retrieval_tuning_splade_template.yaml
```

Run only selected experiments:

```bash
python scripts/run_retrieval_ablation.py \
  --config configs/retrieval_tuning.yaml \
  --only baseline_current,baseline_rerank,rewrite_rerank
```

Outputs:

- `results/ablations/ablation/retrieval_ablation_summary.csv`
- `results/ablations/ablation/retrieval_ablation_best_by_k.csv`
- `results/ablations/ablation/retrieval_ablation_summary.json`

Optional markdown report:

```bash
python scripts/report_retrieval_ablation.py \
  --summary-csv results/ablations/ablation/retrieval_ablation_summary.csv
```

## Batch Processing

Use the batch runner with a JSON config to process a folder of PDFs.

Example config: `configs/examples/batch.example.json`

```json
{
  "pdf_dir": "/path/to/pdfs",
  "pdf_glob": "*.pdf",
  "out_root": "/path/to/output_root",
  "embed_model_name": "/path/to/models/all-MiniLM-L6-v2"
}
```

Run:

```bash
python scripts/run_batch.py --config configs/batch.json
```

Batch runner flags + outputs:

- `--force` to reprocess even if outputs already exist
- Logs per PDF: `<out_root>/<DOC_ID>/preprocess.log`
- Summary CSV: `<out_root>/batch_summary.csv`
- If `configs/batch.json` does not exist, `scripts/run_batch.py` falls back to the legacy local path `config/batch.json`

## Consolidated Exports (Multi-Doc)

Generate canonical cross-document consolidated outputs from per-document `retrieval_results.json` files:

```bash
python scripts/export_consolidated_answers.py \
  --data-root data_processed \
  --out-dir data_processed/consolidated \
  --top-k 1
```

Optional generation enrichment from API search logs (JSONL):

```bash
python scripts/export_consolidated_answers.py \
  --data-root data_processed \
  --out-dir data_processed/consolidated \
  --top-k 1 \
  --search-log-jsonl results/search_api.jsonl
```

`--search-log-jsonl` enriches/overrides generation-related fields in consolidated answers when keyed by
`(doc_id, query_id, question)`:
- `generation_status`
- `generation_confidence`
- `low_retrieval_margin`
- `retrieval_margin`
- `answer_mode`
- `final_answer`

Outputs:
- `data_processed/consolidated/consolidated_answers.csv`
- `data_processed/consolidated/consolidated_answers.jsonl`
- `data_processed/consolidated/consolidated_table_facts.csv` (unless `--no-table-facts`)

## Minimal Retrieval UI

A minimal upload/search interface is available in:

- API: `app/api/main.py`
- Streamlit UI: `app/ui/streamlit_app.py`
- Setup/run guide: `docs/ui/README_UI.md`

With `rag-pipeline` activated, the demo launch commands are:

```bash
./scripts/run_api_demo.sh
./scripts/run_streamlit_demo.sh
```

## License

MIT License (see `LICENSE`).
