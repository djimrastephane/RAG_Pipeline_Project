# Retrieval Debug UI (MVP)

Minimal interface for:
- Upload PDF
- Show page/chunk counts
- Ask question
- Display top-k chunks, pages, similarity scores
- Highlight expected page(s) when `query_id` is supplied from `eval_set.json`

## 1) Install UI dependencies

```bash
.venv/bin/pip install -r requirements-ui.txt
```

## 2) Run API

```bash
.venv/bin/uvicorn app.api.main:app --reload --port 8000
```

## 3) Run Streamlit UI

```bash
.venv/bin/streamlit run app/ui/streamlit_app.py
```

## 4) Demo Mode (read-only)

Demo mode disables upload/processing and uses only existing processed documents.

API:

```bash
DEMO_MODE=1 .venv/bin/uvicorn app.api.main:app --reload --port 8000
```

UI:

```bash
DEMO_MODE=1 .venv/bin/streamlit run app/ui/streamlit_app.py
```

## API Endpoints

- `GET /api/v1/health`
- `GET /api/v1/metrics`
- `GET /api/v1/docs`
- `POST /api/v1/docs/upload` (JSON/base64: `pdf_filename`, `pdf_base64`, optional `eval_filename`, `eval_base64`)
- `GET /api/v1/docs/{doc_id}/stats`
- `GET /api/v1/docs/{doc_id}/eval-items`
- `GET /api/v1/docs/{doc_id}/tables?limit=200&page=<int>`
- `GET /api/v1/docs/{doc_id}/logs?last_n=200`
- `POST /api/v1/docs/{doc_id}/search`

## Search API Contract

Request body (`POST /api/v1/docs/{doc_id}/search`):

```json
{
  "question": "What was the underlying deficit?",
  "k": 5,
  "query_id": "Q_2024_FIN_01",
  "include_generated_answer": false,
  "gen_max_context_chunks": 5,
  "gen_max_context_chars": 9000,
  "gen_max_chunk_chars": 2200,
  "gen_timeout_seconds": 20
}
```

Key behavior:
- `include_generated_answer=false` (default): generation is skipped.
- `include_generated_answer=true`: backend may generate an answer, then applies grounding checks.
- `gen_max_context_chunks`, `gen_max_context_chars`, `gen_max_chunk_chars`, `gen_timeout_seconds`:
  optional per-request generation overrides for live troubleshooting in UI.
- Generated answers are gated if grounded citations are missing/invalid.

Relevant response fields:
- `predicted_answer`: retrieval-only fallback answer extracted from top-ranked evidence
- `generated_answer`: string or `null`
- `generated_citations`: list of `{ "chunk_id": "...", "page": 12 }`
- `generation_status`: `ok | skipped | insufficient_evidence | error`
- `generation_confidence`: float or `null`
- `generation_debug`: provider/status/error + citation parse diagnostics + prompt/context/latency stats

## Security/Runtime Env Vars

- `API_KEY`: optional; when set, protected endpoints require `X-API-Key`.
- `UI_ALLOWED_ORIGINS`: comma-separated CORS allowlist.
- `MAX_UPLOAD_MB`: upload payload size cap.
- `MAX_LOG_TAIL_LINES`: cap for `/logs?last_n=`.
- `MAX_TABLE_LIMIT`: cap for `/tables?limit=`.
- `RATE_LIMIT_REQ_PER_MIN`: global fallback limit.
- `RATE_LIMIT_UPLOAD_PER_MIN`: upload endpoint limit.
- `RATE_LIMIT_SEARCH_PER_MIN`: search endpoint limit.
- `RATE_LIMIT_RANK_PER_MIN`: rank endpoint limit.
- `RATE_LIMIT_READ_PER_MIN`: read endpoints limit.
- `GEN_MAX_CONTEXT_CHUNKS`: max retrieved chunks injected into LLM prompt.
- `GEN_MAX_CONTEXT_CHARS`: max total context chars injected into LLM prompt.
- `GEN_MAX_CHUNK_CHARS`: max chars per retrieved chunk before truncation in prompt.
- `GEN_TIMEOUT_SECONDS`: generation timeout (seconds) for `/search` generation path.
- `RETRIEVAL_MARGIN_LOW_THRESHOLD`: threshold for low top1-top2 fused score margin flag.

Metrics endpoint notes:
- `GET /api/v1/metrics` returns in-memory counters for:
  - generation status distribution
  - citation parsed/valid/rejected totals
  - derived citation rates and average generation latency
- Endpoint is protected by the same API key + read rate limit policy.

## Notes

- Uploaded PDFs are processed into `data_processed_ui/<DOC_ID>/`.
- Indexing uses your local model path: `models/all-MiniLM-L6-v2`.
- Per-document pipeline logs are written to `data_processed_ui/<DOC_ID>/ui_pipeline.log`.
- Processing calls existing scripts:
  - `preprocess_hybrid.py`
  - `scripts/build_index.py`
