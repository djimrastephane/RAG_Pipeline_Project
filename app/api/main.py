from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional
import sys
import base64
import os
import re
import hmac
import time
import threading
from collections import defaultdict, deque

from fastapi import FastAPI, HTTPException, Depends, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    DocRankRequest,
    MetricsResponse,
    SearchRequest,
    SearchResponse,
    UploadRequest,
    UploadResponse,
)
REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPO_ROOT / "models" / "all-MiniLM-L6-v2"
SRC_PATH = REPO_ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from rag_pdf.services.process_service import ProcessService
from rag_pdf.services.search_service import SearchService
from rag_pdf.services.storage_service import StorageService

storage = StorageService(repo_root=REPO_ROOT)
process_service = ProcessService(repo_root=REPO_ROOT, data_root=storage.data_root, model_path=MODEL_PATH)
search_service = SearchService(repo_root=REPO_ROOT, model_path=MODEL_PATH)
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"
API_KEY = str(os.getenv("API_KEY", "")).strip()
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "20"))
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)
MAX_LOG_TAIL_LINES = int(os.getenv("MAX_LOG_TAIL_LINES", "1000"))
MAX_TABLE_LIMIT = int(os.getenv("MAX_TABLE_LIMIT", "500"))
RATE_LIMIT_REQ_PER_MIN = int(os.getenv("RATE_LIMIT_REQ_PER_MIN", "60"))
RATE_LIMIT_UPLOAD_PER_MIN = int(os.getenv("RATE_LIMIT_UPLOAD_PER_MIN", "8"))
RATE_LIMIT_SEARCH_PER_MIN = int(os.getenv("RATE_LIMIT_SEARCH_PER_MIN", "60"))
RATE_LIMIT_RANK_PER_MIN = int(os.getenv("RATE_LIMIT_RANK_PER_MIN", "20"))
RATE_LIMIT_READ_PER_MIN = int(os.getenv("RATE_LIMIT_READ_PER_MIN", "120"))
SAFE_UPLOAD_FILENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,180}$")
ALLOWED_ORIGINS = [
    x.strip()
    for x in os.getenv(
        "UI_ALLOWED_ORIGINS",
        "http://localhost:8501,http://127.0.0.1:8501",
    ).split(",")
    if x.strip()
]
ALLOW_CREDENTIALS = not (len(ALLOWED_ORIGINS) == 1 and ALLOWED_ORIGINS[0] == "*")

app = FastAPI(title="RAG Retrieval UI API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["http://localhost:8501"],
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class SlidingWindowRateLimiter:
    """Simple in-memory sliding-window limiter."""

    def __init__(self, max_requests: int, window_seconds: int = 60) -> None:
        self.max_requests = int(max_requests)
        self.window_seconds = int(window_seconds)
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> tuple[bool, int]:
        """
        Return (allowed, retry_after_seconds).
        retry_after_seconds is 0 when allowed.
        """
        if self.max_requests <= 0:
            return True, 0
        now = time.time()
        cutoff = now - float(self.window_seconds)
        with self._lock:
            q = self._events[key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= self.max_requests:
                retry_after = int(max(1, q[0] + self.window_seconds - now))
                return False, retry_after
            q.append(now)
            if not q:
                self._events.pop(key, None)
            return True, 0


rate_limiter = SlidingWindowRateLimiter(
    max_requests=RATE_LIMIT_REQ_PER_MIN,
    window_seconds=60,
)
read_rate_limiter = SlidingWindowRateLimiter(
    max_requests=RATE_LIMIT_READ_PER_MIN if RATE_LIMIT_READ_PER_MIN > 0 else RATE_LIMIT_REQ_PER_MIN,
    window_seconds=60,
)
upload_rate_limiter = SlidingWindowRateLimiter(
    max_requests=RATE_LIMIT_UPLOAD_PER_MIN if RATE_LIMIT_UPLOAD_PER_MIN > 0 else RATE_LIMIT_REQ_PER_MIN,
    window_seconds=60,
)
search_rate_limiter = SlidingWindowRateLimiter(
    max_requests=RATE_LIMIT_SEARCH_PER_MIN if RATE_LIMIT_SEARCH_PER_MIN > 0 else RATE_LIMIT_REQ_PER_MIN,
    window_seconds=60,
)
rank_rate_limiter = SlidingWindowRateLimiter(
    max_requests=RATE_LIMIT_RANK_PER_MIN if RATE_LIMIT_RANK_PER_MIN > 0 else RATE_LIMIT_REQ_PER_MIN,
    window_seconds=60,
)


def _require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")) -> None:
    """Require API key for sensitive/expensive endpoints when API_KEY is configured."""
    if not API_KEY:
        return
    if not x_api_key or not hmac.compare_digest(str(x_api_key), API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized.")


def _rate_limit_key(request: Request, x_api_key: Optional[str]) -> str:
    """Build stable limiter key from API key (when set) and client address."""
    client_host = request.client.host if request.client else "unknown"
    if API_KEY:
        return f"api_key:{(x_api_key or '').strip()}|ip:{client_host}"
    return f"ip:{client_host}"


def _enforce_rate_limit(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    """Enforce request budget per minute on protected endpoints."""
    key = _rate_limit_key(request, x_api_key)
    allowed, retry_after = rate_limiter.check(key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )


def _enforce_read_rate_limit(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    key = _rate_limit_key(request, x_api_key)
    allowed, retry_after = read_rate_limiter.check(key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for read endpoints. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )


def _enforce_upload_rate_limit(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    key = _rate_limit_key(request, x_api_key)
    allowed, retry_after = upload_rate_limiter.check(key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for uploads. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )


def _enforce_search_rate_limit(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    key = _rate_limit_key(request, x_api_key)
    allowed, retry_after = search_rate_limiter.check(key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for search. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )


def _enforce_rank_rate_limit(
    request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> None:
    key = _rate_limit_key(request, x_api_key)
    allowed, retry_after = rank_rate_limiter.check(key)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded for ranking. Try again later.",
            headers={"Retry-After": str(retry_after)},
        )


def _sanitize_upload_filename(filename: str, expected_ext: str) -> str:
    """Reject path traversal and non-portable filenames."""
    raw = str(filename or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Missing filename.")
    if Path(raw).name != raw:
        raise HTTPException(status_code=400, detail="Filename must not contain directory components.")
    if not SAFE_UPLOAD_FILENAME_RE.fullmatch(raw):
        raise HTTPException(status_code=400, detail="Filename contains invalid characters.")
    if not raw.lower().endswith(expected_ext):
        raise HTTPException(status_code=400, detail=f"Filename must end with {expected_ext}.")
    return raw


def _decode_base64_payload(payload: str, field_name: str) -> bytes:
    """Decode base64 and enforce payload size limits."""
    try:
        decoded = base64.b64decode(payload.encode("utf-8"), validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid {field_name}.") from e
    if len(decoded) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"{field_name} exceeds max size of {int(MAX_UPLOAD_MB)} MB.",
        )
    return decoded


@app.get("/api/v1/health")
def health() -> dict[str, bool]:
    """Health check endpoint."""
    return {"ok": True}


@app.get("/api/v1/metrics", response_model=MetricsResponse)
def get_metrics(
    _: None = Depends(_require_api_key),
    __: None = Depends(_enforce_read_rate_limit),
) -> MetricsResponse:
    """Return runtime generation observability counters and derived rates."""
    return MetricsResponse(**search_service.get_generation_observability_snapshot())


@app.get("/api/v1/docs")
def list_docs(
    _: None = Depends(_require_api_key),
    __: None = Depends(_enforce_read_rate_limit),
) -> dict[str, Any]:
    """List available processed documents for UI selection."""
    docs_detail = storage.list_docs_with_titles()
    return {
        "docs": [d["doc_id"] for d in docs_detail],
        "docs_detail": docs_detail,
    }


@app.post("/api/v1/docs/upload", response_model=UploadResponse)
async def upload_doc(
    req: UploadRequest,
    _: None = Depends(_require_api_key),
    __: None = Depends(_enforce_upload_rate_limit),
) -> UploadResponse:
    """
    Upload PDF (+ optional eval_set.json) as base64 JSON, run preprocessing and index build.
    """
    if DEMO_MODE:
        raise HTTPException(status_code=403, detail="Upload disabled in demo mode.")
    pdf_filename = _sanitize_upload_filename(req.pdf_filename, ".pdf")

    with tempfile.TemporaryDirectory(prefix="rag_ui_upload_") as td:
        tmp_dir = Path(td)
        pdf_path = tmp_dir / pdf_filename
        pdf_bytes = _decode_base64_payload(req.pdf_base64, "pdf_base64")
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        eval_path = None
        if req.eval_filename and req.eval_base64:
            eval_filename = _sanitize_upload_filename(req.eval_filename, ".json")
            eval_path = tmp_dir / eval_filename
            eval_bytes = _decode_base64_payload(req.eval_base64, "eval_base64")
            with open(eval_path, "wb") as f:
                f.write(eval_bytes)

        try:
            out = process_service.process_pdf(pdf_path=pdf_path, eval_set_path=eval_path)
            stats = storage.read_doc_stats(out["doc_id"])
        except Exception as e:
            raise HTTPException(status_code=500, detail="Pipeline failed.") from e

    return UploadResponse(
        doc_id=out["doc_id"],
        data_dir=out["data_dir"],
        page_count=stats["page_count"],
        chunk_count=stats["chunk_count"],
        table_chunk_count=stats["table_chunk_count"],
        chunk_size_tokens=stats.get("chunk_size_tokens"),
        chunk_overlap_tokens=stats.get("chunk_overlap_tokens"),
        has_eval_set=stats["has_eval_set"],
        has_pipeline_log=stats.get("has_pipeline_log", False),
        pipeline_log_path=out.get("pipeline_log_path", ""),
        status="ready",
    )


@app.get("/api/v1/docs/{doc_id}/stats")
def get_doc_stats(
    doc_id: str,
    _: None = Depends(_require_api_key),
    __: None = Depends(_enforce_read_rate_limit),
) -> dict[str, Any]:
    """Get page/chunk counts and availability flags for one processed document."""
    try:
        return storage.read_doc_stats(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/api/v1/docs/{doc_id}/eval-items")
def get_eval_items(
    doc_id: str,
    _: None = Depends(_require_api_key),
    __: None = Depends(_enforce_read_rate_limit),
) -> dict[str, list[dict[str, Any]]]:
    """Return eval questions and expected pages for UI highlighting."""
    try:
        items = storage.read_eval_items(doc_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    out = []
    for item in items:
        out.append(
            {
                "query_id": item.get("query_id"),
                "question": item.get("question"),
                "expected_pages": item.get("expected_pages", []),
            }
        )
    return {"items": out}


@app.get("/api/v1/docs/{doc_id}/logs")
def get_doc_logs(
    doc_id: str,
    last_n: int = Query(200, ge=1, le=MAX_LOG_TAIL_LINES),
    _: None = Depends(_require_api_key),
    __: None = Depends(_enforce_read_rate_limit),
) -> dict[str, Any]:
    """Return tail logs for one processed document."""
    try:
        return storage.read_doc_log_tail(doc_id, last_n=last_n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/api/v1/docs/{doc_id}/tables")
def get_doc_tables(
    doc_id: str,
    limit: int = Query(200, ge=1, le=MAX_TABLE_LIMIT),
    page: Optional[int] = Query(default=None, ge=1),
    _: None = Depends(_require_api_key),
    __: None = Depends(_enforce_read_rate_limit),
) -> dict[str, Any]:
    """Return extracted tables for one processed document."""
    try:
        return storage.read_tables_structured(doc_id=doc_id, limit=limit, page=page)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/api/v1/docs/{doc_id}/pages/{page_no}/chunks")
def get_doc_page_chunks(
    doc_id: str,
    page_no: int,
    _: None = Depends(_require_api_key),
    __: None = Depends(_enforce_read_rate_limit),
) -> dict[str, Any]:
    """Return per-page chunks with tokenized overlap metadata for UI inspection."""
    try:
        return storage.read_page_chunk_inspector(doc_id=doc_id, page=page_no)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/api/v1/docs/{doc_id}/search", response_model=SearchResponse)
def search_doc(
    doc_id: str,
    req: SearchRequest,
    _: None = Depends(_require_api_key),
    __: None = Depends(_enforce_search_rate_limit),
) -> SearchResponse:
    """Run top-k similarity search for a question against one document."""
    data_dir = storage.doc_dir(doc_id)
    if not data_dir.exists():
        raise HTTPException(status_code=404, detail=f"Unknown doc_id: {doc_id}")
    try:
        return search_service.search(
            data_dir=data_dir,
            question=req.question,
            k=req.k,
            query_id=req.query_id,
            include_generated_answer=bool(req.include_generated_answer),
            retrieval_scope=req.retrieval_scope,
            lexical_scope=req.lexical_scope,
            filters={
                "doc_id": req.filter_doc_id,
                "trust_id": req.filter_trust_id,
                "year": req.filter_year,
                "is_table": req.filter_is_table,
                "section_contains": req.filter_section_contains,
                "subsection_contains": req.filter_subsection_contains,
            },
            generation_overrides={
                "max_context_chunks": req.gen_max_context_chunks,
                "max_context_chars": req.gen_max_context_chars,
                "max_chunk_chars": req.gen_max_chunk_chars,
                "timeout_seconds": req.gen_timeout_seconds,
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="Search failed.") from e


@app.post("/api/v1/docs/rank")
def rank_docs(
    req: DocRankRequest,
    _: None = Depends(_require_api_key),
    __: None = Depends(_enforce_rank_rate_limit),
) -> dict[str, Any]:
    """Rank all processed documents by top-1 similarity for the given query."""
    docs_detail = storage.list_docs_with_titles()
    if not docs_detail:
        return {
            "question": req.question,
            "requested_top_n": int(req.top_n),
            "total_docs": 0,
            "ranked_docs": 0,
            "items": [],
            "skipped_docs": [],
        }

    items: list[dict[str, Any]] = []
    skipped_docs: list[dict[str, str]] = []
    for doc in docs_detail:
        doc_id = str(doc.get("doc_id") or "").strip()
        if not doc_id:
            continue
        data_dir = storage.doc_dir(doc_id)
        if not data_dir.exists():
            skipped_docs.append({"doc_id": doc_id, "reason": "missing_data_dir"})
            continue
        try:
            out = search_service.search(
                data_dir=data_dir,
                question=req.question,
                k=1,
                query_id=None,
            )
            results = out.get("results", []) if isinstance(out, dict) else []
            top_score = float(results[0].get("score")) if results else None
            top_chunk_id = str(results[0].get("chunk_id") or "") if results else ""
            top_pages = results[0].get("pages", []) if results else []
            items.append(
                {
                    "doc_id": doc_id,
                    "title": str(doc.get("title") or doc_id),
                    "similarity": top_score,
                    "top_chunk_id": top_chunk_id,
                    "top_pages": top_pages,
                }
            )
        except Exception as e:
            # Keep ranking robust even if one document is malformed.
            skipped_docs.append({"doc_id": doc_id, "reason": f"search_error:{type(e).__name__}"})
            continue

    ranked = [x for x in items if x.get("similarity") is not None]
    ranked.sort(key=lambda x: float(x["similarity"]), reverse=True)
    ranked = ranked[: int(req.top_n)]
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return {
        "question": req.question,
        "requested_top_n": int(req.top_n),
        "total_docs": len(docs_detail),
        "ranked_docs": len(ranked),
        "items": ranked,
        "skipped_docs": skipped_docs,
    }
