from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    """Request body for top-k retrieval."""

    question: str = Field(..., min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=50)
    query_id: Optional[str] = None
    include_generated_answer: bool = False
    retrieval_scope: str = Field("doc", description="One of: doc, trust, global")
    lexical_scope: str = Field("doc", description="One of: doc, trust, global")
    filter_doc_id: Optional[str] = None
    filter_trust_id: Optional[str] = None
    filter_year: Optional[int] = None
    filter_is_table: Optional[bool] = None
    filter_section_contains: Optional[str] = None
    filter_subsection_contains: Optional[str] = None
    gen_max_context_chunks: Optional[int] = Field(None, ge=1, le=20)
    gen_max_context_chars: Optional[int] = Field(None, ge=1000, le=50000)
    gen_max_chunk_chars: Optional[int] = Field(None, ge=200, le=10000)
    gen_timeout_seconds: Optional[float] = Field(None, ge=1.0, le=300.0)


class DocRankRequest(BaseModel):
    """Request body for ranking processed documents by query similarity."""

    question: str = Field(..., min_length=1, max_length=2000)
    top_n: int = Field(10, ge=1, le=100)


class UploadResponse(BaseModel):
    """Response payload for successful upload/process request."""

    doc_id: str
    page_count: int
    chunk_count: int
    table_chunk_count: int
    chunk_size_tokens: Optional[int] = None
    chunk_overlap_tokens: Optional[int] = None
    table_chunking: Optional[str] = None
    has_eval_set: bool
    has_pipeline_log: bool
    status: str


class UploadRequest(BaseModel):
    """Request payload for JSON/base64 document upload."""

    pdf_filename: str = Field(..., min_length=1)
    pdf_base64: str = Field(..., min_length=1)
    eval_filename: Optional[str] = None
    eval_base64: Optional[str] = None


class SearchCitation(BaseModel):
    """Validated citation emitted from generated answer text."""

    chunk_id: str
    page: int


class SearchResponse(BaseModel):
    """Response payload for retrieval + optional generated answer."""

    question: str
    k: int
    retrieval_mode: str
    retrieval_config: dict[str, Any]
    retrieval_scope: str
    lexical_scope: str
    filters_applied: dict[str, Any]
    query_id: Optional[str] = None
    expected_pages: list[int] = []
    expected_answer: Optional[str] = None
    expected_subsection: Optional[str] = None
    answer_type: Optional[str] = None
    hit_at_k: bool
    predicted_answer: Optional[str] = None
    predicted_answer_raw: Optional[str] = None
    answer_source_chunk_id: Optional[str] = None
    answer_debug: dict[str, Any] = {}
    include_generated_answer: bool = False
    generated_answer: Optional[str] = None
    generated_answer_raw: Optional[str] = None
    generated_citations: list[SearchCitation] = []
    generation_status: str = "skipped"
    generation_confidence: Optional[float] = None
    generation_debug: dict[str, Any] = {}
    results: list[dict[str, Any]] = []


class MetricsResponse(BaseModel):
    """Response payload for runtime observability metrics."""

    generation_counts: dict[str, int]
    citation_counts: dict[str, int]
    derived: dict[str, Optional[float]]
