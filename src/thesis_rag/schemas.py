from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _dataclass_to_dict(instance: Any) -> dict[str, Any]:
    return _json_safe(asdict(instance))


@dataclass(slots=True)
class PathsConfig:
    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    processed_dir: Path = Path("processed")
    indexes_dir: Path = Path("indexes")
    runs_dir: Path = Path("runs")
    query_set_path: Path = Path("data/eval_set.json")
    model_cache_dir: Path = Path("models")


@dataclass(slots=True)
class RuntimeConfig:
    device: str = "cpu"
    random_seed: int = 13
    deterministic_torch: bool = True
    offline: bool = True
    log_level: str = "INFO"
    corpus_name: str = "default-corpus"
    dataset_version: str = "unknown"


@dataclass(slots=True)
class OCRConfig:
    enabled: bool = True
    min_chars_before_fallback: int = 80
    min_alpha_ratio: float = 0.30
    min_digit_ratio: float = 0.60


@dataclass(slots=True)
class ChunkingConfig:
    chunk_size_tokens: int = 224
    chunk_overlap_tokens: int = 56
    min_chunk_words: int = 20
    table_chunking_strategy: str = "baseline"


@dataclass(slots=True)
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize_embeddings: bool = True
    batch_size: int = 32
    expected_dimension: int = 384


@dataclass(slots=True)
class FaissConfig:
    index_type: str = "IndexFlatIP"


@dataclass(slots=True)
class BM25Config:
    k1: float = 1.5
    b: float = 0.75


@dataclass(slots=True)
class RetrievalConfig:
    dense_top_k: int = 20
    sparse_top_k: int = 20
    hybrid_top_k: int = 20
    rrf_k: int = 20
    dense_weight: float = 0.5
    sparse_weight: float = 2.0


@dataclass(slots=True)
class EvaluationConfig:
    ks: list[int] = field(default_factory=lambda: [1, 3, 5, 10])


@dataclass(slots=True)
class PipelineConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    faiss: FaissConfig = field(default_factory=FaissConfig)
    bm25: BM25Config = field(default_factory=BM25Config)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    pdf_path: str


@dataclass(slots=True)
class PageRecord:
    page_id: str
    doc_id: str
    page_number: int
    raw_text: str
    clean_text: str
    extractor_used: str
    quality_note: str
    ocr_used: bool
    is_table: bool = False
    table_type: Optional[str] = None
    heading_candidates: list[str] = field(default_factory=list)
    top_lines: list[dict[str, Any]] = field(default_factory=list)
    header_lines_removed: list[str] = field(default_factory=list)
    footer_lines_removed: list[str] = field(default_factory=list)
    rotation: int = 0
    page_width: float = 0.0
    page_height: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    page_number: int
    chunk_index: int
    text: str
    token_count: int
    word_count: int
    chunk_id_global: Optional[str] = None
    page_start: int | None = None
    page_end: int | None = None
    pages: list[int] = field(default_factory=list)
    part: Optional[str] = None
    section_title: Optional[str] = None
    subsection_title: Optional[str] = None
    is_table: bool = False
    table_type: Optional[str] = None
    table_chunk_kind: Optional[str] = None
    segment_boundary_type: Optional[str] = None
    segment_has_search_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(slots=True)
class RetrievalHit:
    query_id: str
    query_text: str
    rank: int
    score: float
    retrieval_method: str
    doc_id: str
    page_number: int
    chunk_id: Optional[str]
    pages: list[int] = field(default_factory=list)
    text: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(slots=True)
class QueryRecord:
    query_id: str
    query_text: str
    doc_id: str
    gold_pages: list[int]
    expected_answer: Optional[str] = None
    difficulty: Optional[str] = None
    evidence_layout: Optional[str] = None
    expected_section: Optional[str] = None
    expected_subsection: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(slots=True)
class EvaluationResult:
    query_id: str
    doc_id: str
    gold_pages: list[int]
    predicted_pages: list[int]
    hit_at_1: bool
    hit_at_3: bool
    reciprocal_rank: float
    first_relevant_rank: Optional[int]
    failure_type: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(slots=True)
class QueryDiagnostics:
    query_id: str
    query_text: str
    doc_id: str
    gold_pages: list[int]
    dense_top_k_pages: list[int]
    bm25_top_k_pages: list[int]
    hybrid_top_k_pages: list[int]
    hit_at_1: bool
    hit_at_3: bool
    reciprocal_rank: float
    dense_top1_score: Optional[float]
    dense_top2_score: Optional[float]
    dense_margin: Optional[float]
    hybrid_top1_item: Optional[str]
    evidence_layout: Optional[str]
    difficulty: Optional[str]
    failure_type: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(slots=True)
class RunMetadata:
    run_id: str
    timestamp_utc: str
    config: dict[str, Any]
    corpus_name: str
    dataset_version: str
    number_of_documents: int
    number_of_pages: int
    number_of_chunks: int
    number_of_ocr_pages: int
    model_name: str
    git_commit_hash: Optional[str]
    final_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)
