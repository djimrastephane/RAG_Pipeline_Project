from __future__ import annotations

import logging
from pathlib import Path

import faiss

from .artifacts import load_chunks, load_queries, save_chunks, save_hits, save_manifest, save_pages
from .diagnostics import build_query_diagnostics, save_diagnostics_csv
from .embedding import embed_chunks, embed_queries
from .evaluator import aggregate_metrics, evaluate_page_hits
from .fusion import reciprocal_rank_fusion
from .indexing import build_faiss_index, save_chunk_metadata, save_embeddings, save_faiss_index
from .loader import discover_documents, extract_page_structures
from .preprocessing import build_chunk_records, build_page_records
from .ranking import chunk_hits_to_page_hits
from .retrieval_dense import dense_retrieve_legacy_style, search_faiss_stably
from .retrieval_hybrid import hybrid_retrieve_legacy_style
from .retrieval_sparse import build_bm25, sparse_retrieve_legacy_style
from .schemas import PipelineConfig, RunMetadata
from .utils import (
    configure_logging,
    dependency_check,
    git_commit_hash,
    make_run_dir,
    now_utc_iso,
    resolve_device,
    set_global_determinism,
    write_json,
)

LOGGER = logging.getLogger(__name__)

REQUIRED_MODULES = {
    "numpy": "1.26.4",
    "pandas": "2.3.3",
    "faiss": "",
    "yaml": "6.0.3",
    "pymupdf": "1.26.7",
    "pdfplumber": "0.11.9",
    "sentence_transformers": "5.2.2",
}


def prepare_run(config: PipelineConfig, prefix: str) -> Path:
    run_dir = make_run_dir(config.paths.runs_dir, prefix=prefix)
    configure_logging(run_dir / "pipeline.log", config.runtime.log_level)
    set_global_determinism(config.runtime.random_seed, config.runtime.deterministic_torch)
    dependency_check(REQUIRED_MODULES)
    write_json(run_dir / "config.json", config.to_dict())
    return run_dir


def preprocess_corpus(config: PipelineConfig) -> Path:
    run_dir = prepare_run(config, prefix="preprocess")
    documents = discover_documents(config.paths.data_dir)
    if not documents:
        raise FileNotFoundError(f"No PDFs found in {config.paths.data_dir}")
    total_pages = 0
    total_chunks = 0
    total_ocr_pages = 0
    for document in documents:
        doc_out = run_dir / document.doc_id
        page_structs = extract_page_structures(document)
        pages = build_page_records(document.doc_id, page_structs, config.ocr)
        chunks = build_chunk_records(document.doc_id, pages, config.chunking, source_pdf_path=document.pdf_path)
        save_pages(pages, doc_out)
        save_chunks(chunks, doc_out)
        total_pages += len(pages)
        total_chunks += len(chunks)
        total_ocr_pages += sum(page.ocr_used for page in pages)
    manifest = RunMetadata(
        run_id=run_dir.name,
        timestamp_utc=now_utc_iso(),
        config=config.to_dict(),
        corpus_name=config.runtime.corpus_name,
        dataset_version=config.runtime.dataset_version,
        number_of_documents=len(documents),
        number_of_pages=total_pages,
        number_of_chunks=total_chunks,
        number_of_ocr_pages=total_ocr_pages,
        model_name=config.embedding.model_name,
        git_commit_hash=git_commit_hash(config.paths.project_root),
    )
    save_manifest(manifest.to_dict(), run_dir / "run_manifest.json")
    LOGGER.info("Preprocessing finished: docs=%s pages=%s chunks=%s", len(documents), total_pages, total_chunks)
    return run_dir


def build_indexes(config: PipelineConfig, chunks_path: Path) -> Path:
    run_dir = prepare_run(config, prefix="index")
    chunks = load_chunks(chunks_path)
    device = resolve_device(config.runtime.device)
    vectors = embed_chunks(
        chunks,
        config.embedding,
        device=device,
        cache_dir=str(config.paths.model_cache_dir),
    )
    index = build_faiss_index(vectors, config.faiss)
    save_embeddings(vectors, run_dir / "embeddings.npy")
    save_faiss_index(index, run_dir / "faiss.index")
    save_chunk_metadata(chunks, run_dir / "chunk_metadata.parquet")
    write_json(
        run_dir / "index_manifest.json",
        {
            "chunk_count": len(chunks),
            "embedding_dimension": int(vectors.shape[1]),
            "faiss_ntotal": int(index.ntotal),
            "device": device,
            "model_name": config.embedding.model_name,
        },
    )
    LOGGER.info("Index build finished for %s chunks", len(chunks))
    return run_dir


def retrieve_queries(config: PipelineConfig, chunk_metadata_path: Path, faiss_index_path: Path, query_set_path: Path) -> Path:
    run_dir = prepare_run(config, prefix="retrieve")
    chunks = load_chunks(chunk_metadata_path)
    queries = load_queries(query_set_path)
    _validate_queries(queries, chunks)
    index = faiss.read_index(str(faiss_index_path))
    device = resolve_device(config.runtime.device)
    query_vectors = embed_queries(
        [query.query_text for query in queries],
        config.embedding,
        device=device,
        cache_dir=str(config.paths.model_cache_dir),
    )
    dense_chunk_hits = dense_retrieve_legacy_style(
        index,
        chunks,
        queries,
        query_vectors,
        top_k=config.retrieval.dense_top_k,
        max_k_search=max(100, config.retrieval.hybrid_top_k),
    )
    dense_page_hits = chunk_hits_to_page_hits(
        dense_chunk_hits,
        "dense_pages",
        chunk_limit=config.retrieval.dense_top_k,
    )
    bm25 = build_bm25(chunks, config.bm25)
    sparse_chunk_hits = sparse_retrieve_legacy_style(bm25, chunks, queries, top_k=max(100, config.retrieval.sparse_top_k))
    sparse_page_hits = chunk_hits_to_page_hits(
        sparse_chunk_hits,
        "bm25_pages",
        chunk_limit=config.retrieval.sparse_top_k,
    )
    raw_dense_scores, raw_dense_indices = search_faiss_stably(
        index,
        query_vectors,
        min(max(100, config.retrieval.hybrid_top_k), len(chunks)),
    )
    _dense_for_hybrid, _bm25_for_hybrid, fused_chunk_hits = hybrid_retrieve_legacy_style(
        chunks=chunks,
        queries=queries,
        dense_scores=raw_dense_scores,
        dense_indices=raw_dense_indices,
        bm25=bm25,
        max_k_search=max(100, config.retrieval.hybrid_top_k),
        dense_weight=config.retrieval.dense_weight,
        bm25_weight=config.retrieval.sparse_weight,
        rrf_k=config.retrieval.rrf_k,
    )
    fused_page_hits = chunk_hits_to_page_hits(
        fused_chunk_hits,
        "hybrid_pages",
        chunk_limit=config.retrieval.hybrid_top_k,
    )
    save_hits(dense_page_hits, run_dir / "dense_page_hits.jsonl")
    save_hits(sparse_page_hits, run_dir / "bm25_page_hits.jsonl")
    save_hits(fused_page_hits, run_dir / "hybrid_page_hits.jsonl")
    write_json(
        run_dir / "retrieval_manifest.json",
        {
            "query_count": len(queries),
            "dense_top_k": config.retrieval.dense_top_k,
            "sparse_top_k": config.retrieval.sparse_top_k,
            "hybrid_top_k": config.retrieval.hybrid_top_k,
            "rrf_k": config.retrieval.rrf_k,
        },
    )
    return run_dir


def evaluate_retrieval(
    config: PipelineConfig,
    query_set_path: Path,
    dense_hits_path: Path,
    sparse_hits_path: Path,
    hybrid_hits_path: Path,
) -> Path:
    run_dir = prepare_run(config, prefix="evaluate")
    queries = load_queries(query_set_path)
    dense_hits = _load_hits_csv_or_jsonl(dense_hits_path)
    sparse_hits = _load_hits_csv_or_jsonl(sparse_hits_path)
    hybrid_hits = _load_hits_csv_or_jsonl(hybrid_hits_path)
    results = evaluate_page_hits(queries, hybrid_hits)
    metrics = aggregate_metrics(results, config.evaluation.ks)
    diagnostics = build_query_diagnostics(queries, dense_hits, sparse_hits, hybrid_hits, results)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "per_query_results.json", [result.to_dict() for result in results])
    save_diagnostics_csv(diagnostics, run_dir / "diagnostics.csv")
    manifest = RunMetadata(
        run_id=run_dir.name,
        timestamp_utc=now_utc_iso(),
        config=config.to_dict(),
        corpus_name=config.runtime.corpus_name,
        dataset_version=config.runtime.dataset_version,
        number_of_documents=len({query.doc_id for query in queries}),
        number_of_pages=0,
        number_of_chunks=0,
        number_of_ocr_pages=0,
        model_name=config.embedding.model_name,
        git_commit_hash=git_commit_hash(config.paths.project_root),
        final_metrics=metrics,
    )
    save_manifest(manifest.to_dict(), run_dir / "run_manifest.json")
    return run_dir


def _validate_queries(queries, chunks) -> None:
    doc_to_pages: dict[str, set[int]] = {}
    for chunk in chunks:
        doc_to_pages.setdefault(chunk.doc_id, set()).add(chunk.page_number)
    for query in queries:
        if not query.gold_pages:
            raise ValueError(f"Query {query.query_id} has no gold pages.")
        available_pages = doc_to_pages.get(query.doc_id, set())
        missing = sorted(set(query.gold_pages) - available_pages)
        if missing:
            raise ValueError(f"Query {query.query_id} references missing gold pages: {missing}")


def _load_hits_csv_or_jsonl(path: Path):
    import pandas as pd

    frame = pd.read_csv(path.with_suffix(".csv")) if path.suffix == ".jsonl" else pd.read_csv(path)
    from .schemas import RetrievalHit

    return [RetrievalHit(**row) for row in frame.to_dict(orient="records")]
