# Pipeline Function Map (Ingestion -> Generation)

| Stage | Function | Brief Description | File |
|---|---|---|---|
| Ingestion | `parse_args` | Reads preprocessing CLI args (PDF path, chunk settings, modes). | `scripts/preprocess_hybrid.py` |
| Ingestion | `main` | Orchestrates full preprocessing flow. | `scripts/preprocess_hybrid.py` |
| Config | `_apply_config_overrides` | Propagates runtime config values into extraction/table modules. | `scripts/preprocess_hybrid.py` |
| Page extraction | `extract_page_struct_hybrid` | Hybrid extraction per page (PyMuPDF/pdfplumber/OCR fallback). | `src/rag_pdf/extract_page.py` |
| OCR fallback | `extract_page_with_ocr` | OCR-based fallback text extraction. | `src/rag_pdf/extract_page.py` |
| Boilerplate cleanup | `strip_by_coordinates` | Removes coordinate-based page headers/footers. | `src/rag_pdf/boilerplate.py` |
| Boilerplate cleanup | `remove_repeated_header_footer_lines` | Removes repeated lines across pages. | `src/rag_pdf/boilerplate.py` |
| Heading detection | `select_heading_candidates` | Finds heading-like lines for section inference. | `src/rag_pdf/headings.py` |
| Section building | `build_sections_from_pages` | Builds part/section/subsection timeline. | `src/rag_pdf/sections.py` |
| Section lookup | `find_section_for_page` | Maps each page to current section/subsection. | `src/rag_pdf/sections.py` |
| Table detection | `classify_page_content` | Classifies page as text/table-like and related flags. | `src/rag_pdf/table_detect.py` |
| Table extraction | `process_table_pages` | Extracts table chunks, summaries, and markdown views. | `src/rag_pdf/table_extract.py` |
| Table canonicalization | `extract_table_facts_from_markdown` | Converts table markdown into canonical facts. | `src/rag_pdf/table_canonicalize.py` |
| Table canonicalization | `extract_table_facts_from_dataframe` | Converts dataframe tables into canonical facts. | `src/rag_pdf/table_canonicalize.py` |
| Segment split | `split_text_for_segment_aware_chunking` | Splits text into structure-aware segments. | `src/rag_pdf/chunking.py` |
| Token chunking | `chunk_text_by_tokens` | Creates overlapping token chunks. | `src/rag_pdf/chunking.py` |
| Chunk assembly | `_build_page_chunks` | Wraps segment split + token chunking for a page. | `scripts/preprocess_hybrid.py` |
| Chunk assembly | `_append_text_chunks_for_page` | Writes text chunk records with metadata/page grounding. | `scripts/preprocess_hybrid.py` |
| Indexing | `main` | Entry point for embedding + index construction. | `scripts/build_index.py` |
| Indexing | `iter_document_dirs` | Finds document folders to index. | `scripts/build_index.py` |
| Indexing | `build_embedding_text` | Builds embedding input text from chunk + headings. | `scripts/build_index.py` |
| Indexing | `build_meta_table` | Produces retrieval metadata table aligned with FAISS rows. | `scripts/build_index.py` |
| Indexing | `build_index_for_doc` | Creates `embeddings.npy`, `faiss.index`, `chunk_meta.parquet`. | `scripts/build_index.py` |
| Retrieval entry | `SearchService.search` | Runtime retrieval pipeline entry. | `src/rag_pdf/services/search_service.py` |
| Retrieval load | `_load_doc` | Loads per-document retrieval artifacts. | `src/rag_pdf/services/search_service.py` |
| Retrieval load | `_load_global` | Loads global multi-document retrieval artifacts. | `src/rag_pdf/services/search_service.py` |
| Candidate filtering | `_apply_filters` | Applies metadata filters (doc/year/section/table/etc.). | `src/rag_pdf/services/search_service.py` |
| Dense ranking | `Index.search` (inside `search`) | Computes dense top candidates from FAISS. | `src/rag_pdf/services/search_service.py` |
| Lexical ranking | `BM25Index.score_query` | Computes BM25 lexical scores for candidates. | `src/rag_pdf/services/search_service.py` |
| Fusion | `rrf_fuse` | Combines dense + BM25 rank lists via RRF. | `src/rag_pdf/services/search_service.py` |
| Heuristic rerank | `query_overlap_boost` | Adds overlap-based boost for matched query entities. | `src/rag_pdf/retrieval/rerank.py` |
| Heuristic rerank | `numeric_density_boost` | Boosts numeric-dense chunks for numeric questions. | `src/rag_pdf/retrieval/rerank.py` |
| Heuristic rerank | `table_priority_boost` | Boosts table chunks for table_metric intents. | `src/rag_pdf/retrieval/rerank.py` |
| Optional CE rerank | Cross-encoder step in `search` | Re-scores top fused candidates with local cross-encoder. | `src/rag_pdf/services/search_service.py` |
| Extractive answer | `_predict_answer` | Chooses best extractive answer from retrieved chunks. | `src/rag_pdf/services/search_service.py` |
| Generative prompt | `_build_local_generation_prompt` | Builds grounded prompt with chunk/page citations. | `src/rag_pdf/services/search_service.py` |
| Local generation | `_generate_local_answer` | Calls local LLM for JSON answer + citations. | `src/rag_pdf/services/search_service.py` |
| Post-processing | `_parse_generation_json_payload` | Parses generation output into structured fields. | `src/rag_pdf/services/search_service.py` |
| Citation checks | `_validate_citations` | Validates generated citations against retrieved context. | `src/rag_pdf/services/search_service.py` |
| Observability | `_update_generation_observability` | Tracks generation status/latency counters. | `src/rag_pdf/services/search_service.py` |
| Observability | `get_generation_observability_snapshot` | Returns current generation observability snapshot. | `src/rag_pdf/services/search_service.py` |
