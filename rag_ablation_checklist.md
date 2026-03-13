# RAG Ablation Checklist (Project-Specific)

This checklist defines the recommended order of ablations for this repository, with script/file pointers and expected outputs.

## 1) Corpus & Chunking Baseline
Goal: lock stable chunk construction before retrieval tuning.

Run:
- `scripts/preprocess_hybrid.py`
- `scripts/build_index.py`
- Optional batch orchestration: `scripts/run_retrieval_ablation.py`

Ablate:
- `CHUNK_SIZE_TOKENS` (e.g., 280 vs 320)
- `CHUNK_OVERLAP_TOKENS` (e.g., 90)
- `SEGMENT_AWARE_CHUNKING` on/off
- markdown table/header injection flags if used

Track:
- retrieval hit/mrr downstream (k=1,3,5,10)
- chunk counts, table/text balance, OCR counters in `metrics.json`

Outputs:
- per-doc: `chunks.parquet`, `chunk_meta.parquet`, `faiss.index`
- eval summaries: `retrieval_summary_*.csv`, `retrieval_metrics_*.json`

---

## 2) Retriever Baselines (No Fusion Tuning Yet)
Goal: establish reference performance ladder.

Run:
- Dense: `scripts/retrieval_eval.py`
- BM25: `scripts/retrieval_eval_bm25.py`
- Hybrid defaults: `scripts/retrieval_eval_hybrid.py`

Track:
- `Hit@1`, `Hit@3`, `MRR@10`
- FP2 count (`failure_type=FP2_MISSED_TOP_RANK`)

Outputs:
- `retrieval_summary.csv`, `retrieval_summary_bm25.csv`, `retrieval_summary_hybrid.csv`

---

## 3) Hybrid Fusion Tuning (Stage-1 Ranking)
Goal: optimize first-stage candidate ranking.

Run:
- `scripts/tune_hybrid_rrf_weights.py` (single split)
- `scripts/tune_hybrid_rrf_weights_cv.py` (preferred)

Ablate:
- `rrf_k`
- `dense_weight`
- `bm25_weight`

Track:
- mean and std of test `Hit@1`, `MRR@10`
- per-doc delta and variance

Outputs:
- tuning folders under `data_processed/.../final_selection/hybrid_weight_tuning*`

---

## 4) Candidate Depth / Search Breadth
Goal: validate `MAX_K_SEARCH` tradeoff.

Run:
- `scripts/ablate_max_k_search.py`

Ablate:
- `MAX_K_SEARCH` (e.g., 25, 50, 100, 150, 200)

Track:
- `Hit@1`, `Hit@3`, `MRR@10`
- edge-case regressions per doc
- latency trend

Outputs:
- `max_k_search_sensitivity_*` CSV/JSON/charts

---

## 5) Reranker / Boost Tuning (Stage-2 Ranking)
Goal: improve top-rank ordering after fusion.

Run:
- Hybrid eval with lexical boosts: `scripts/retrieval_eval_hybrid.py`
- Optional CE rerank on/off: `--enable-cross-encoder-rerank`
- Grid/CV loops (custom scripts used in this project)

Ablate:
- `ENTITY_MATCH_BOOST`
- `NUMERIC_DENSITY_BOOST`
- `MAX_ENTITY_MATCHES`
- CE topN/weight if CE enabled

Track:
- FP2 delta vs tuned hybrid baseline
- swap analysis (rank2->rank1 vs miss->hit)

Outputs:
- comparison CSVs in `final_selection/*rerank*`

---

## 6) OCR / Extraction Threshold Sensitivity
Goal: validate preprocessing fallback knobs on OCR-heavy docs.

Run:
- `scripts/preprocess_hybrid.py` with `--fallback-min-chars`
- then `scripts/build_index.py` + `scripts/retrieval_eval_hybrid.py`

Ablate:
- `FALLBACK_MIN_CHARS` (e.g., 50, 80, 100, 150, 200)

Track:
- retrieval metrics (Hit/MRR)
- extractor diagnostics (`fallback_used`, `too_short<...`, OCR counters)

Outputs:
- `fallback_min_chars_ablation_*` folders with metrics/charts

---

## 7) Failure-Slice Diagnostics
Goal: ensure improvements are broad, not narrow.

Run:
- existing diagnostics scripts/output processing in `final_selection/diagnostics`

Break down by:
- difficulty buckets: `LEX`, `MOD`, `STR`
- failure points: `FP1` ... `FP7`
- document-level deltas

Track:
- concentration risk (e.g., only LEX improves)

---

## 8) Statistical Validation
Goal: test stability/significance of observed deltas.

Run:
- paired comparisons + bootstrap loops (project uses ad-hoc scripts/commands)

Required:
- bootstrap with replacement (>=1000)
- paired delta CI and p-values for key metrics

Track:
- CI crossing 0
- probability(delta > 0)

---

## 9) Production Freeze
Goal: one stable, documented setting.

Do:
- select final config from CV + diagnostics
- write a production recommendation markdown
- store all comparison tables/charts in `final_selection/`

Include:
- chosen chunking
- chosen fusion weights and `rrf_k`
- chosen rerank parameters
- chosen `MAX_K_SEARCH`
- known residual risks

---

## Minimal “Must-Have” Sequence
1. Chunking ablation
2. Dense/BM25/Hybrid baseline
3. Hybrid CV tuning (`rrf_k`, weights)
4. `MAX_K_SEARCH` sensitivity
5. Reranker tuning + FP2 diagnostics
6. Bootstrap/paired validation
7. Production freeze

