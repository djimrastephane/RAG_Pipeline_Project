# Hybrid RRF Default Parameters

Last updated: 2026-02-26

## Default values
Use these defaults for Dense+BM25 hybrid retrieval unless an experiment overrides them:

- `rrf_k = 20`
- `dense_weight = 0.5`
- `bm25_weight = 2.0`

## Selection protocol
These defaults were selected from repeated cross-validation over the complete ground-truth set:

- Documents: 5 (`Grampian-2020-2021` to `Grampian-2024-2025`)
- Queries: 250
- CV protocol: 5 repeats x 5 folds
- Grid size: 180 configs (`5 rrf_k x 6 dense_weight x 6 bm25_weight`)
- Objective: maximize mean test `Hit@1`, tie-break by `MRR@10`, then stability

## Key result (vs previous baseline)
Previous baseline:
- `rrf_k = 60`, `dense_weight = 1.0`, `bm25_weight = 1.0`

Selected defaults:
- `rrf_k = 20`, `dense_weight = 0.5`, `bm25_weight = 2.0`

Performance deltas on repeated CV aggregate:
- `Hit@1`: `0.7577` vs `0.7335` (`+0.0242`)
- `MRR@10`: `0.8208` vs `0.8177` (`+0.0031`)

## Source artifacts
- `data_processed/ablation_thesis_5docs_q50/final_selection/hybrid_weight_tuning_cv_rerun_2026-02-26/hybrid_weight_cv_summary.json`
- `data_processed/ablation_thesis_5docs_q50/final_selection/hybrid_weight_tuning_cv_rerun_2026-02-26/hybrid_weight_cv_config_aggregate.csv`

## Where defaults are set in code
- `src/rag_pdf/services/search_service.py`
- `scripts/retrieval_eval_hybrid.py`
- `scripts/run_retrieval_ablation.py` (fallbacks when config omits values)
