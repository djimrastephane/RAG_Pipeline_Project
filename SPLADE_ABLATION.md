# SPLADE Ablation Protocol

## Purpose
Add and evaluate `Dense + SPLADE + RRF` retrieval as an alternative to `Dense + BM25 + RRF`.

## Pipeline Mode
`run_retrieval_ablation.py` now supports:
- `mode: splade_hybrid`

Evaluator script:
- `scripts/retrieval_eval_splade_hybrid.py`

## Required Inputs per data dir
- `faiss.index`
- `chunk_meta.parquet`
- `chunks.parquet`
- `eval_set.json`

## Recommended Run Command
```bash
.venv/bin/python scripts/run_retrieval_ablation.py \
  --config configs/retrieval_tuning_splade_template.yaml
```

Run only SPLADE experiments:
```bash
.venv/bin/python scripts/run_retrieval_ablation.py \
  --config configs/retrieval_tuning_splade_template.yaml \
  --only hybrid_dense_splade_rrf
```

## SPLADE Parameters (config keys)
- `splade_model`
- `splade_device` (`auto|cpu|cuda|mps`)
- `splade_local_only` (`true|false`)
- `splade_max_length`
- `splade_doc_batch_size`
- `splade_query_batch_size`
- `splade_doc_top_terms`
- `splade_query_top_terms`
- `splade_min_weight`

RRF parameters:
- `rrf_k`
- `dense_weight`
- `splade_weight`

Optional local cross-encoder reranker:
- `cross_encoder.enabled`
- `cross_encoder.model`
- `cross_encoder.topn`
- `cross_encoder.weight`

## Outputs
Per run data directory (`<run>/<doc_id>/`):
- `retrieval_results_splade_hybrid.json`
- `retrieval_metrics_splade_hybrid.json`
- `retrieval_summary_splade_hybrid.csv`

Ablation-level outputs (`output_dir`):
- `retrieval_ablation_summary.csv`
- `retrieval_ablation_best_by_k.csv`
- `retrieval_ablation_summary.json`

## Comparison Checklist
1. Compare `Hit@1`, `MRR@10`, and `Hit@3` vs current hybrid BM25 baseline.
2. Compare FP2 exact metrics (`gold in top100 pre but not top1 post`) for rerank regret sensitivity.
3. Break down by difficulty (`LEX|MOD|STR`) and route intent.
4. Inspect query-level deltas for worst regressions before adopting.

## Reproducibility Notes
- Prefer local SPLADE model path and `splade_local_only: true` for deterministic reruns.
- Keep `eval_set.json` version fixed per ablation wave.
- Record exact config path and commit hash in experiment logs/reports.

## Failure Modes
- If SPLADE model is unavailable locally and downloads are blocked, run fails at model load.
- If GPU memory is constrained, reduce `splade_doc_batch_size` and `splade_query_batch_size`.
- If sparse vectors are too dense/slow, lower `splade_doc_top_terms` and/or raise `splade_min_weight`.
