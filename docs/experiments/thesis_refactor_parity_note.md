# Thesis Refactor Parity Note

This note records the final parity status of the `src/thesis_rag/` refactor against the legacy retrieval pipeline on a fixed benchmark subset.

## Benchmark Scope

- Corpus: `Grampian-2022-2023`
- Query subset: first 10 queries from `data_processed/Grampian-2022-2023/eval_set.json`
- Final benchmark run: [runs/parity_validation/manual_2026-04-15_grampian_subset10_fix10b](/Users/djimra/MSc%20Data%20Science%20Jan%202025/Thesis%20documents/RAG_Pipeline_Project/runs/parity_validation/manual_2026-04-15_grampian_subset10_fix10b:1)
- Final benchmark summary: [parity_report_fix10b.json](/Users/djimra/MSc%20Data%20Science%20Jan%202025/Thesis%20documents/RAG_Pipeline_Project/runs/parity_validation/manual_2026-04-15_grampian_subset10_fix10b/parity_report_fix10b.json:1)

## Final Outcome

- Preprocessing parity restored:
  - cleaned page count matched
  - OCR page count matched
  - chunk count matched
  - chunk-to-page behavior matched, including cross-page page expansion
- Sparse retrieval parity restored:
  - BM25 exact top-10 matched on the checked subset
- Hybrid retrieval parity restored:
  - hybrid exact top-10 matched on the checked subset
- Page-level evaluation parity restored:
  - `Hit@1 = 0.70`
  - `Hit@3 = 0.90`
  - `MRR = 0.80`

## Documented Parity Exception

One dense-only exact-order discrepancy remains on the checked subset.

- Query: `Q_2023_FIN_07`
- Legacy dense pages: `[20, 29, 25, 24, 148, 78, 26]`
- Refactor dense pages: `[25, 29, 20, 24, 26, 79, 80, 22]`

This exception was investigated to vector level.

Confirmed findings:

- candidate chunk texts match between legacy and refactor for the relevant page neighborhood
- embedding text composition matches between legacy and refactor
- stored legacy chunk vectors in `embeddings.npy` match freshly encoded refactor chunk vectors exactly for the candidate chunks
- dense rerank logic matches the legacy evaluator logic used in `scripts/retrieval_eval.py`
- eval-set provenance matches the legacy recorded hash

Interpretation:

- the remaining discrepancy is not explained by a refactor logic bug
- it is most likely due to unrecoverable query-side embedding variance in the archived legacy run for this one query

## Acceptance Decision

The refactor is accepted with this documented parity exception because:

- sparse and hybrid retrieval behavior are reproduced on the benchmark subset
- page-level retrieval metrics are reproduced
- the remaining dense discrepancy is isolated, query-specific, and not attributable to a refactor rule change

This means the refactor can be treated as the maintained research pipeline, while the legacy path remains the archived reference implementation.
