# Retrieval Stability Interpretation (Hybrid Pipeline)

## Configuration Scope
- Documents: `Grampian-2022-2023`, `Grampian-2023-2024`, `Grampian-2024-2025`
- Runs compared:
  - `hybrid_rerankoff_subon`
  - `hybrid_rerankon_subon`
  - `hybrid_maxk50`
  - `hybrid_maxk40`
  - `hybrid_maxk30`
- Queries complete across all runs: `55`

## Key Stability Findings
- **Top-1 chunk flip rate = 0.364**  
  Around 36.4% of queries changed their top-1 chunk across tested settings.
- **Top-1 page flip rate = 0.345**  
  Around 34.5% of queries changed their top-1 page.
- **Mean pairwise top-10 rank correlation = 0.750**  
  Ranking is moderately stable overall, but not invariant.
- **Hit@1 mean ± std = 0.618 ± 0.095**  
  Top-rank success is sensitive to small retrieval-setting changes.
- **MRR@10 mean ± std = 0.719 ± 0.061**  
  Mid-rank quality is more stable than top-rank choice.

## FP2 Persistence
- FP2 in at least one run: `24` queries
- FP2 in at least two runs: `12` queries
- FP2 in at least three runs: `12` queries
- FP2 in all compared runs: `7` queries

Interpretation:
- A non-trivial subset of errors is **persistent** across ablations, indicating structural ranking ambiguity rather than random noise.
- These persistent FP2s are priority candidates for targeted mitigation (e.g., entity/number proximity gates and local tie-break policies), rather than global parameter tuning.

## Thesis-Ready Conclusion
Deterministic preprocessing and fixed retrieval settings improve reproducibility, but they do **not fully eliminate ranking instability**.  
The evidence indicates that remaining instability is concentrated in a smaller set of persistent, structurally ambiguous queries (FP2), where adjacent or semantically similar evidence competes at top rank.
