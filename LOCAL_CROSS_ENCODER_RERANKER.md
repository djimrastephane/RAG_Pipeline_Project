# Local Cross-Encoder Reranker

## What was added
A local built-in cross-encoder reranker can now be enabled in both:
- `scripts/retrieval_eval_hybrid.py`
- `scripts/retrieval_eval_splade_hybrid.py`
- runtime retrieval in `src/rag_pdf/services/search_service.py` (env-driven)

The reranker is applied after RRF fusion and before final ranking output.

## How scoring is applied
For top `N` fused candidates:
1. Score `(query, chunk_text)` pairs with a cross-encoder.
2. Min-max normalize scores to `[0,1]` per query.
3. Add to fused score: `score += cross_encoder_weight * normalized_ce_score`.
4. Re-sort candidates.

## Ablation config keys
```yaml
cross_encoder:
  enabled: true
  model: models/bge-reranker-v2-m3
  topn: 50
  weight: 0.2
```

## CLI flags (hybrid evaluators)
- `--enable-cross-encoder-rerank`
- `--cross-encoder-model`
- `--cross-encoder-topn`
- `--cross-encoder-weight`

## Runtime env vars (search service)
- `ENABLE_CROSS_ENCODER_RERANK=1`
- `CROSS_ENCODER_MODEL_NAME=models/bge-reranker-v2-m3`
- `CROSS_ENCODER_TOPN=50`
- `CROSS_ENCODER_WEIGHT=0.2`

## Notes
- Use a local model path for reproducibility/offline runs.
- Start with small `topn` (e.g., 30-50) to control latency.
- Tune `weight` via ablation; larger values increase reranker influence.
