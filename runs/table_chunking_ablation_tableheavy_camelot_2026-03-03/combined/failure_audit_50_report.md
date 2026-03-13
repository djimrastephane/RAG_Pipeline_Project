# Failure Audit (50 strict failures)

- Sample size: **50** (random_state=42)

- model_extraction_error: **32** (64.0%)
- context_missing: **9** (18.0%)
- ambiguous_question: **9** (18.0%)

## Interpretation
- `gold_mismatch` is not dominant in this audit -> bottleneck appears more in extraction/reasoning than gold-label mismatch.