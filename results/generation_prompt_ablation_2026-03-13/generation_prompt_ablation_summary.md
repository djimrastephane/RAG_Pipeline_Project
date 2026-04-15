# Generation Prompt Ablation

- Design: fixed retrieval, grounded-answer prompt variants only.
- docs: `5`
- queries_per_doc: `3`
- k: `5`
- arms: `baseline, grounded_reasoning, quote_then_answer, constrained_extraction`
- total evaluations: `60`
- Evaluation performed on `n = 15` queries.

## Recommendation

- Current top arm by ranking rule: `baseline` (answer_accuracy=0.400, citation_valid_rate=0.667, latency_p50_ms=5216.0).

## Ranked arms

| arm                    |   n_queries |   answer_accuracy |   generation_ok_rate |   insufficient_evidence_rate |   context_truncated_rate |   citation_valid_rate |   json_parse_success_rate |   latency_mean_ms |   latency_p50_ms |   latency_p95_ms |   prompt_chars_mean |
|:-----------------------|------------:|------------------:|---------------------:|-----------------------------:|-------------------------:|----------------------:|--------------------------:|------------------:|-----------------:|-----------------:|--------------------:|
| baseline               |          15 |          0.4      |             0.666667 |                     0.333333 |                        0 |              0.666667 |                         1 |           5122.89 |          5216.04 |          6547.61 |             6860.73 |
| grounded_reasoning     |          15 |          0.333333 |             0.666667 |                     0.333333 |                        0 |              0.666667 |                         1 |           5551.7  |          5160.38 |          7366.66 |             7162.73 |
| quote_then_answer      |          15 |          0.333333 |             0.8      |                     0.2      |                        0 |              0.8      |                         1 |           7044.09 |          7373.6  |          8825.53 |             7118.73 |
| constrained_extraction |          15 |          0.266667 |             0.733333 |                     0.133333 |                        0 |              0        |                         1 |           7302.93 |          6951.18 |         10245.6  |                0    |

## Generation Status By Arm

| arm                    | generation_status     |   count |      pct |
|:-----------------------|:----------------------|--------:|---------:|
| baseline               | ok                    |      10 | 0.666667 |
| baseline               | insufficient_evidence |       5 | 0.333333 |
| baseline               | error                 |       0 | 0        |
| grounded_reasoning     | ok                    |      10 | 0.666667 |
| grounded_reasoning     | insufficient_evidence |       5 | 0.333333 |
| grounded_reasoning     | error                 |       0 | 0        |
| quote_then_answer      | ok                    |      12 | 0.8      |
| quote_then_answer      | insufficient_evidence |       3 | 0.2      |
| quote_then_answer      | error                 |       0 | 0        |
| constrained_extraction | ok                    |      11 | 0.733333 |
| constrained_extraction | error                 |       2 | 0.133333 |
| constrained_extraction | insufficient_evidence |       2 | 0.133333 |

## Difficulty Breakdown

| arm                    | difficulty   |   n |   answer_accuracy |   ok_rate |
|:-----------------------|:-------------|----:|------------------:|----------:|
| baseline               | LEX          |   5 |               0.6 |       0.8 |
| baseline               | MOD          |   5 |               0.4 |       0.8 |
| baseline               | STR          |   5 |               0.2 |       0.4 |
| grounded_reasoning     | LEX          |   5 |               0.4 |       1   |
| grounded_reasoning     | MOD          |   5 |               0.4 |       0.8 |
| grounded_reasoning     | STR          |   5 |               0.2 |       0.2 |
| quote_then_answer      | LEX          |   5 |               0.4 |       1   |
| quote_then_answer      | MOD          |   5 |               0.4 |       0.8 |
| quote_then_answer      | STR          |   5 |               0.2 |       0.6 |
| constrained_extraction | LEX          |   5 |               0.6 |       1   |
| constrained_extraction | MOD          |   5 |               0.2 |       0.8 |
| constrained_extraction | STR          |   5 |               0   |       0.4 |

## Charts

- Evaluation performed on n = 15 queries.

- `results/generation_prompt_ablation_2026-03-13/charts/prompt_ablation_quality_metrics.png`
- `results/generation_prompt_ablation_2026-03-13/charts/prompt_ablation_accuracy_vs_latency.png`
- `results/generation_prompt_ablation_2026-03-13/charts/prompt_ablation_status_mix.png`