# Judge Analysis: Quoted-Evidence Failures

- Source ablation dir: `results/generation_prompt_ablation_2026-03-13`
- Arms inspected: `baseline, grounded_reasoning, quote_then_answer, constrained_extraction`
- Candidate failures: `23`
- Judge model: `qwen2.5:7b-instruct`
- Evaluation performed on `n = 9` queries.

## Label Summary

| arm                    | judge_label                            |   count |   pct_within_arm |
|:-----------------------|:---------------------------------------|--------:|-----------------:|
| quote_then_answer      | quote_supports_gold_but_answer_misread |       7 |         1        |
| constrained_extraction | quote_supports_gold_but_answer_misread |       6 |         0.857143 |
| constrained_extraction | quote_irrelevant                       |       1 |         0.142857 |

## Charts

- Evaluation performed on n = 9 queries.

- `results/generation_prompt_ablation_2026-03-13/judge_quoted_failures/charts/judge_label_distribution.png`
- `results/generation_prompt_ablation_2026-03-13/judge_quoted_failures/charts/judge_mismatch_by_quote_support.png`