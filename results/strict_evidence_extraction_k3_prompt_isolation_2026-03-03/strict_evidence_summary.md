# Strict Evidence-Constrained Extraction

- sample_csv: `results/context_chunks_3_vs_5_stats_2026-03-02/sampled_50_queries.csv`
- n_queries: `50`
- answer_accuracy: `0.4`
- quote_support_rate: `0.4`
- strict_accuracy: `0.4`
- json_parse_ok_rate: `0.96`
- evidence_verbatim_rate: `0.54`

## Failure Modes

| failure_mode | count | pct |
| --- | --- | --- |
| did_not_identify_gold | 30 | 0.6 |
| strict_correct | 20 | 0.4 |

## By Difficulty

| difficulty | n | answer_accuracy | quote_support_rate | strict_accuracy | retrieval_hit_at_k |
| --- | --- | --- | --- | --- | --- |
| LEX | 25 | 0.48 | 0.48 | 0.48 | 0.8 |
| MOD | 11 | 0.636364 | 0.636364 | 0.636364 | 0.909091 |
| STR | 14 | 0.0714286 | 0.0714286 | 0.0714286 | 0.785714 |