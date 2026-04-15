# tiktoken Rerun Audit (Five Grampian Documents)

Artifacts rerun under `tiktoken`:

- `data_processed_tiktoken_5docs/Grampian-2020-2021`
- `data_processed_tiktoken_5docs/Grampian-2021-2022`
- `data_processed_tiktoken_5docs/Grampian-2022-2023`
- `data_processed_tiktoken_5docs/Grampian-2023-2024`
- `data_processed_tiktoken_5docs/Grampian-2024-2025`

Comparison outputs:

- `results/tiktoken_vs_fallback_2020_2025.csv`
- `results/tiktoken_vs_fallback_2020_2025.png`
- `results/retrieval_compare_fallback_vs_tiktoken_2020_2025.csv`
- `results/retrieval_compare_fallback_vs_tiktoken_2020_2025.png`

## Main conclusion

The original five-document baseline was materially affected by fallback chunking. Reprocessing with `tiktoken` improved baseline hybrid retrieval on all five documents at `Hit@1`, with gains from `+0.04` to `+0.10`, and improved `MRR@10` on all five documents.

## Definitely stale

- Any baseline retrieval claims or charts drawn directly from `data_processed/Grampian-2020-2021` to `data_processed/Grampian-2024-2025`
- Any chart/table using those five documents as the baseline chunk/index artifacts
- Any result explicitly described as `chunk_size_tokens=280`, `chunk_overlap_tokens=90` when those artifacts were built under `word_fallback`

## High-priority candidates to rerun

- Five-document baseline retrieval summaries
- Any paired comparison where system A or B is the current five-document fallback baseline
- Any figures that compare dense vs hybrid, hybrid defaults, or retrieval stability using these exact five baseline artifacts
- Any chunking-sensitive embedding diagnostics derived from the fallback-built chunk set

## Probably still usable for now

- Method-vs-method comparisons where all arms were built from the same fallback chunk root and the claim is relative rather than absolute
- Generation-only analyses that do not depend on changing retrieval candidates
- Charts from unrelated document sets not tied to these five fallback artifacts

## Minimum defensible next step

1. Replace the five-document baseline with the new `data_processed_tiktoken_5docs` baseline.
2. Regenerate baseline retrieval charts/tables that feed the thesis narrative.
3. Rerun only ablations whose conclusion depends on chunk construction, chunk counts, or direct comparison to the five-document baseline.
