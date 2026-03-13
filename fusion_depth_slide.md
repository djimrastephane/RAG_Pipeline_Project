# Why Fusion Depth Is Larger Than Final Top-k

## Core Idea
- `k_out` = final number of chunks returned to downstream steps (for example `k_out = 3`).
- RRF needs a wider candidate pool from both branches before selecting final top-k.
- We use:

`L = min(max(100, 20 * k_out), corpus_size)`

where `L` is the fusion depth per branch.

## Practical Example (`k_out = 3`)
- Compute fusion depth:
  `L = min(max(100, 20*3), corpus_size) = min(100, corpus_size)`
- If corpus has >100 chunks:
  - Dense branch sends top-100 ranks to RRF.
  - BM25 branch sends top-100 ranks to RRF.
  - RRF fuses both ranked lists.
  - Final output keeps only top-3 (`k_out=3`).

## Why This Helps
- A chunk ranked low in dense (for example `#40`) but high in BM25 (for example `#2`) can still be promoted by RRF.
- If both branches only sent top-3, many useful cross-branch candidates would never be considered.

## Flow Label Suggestion
`Query -> Dense Top-L`

`Query -> BM25 Top-L`

`Dense Top-L + BM25 Top-L -> RRF -> Top-k_out`
