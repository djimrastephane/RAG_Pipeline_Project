% Trim Audit Interpretation
% How to Read the Chart

# Trim Audit (250 Queries) — Interpretation

## What the two panels show
- **Left (histogram):** distribution of trim severity per trimmed chunk.  
  Value plotted = `original_chars - 2200`.
- **Right (boxplots):** the same trim severity split by document year.

## Key findings from this chart
- Total trimmed chunks: **28**
- All trimmed chunks are **table chunks** (`table_chunks = 28`)
- Trim severity is mixed:
  - light trims (~`300` to `2,000` chars removed)
  - heavy trims (~`8,000` to `14,000` chars removed)
- Highest trim severities appear in **2023–2024** and **2024–2025**.

## Practical meaning for the pipeline
- Trimming pressure is concentrated in **very long retrieved table chunks**, not normal narrative chunks.
- The 2200-char cap can cut a large tail for some table chunks.
- In the 250-query audit, this did **not** produce observed loss of expected-answer strings from retrieved context.

## One-line takeaway
`Trimming is real and sometimes large, but in this audit it was localized to table chunks and did not show measurable answer-string loss.`
