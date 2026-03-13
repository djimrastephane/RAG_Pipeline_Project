% Citation Validation (Compact)
% Use Beside Pipeline Flowchart

# Citation Parse & Validate (LLM Branch)

`LLM output -> parse answer+citations -> validate vs retrieved top-k -> accept or gate`

## 1. Parse
- Expect JSON: `answer` + `citations[{chunk_id,page}]`
- Fallback: extract citation patterns from text

## 2. Validate
- Build allowlist from retrieved results:
- `(chunk_id, page)` pairs only from top-k evidence
- Keep citation only if pair exists in allowlist

## 3. Decision
- `>=1 valid citation` -> return generated answer + validated citations
- `0 valid citations` (or generation error/empty) -> `insufficient_evidence`

## Why
- Blocks unsupported references
- Keeps generated output grounded in retrieved evidence
