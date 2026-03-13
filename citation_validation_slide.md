% Citation Parse & Validate (LLM Branch)
% RAG Pipeline Clarification

# What "Citation Parse & Validate" Does

**Goal:** accept generated answers only when citations map to retrieved evidence.

## 1) Parse output from the LLM
- Expected format: `{"answer":"...","citations":[{"chunk_id":"...","page":21}]}`
- If strict JSON parsing fails, fallback regex extracts citation-like text patterns.
- Parsed citations are normalized to `(chunk_id, page)` pairs.

## 2) Build allowlist from retrieved top-k results
- From retrieval output, create:
- `allowed_pages_by_chunk = { chunk_id -> set(pages from that chunk) }`
- This allowlist is the only evidence the generated answer is allowed to cite.

## 3) Validate each citation
- Citation is valid only if:
- `chunk_id` exists in retrieved top-k chunks, and
- `page` is an integer present in that chunk's retrieved `pages`.
- Otherwise citation is rejected.

## 4) Gate the final generated answer
- If generation failed, answer text is empty, or no valid citations remain:
- `generation_status = "insufficient_evidence"` and `generated_answer = null`
- If at least one citation validates:
- return `generated_answer` + `generated_citations`

## Why this matters
- Prevents unsupported hallucinated references.
- Ensures every accepted citation is traceable to retrieved evidence.
- Keeps answer quality tied to retrieval grounding.
