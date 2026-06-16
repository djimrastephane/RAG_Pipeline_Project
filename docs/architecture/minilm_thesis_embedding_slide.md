% From RAG Chunks to FAISS Vectors
% How MiniLM fits the NHS Grampian RAG pipeline

# From RAG Chunks to FAISS Vectors

## 1. Chunk entering the encoder
- Retrieval setting: `chunk_size=224`, `overlap=56`, `segment_aware=false`
- The same chunk is indexed for both branches:
  - dense with `all-MiniLM-L6-v2`
  - sparse with the BM25 regex tokenizer
- Real corpus example: `Grampian-2024-2025:p0028_001` on page `28`
- That chunk contains the statutory financial targets table, including the core revenue resource limit and outturn

## 2. What MiniLM actually stores
- `tiktoken` created the chunk boundary, but MiniLM tokenizes the text again for embedding
- Each input token becomes a contextualised `384`-value representation
- Mean pooling collapses all token states into one `1 x 384` chunk vector
- That final vector is the row stored in FAISS for dense retrieval

## 3. Why this matters
- Query example: `Q_2025_FIN_02`
- Question: `How much did NHS Grampian actually spend against its core resource budget in 2024/25?`
- Expected evidence page: `28`
- Hybrid retrieval uses `RRF(k=20)` with `dense_weight=0.5` and `bm25_weight=2.0`
- Reranking is enabled, with boosts for table chunks and other evidence cues
- In the recorded example, hybrid retrieves page `28` at top-1 via `Grampian-2024-2025:p0028_001`

## Reader takeaway
- `tiktoken` decides where chunks begin and end
- MiniLM turns each RAG chunk into a searchable semantic vector
- BM25 plus reranking help the same evidence win when the question uses exact finance wording
- In this setup, the sliding-window stride is `168` tokens (`224 - 56`)
