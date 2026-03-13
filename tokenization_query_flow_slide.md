% Query Flow Through the Three Tokenization Stages
% End-to-end retrieval view

# Query Flow Through the Three Tokenization Stages

## Document side
1. PDF pages are extracted as text
2. `tiktoken` is used to build overlapping chunks
3. The same chunk text is indexed twice:
   - by MiniLM for dense retrieval
   - by the custom BM25 tokenizer for sparse retrieval

## Query side
1. User asks a question
2. Query is tokenized by the MiniLM tokenizer for embedding
3. Query is also tokenized by the BM25 regex tokenizer for lexical search

## Retrieval stage
- Dense branch: semantic similarity over embeddings
- Sparse branch: lexical scoring over BM25 terms
- Fusion: RRF combines both ranked lists

## Core point
- `tiktoken` is only used to define chunk boundaries
- It does not tokenize the query for retrieval
- Dense and sparse retrieval each tokenize the query in their own way
