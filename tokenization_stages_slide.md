% Three Tokenization Stages in the Pipeline
% Why the pipeline intentionally uses different tokenizers

# Three Tokenization Stages in the Pipeline

## 1. Document chunking
- Tokenizer: `tiktoken (cl100k_base)`
- Purpose: controls chunk size and overlap during preprocessing
- Current setting: `224` tokens with `56` token overlap
- Output: stable text chunks before indexing

## 2. Dense retrieval
- Tokenizer: `MiniLM tokenizer`
- Purpose: converts chunk text and query text into transformer input IDs
- Used by: `all-MiniLM-L6-v2`
- Output: embedding vectors for semantic similarity search

## 3. Sparse retrieval
- Tokenizer: `custom BM25 regex tokenizer`
- Purpose: builds lexical term lists for exact term matching
- Rule used in code:
  `re.findall(r"[a-z0-9][a-z0-9\\-]{1,}", text.lower())`

## BM25 tokenizer details
- Lowercases all text
- Keeps alphanumeric and hyphenated terms
- Drops punctuation
- Drops single-character tokens
- No stemming, lemmatization, or subword splitting

## Key message
- Different tokenizers here are expected and correct
- `tiktoken` defines chunk boundaries
- MiniLM defines embedding input
- BM25 defines lexical matching terms
