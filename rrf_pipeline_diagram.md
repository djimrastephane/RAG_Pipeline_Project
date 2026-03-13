# RRF In Our Pipeline (Diagram + Example)

```mermaid
flowchart LR
    Q[User Query] --> D[Dense Retrieval<br/>Top-L]
    Q --> B[Lexical Retrieval BM25<br/>Top-L]

    D --> DR[Dense Rank List<br/>r_dense(doc)]
    B --> BR[BM25 Rank List<br/>r_bm25(doc)]

    DR --> F[RRF Fusion]
    BR --> F

    F --> CE{Cross-Encoder<br/>enabled?}
    CE -- Yes --> R[Re-rank top-N]
    CE -- No --> O[Take fused ranking]
    R --> K[Final Top-k_out]
    O --> K
```

## Formula used in this pipeline

\[
\text{RRFScore}(d)=\frac{w_{dense}}{k+r_{dense}(d)}+\frac{w_{bm25}}{k+r_{bm25}(d)}
\]

Pipeline defaults:
- \(k=20\)
- \(w_{dense}=0.5\)
- \(w_{bm25}=2.0\)

Fusion depth before final top-k:
- \(L=\min(\max(100, 20 \cdot k_{out}), \text{corpus size})\)

## Mini numeric example (why rank #40 can still win)

Assume \(k=20\), \(w_{dense}=0.5\), \(w_{bm25}=2.0\):

- Chunk A: dense #40, bm25 #2  
  \(0.5/(20+40) + 2.0/(20+2)=0.0083+0.0909=0.0992\)

- Chunk B: dense #5, bm25 missing  
  \(0.5/(20+5)=0.0200\)

Result: Chunk A can outrank Chunk B after fusion because BM25 contribution is strong.

## Slide caption (short)

`RRF fuses deeper candidate lists (Top-L per branch), then returns only Top-k_out for answering.`
