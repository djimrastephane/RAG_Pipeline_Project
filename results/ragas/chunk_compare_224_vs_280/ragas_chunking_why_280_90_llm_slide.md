% Why 280/90 Performs Better With LLM On
% RAGAS Interpretation Slide

# Why `280/90` Performs Better With `LLM on`

## Main message
- The generator only sees the **top-5 retrieved chunks**.
- With `224/56`, evidence is more often **split across chunk boundaries**.
- With `280/90`, each retrieved item is more likely to contain a **complete answer-bearing span**.
- That makes grounded answer synthesis easier, even when retrieval quality is similar.

## Evidence from RAGAS
- `LLM off`: `224/56` slightly improves answer-level metrics, but retrieval metrics fall.
- `LLM on`: `280/90` is better on **all four** RAGAS means.
- `LLM on` lift over `LLM off` is much larger for `280/90`:
- `280/90`: Answer Relevancy `+0.225`, Faithfulness `+0.295`
- `224/56`: Answer Relevancy `+0.058`, Faithfulness `+0.116`

## Concrete example: `Q_2021_FIN_P1`
Question:
`Did NHS Grampian overspend or underspend its Core Revenue Resource Limit in 2020/21, and by how much?`

`280/90`
- One retrieved chunk contains both the table values and the explicit statement:
- `Core Revenue Resource Limit 1,278,771 1,278,002 769 ... The Board is reporting an underspend of £0.769m against a target of breakeven on the revenue resource limit for 2020/21.`
- Generated answer succeeds:
- `NHS Grampian underspent its Core Revenue Resource Limit by £0.769m in 2020/21.`

`224/56`
- The key statement is split across two chunks:
- chunk 1 ends with:
- `... Underlying Surplus against Core Revenue Resource Limit 769 0% The Board is reporting an underspend of £0.769m against`
- chunk 2 continues separately:
- `£000 Core Revenue Resource Variance Surplus in 2020/21 769 ... The Board is reporting an underspend of £0.769m against a target of breakeven on the revenue resource limit for 2020/21.`
- Generated answer fails.

## Takeaway
- Smaller chunks can help fine-grained extraction.
- But with a fixed top-5 generation window, **chunk coherence matters more than granularity**.
- In this setup, `280/90` gives the LLM better packaged evidence than `224/56`.

## Speaker notes
- The key point is that this is not only a retrieval-ranking issue; it is also a retrieval-packaging issue.
- The RAGAS export passes `k = 5` contexts per query, and the generation prompt also caps context at 5 chunks.
- That means smaller chunks do not get extra context budget to compensate for evidence fragmentation.
- The `Q_2021_FIN_P1` example makes this concrete: `280/90` keeps the decisive evidence in one chunk, while `224/56` splits it across adjacent chunks.
- This explains why `280/90` benefits more when generation is enabled, especially on Answer Relevancy and Faithfulness.
