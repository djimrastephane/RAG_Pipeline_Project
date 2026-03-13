# Ablation Steps Summary

| Step | Test Parameters | Outcome | Statistically significant? |
|---|---|---|---|
| 1. Chunking selection | `chunk_size/overlap` and segment mode (notably `280/90` vs `320/90`, `seg_on/off`) | Final retrieval runs were standardized on `280/90` with `seg_off` for the 5-doc GT benchmark. | Not formally tested |
| 2. Retriever baseline ladder | Dense-only, BM25-only, Hybrid (RRF) | Hybrid selected as production base (better overall balance than single retrievers). | Not formally tested |
| 3. Hybrid fusion CV tuning | `rrf_k`, `dense_weight`, `bm25_weight` (CV) | Stable production setting selected: `rrf_k=20`, `dense_weight=0.5`, `bm25_weight=2.0`. | Selection by CV stability (not a single p-value test) |
| 4. Max search depth sensitivity | `MAX_K_SEARCH = 25, 50, 100, 150, 200` | Flat/near-flat behavior; `25` remained acceptable. In `25 vs 200`: only `11/250` queries changed rank proxy at k=10; paired tests not significant. | Not significant |
| 5. Reranker weight CV (updated logic) | `ENTITY_MATCH_BOOST {0.03,0.04,0.05}`, `NUMERIC_DENSITY_BOOST {0.02,0.03,0.04}`, `MAX_ENTITY_MATCHES {3,4}` | Best: `0.03 / 0.02 / 4` (`em0.03_nd0.02_mx4`). Vs default (`0.04/0.03/4`) on 5 docs: Hit@1 `0.732 vs 0.716` (`+0.016`), FP2 `67 vs 71` (`-4`). | Not formally significance-tested |
| 6. OCR fallback threshold ablation (corrected rerun) | `FALLBACK_MIN_CHARS = 50, 80, 100, 150, 200` | Retrieval metrics were identical across all thresholds on 5-doc GT: Hit@1 `0.748`, MRR@10 `0.8210`, FP2 `63`. Internal fallback markers changed, but end metrics did not. | Identical (no measurable difference) |
| 7. Section/subsection audit (23 docs) | Page-level diagnostic audit + era split | 23-doc totals: 2676 pages. Section unknown low (1.61%), subsection unknown high (84.64%). OCR-heavy in older eras: OCR page rate ~44–49% (older/mid) vs ~3% (recent). | Descriptive audit (not an inferential test) |
| 8. 23-doc vs 5-doc proportion check | Compare diagnostic proportions | 5-doc set is cleaner than 23-doc aggregate on key proxies (e.g., subsection unknown much lower in 5-doc set). | Not formally significance-tested |
| 9. Statistical checks on key deltas | Paired tests + bootstrap (1000) where run | For `MAX_K_SEARCH 25 vs 200`, deltas were small and not statistically significant at 5%. | Not significant |
