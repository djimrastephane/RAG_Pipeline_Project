# Paired Bootstrap Retrieval Comparison

- System A: `/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/archive/2026-02-28_ablation_cleanup/data_processed_ablation_thesis_5docs_q50/thesis_Grampian-2024-2025_chunk_280_90_seg_off_dense_rerank_on/Grampian-2024-2025/retrieval_results_hybrid.json`
- System B: `/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/archive/2026-02-28_ablation_cleanup/data_processed_ablation_thesis_5docs_q50/thesis_Grampian-2024-2025_chunk_280_90_seg_off_dense_rerank_on/Grampian-2024-2025/retrieval_results.json`
- Common queries: `50`
- Bootstrap iterations: `5000`

## Delta = A - B

- Hit@1: observed=0.000000, 95% CI=[-0.100000, 0.100000]
- Hit@3: observed=-0.020000, 95% CI=[-0.100000, 0.040000]
- MRR@10: observed=-0.012667, 95% CI=[-0.075342, 0.045000]