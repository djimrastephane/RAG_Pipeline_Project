# Paired Bootstrap Retrieval Comparison

- System A: `/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed/Grampian-2022-2023/retrieval_results_hybrid.json`
- System B: `/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed/Grampian-2022-2023/retrieval_results.json`
- Common queries: `50`
- Bootstrap iterations: `5000`

## Delta = A - B

- Hit@1: observed=-0.020000, 95% CI=[-0.160000, 0.120000]
- Hit@3: observed=-0.040000, 95% CI=[-0.160000, 0.080000]
- MRR@10: observed=-0.037397, 95% CI=[-0.137411, 0.061275]