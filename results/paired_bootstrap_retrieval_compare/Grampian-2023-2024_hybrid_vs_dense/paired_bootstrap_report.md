# Paired Bootstrap Retrieval Comparison

- System A: `/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed/Grampian-2023-2024/retrieval_results_hybrid.json`
- System B: `/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed/Grampian-2023-2024/retrieval_results.json`
- Common queries: `50`
- Bootstrap iterations: `5000`

## Delta = A - B

- Hit@1: observed=0.000000, 95% CI=[-0.140000, 0.140000]
- Hit@3: observed=0.040000, 95% CI=[-0.040000, 0.120000]
- MRR@10: observed=0.024476, 95% CI=[-0.060536, 0.112810]