# Paired Bootstrap Retrieval Comparison

- System A: `/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed_tiktoken_5docs/Grampian-2023-2024/retrieval_results_hybrid.json`
- System B: `/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed_tiktoken_5docs/Grampian-2023-2024/retrieval_results.json`
- Common queries: `50`
- Bootstrap iterations: `5000`

## Delta = A - B

- Hit@1: observed=0.020000, 95% CI=[-0.100000, 0.140000]
- Hit@3: observed=0.100000, 95% CI=[0.000000, 0.200000]
- MRR@10: observed=0.054333, 95% CI=[-0.028333, 0.143289]