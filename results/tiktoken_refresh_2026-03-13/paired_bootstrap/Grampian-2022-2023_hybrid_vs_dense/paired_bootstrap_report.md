# Paired Bootstrap Retrieval Comparison

- System A: `/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed_tiktoken_5docs/Grampian-2022-2023/retrieval_results_hybrid.json`
- System B: `/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/data_processed_tiktoken_5docs/Grampian-2022-2023/retrieval_results.json`
- Common queries: `50`
- Bootstrap iterations: `5000`

## Delta = A - B

- Hit@1: observed=-0.080000, 95% CI=[-0.240000, 0.060000]
- Hit@3: observed=-0.120000, 95% CI=[-0.240000, 0.000000]
- MRR@10: observed=-0.082000, 95% CI=[-0.187336, 0.021778]