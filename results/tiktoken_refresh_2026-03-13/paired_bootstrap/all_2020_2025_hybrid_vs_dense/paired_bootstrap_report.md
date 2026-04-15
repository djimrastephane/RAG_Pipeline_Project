# Paired Bootstrap Retrieval Comparison

- System A: `/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/results/tiktoken_refresh_2026-03-13/paired_bootstrap/all_2020_2025_hybrid_vs_dense/retrieval_results_hybrid_all.json`
- System B: `/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/results/tiktoken_refresh_2026-03-13/paired_bootstrap/all_2020_2025_hybrid_vs_dense/retrieval_results_dense_all.json`
- Common queries: `250`
- Bootstrap iterations: `5000`

## Delta = A - B

- Hit@1: observed=0.000000, 95% CI=[-0.056000, 0.056000]
- Hit@3: observed=-0.016000, 95% CI=[-0.064000, 0.032000]
- MRR@10: observed=-0.009251, 95% CI=[-0.046505, 0.029009]