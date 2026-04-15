# Table Chunking Ablation

- sample_csv: `results/table_heavy_sample_queries_43.csv`
- docs: `Grampian-2020-2021, Grampian-2021-2022, Grampian-2022-2023, Grampian-2023-2024, Grampian-2024-2025`
- k: `3`
- fixed retrieval: current hybrid default (SearchService)
- fixed constrained prompt/model settings; only table chunking changes
- arms: `baseline, row_preserving, two_stage`
- warnings: `0`