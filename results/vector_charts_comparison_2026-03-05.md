# Vector Chart Refresh Comparison (Old vs New)

- Generated: 2026-03-05 14:36:35
- Old archive source: `/Users/djimra/MSc Data Science Jan 2025/Thesis documents/RAG_Pipeline_Project/results/archive_vector_charts_20260305_143548`
- New source: `results/vector_distribution_summary.csv`

## Quick differences
- Grampian-2022-2023: num_vectors 296 -> 306 (delta +10); cos_mean 0.4155 -> 0.4083; pc1_var 0.1512 -> 0.1481
- Grampian-2023-2024: num_vectors 314 -> 323 (delta +9); cos_mean 0.4142 -> 0.4079; pc1_var 0.1473 -> 0.1460
- Grampian-2024-2025: num_vectors 341 -> 341 (delta +0); cos_mean 0.4172 -> 0.4165; pc1_var 0.1417 -> 0.1417

## Notes
- Charts were regenerated from current `data_processed/*/embeddings.npy` artifacts.
- Histogram values can vary slightly run-to-run due to random pair sampling (seed fixed for reproducibility).