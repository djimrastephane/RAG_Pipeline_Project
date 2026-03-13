# ToC Upgrade Diagnostics (5 Docs)

## Aggregate
- docs_total: `5`
- toc_detected_docs: `5`
- toc_detected_rate: `1.0`
- mean_toc_items: `15.6`
- mean_toc_coverage_pct: `0.9918477284530912`
- mean_toc_override_rate: `0.13584363058579438`
- mean_subsection_unknown_before_sections: `0.6471692535577102`
- mean_subsection_unknown_after_sections: `0.3573079848407027`
- mean_subsection_unknown_delta_sections: `-0.28986126871700757`
- total_subsection_reject_count: `116`

## Per-document table

| doc_id             |   pages_total |   sections_before |   sections_after |   subsection_unknown_pct_before_sections |   subsection_unknown_pct_after_sections |   subsection_unknown_pct_delta_sections | toc_detected   |   toc_pages_count |   toc_items_count |   toc_offset |   toc_offset_support_count |   toc_offset_confidence |   toc_coverage_pct |   toc_override_rate |   subsection_reject_count |   subsection_unknown_pct_before_metric |   subsection_unknown_pct_after_metric | has_toc_artifact   | has_rejected_csv   |
|:-------------------|--------------:|------------------:|-----------------:|-----------------------------------------:|----------------------------------------:|----------------------------------------:|:---------------|------------------:|------------------:|-------------:|---------------------------:|------------------------:|-------------------:|--------------------:|--------------------------:|---------------------------------------:|--------------------------------------:|:-------------------|:-------------------|
| Grampian-2020-2021 |           133 |               114 |              132 |                                 0.921053 |                                0.295455 |                             -0.625598   | True           |                 1 |                15 |            2 |                          2 |                     0.7 |           0.984962 |            0        |                         8 |                               0.909774 |                              0.300752 | True               | True               |
| Grampian-2021-2022 |           146 |               128 |              145 |                                 0.929688 |                                0.131034 |                             -0.798653   | True           |                 1 |                16 |            0 |                          2 |                     0.7 |           0.986301 |            0        |                         8 |                               0.917808 |                              0.136986 | True               | True               |
| Grampian-2022-2023 |           154 |               149 |              153 |                                 0.463087 |                                0.470588 |                              0.00750099 | True           |                 1 |                16 |          -61 |                          1 |                     0.6 |           1        |            0.324675 |                        30 |                               0.448052 |                              0.474026 | True               | True               |
| Grampian-2023-2024 |           161 |               151 |              161 |                                 0.490066 |                                0.453416 |                             -0.0366501  | True           |                 1 |                16 |            0 |                          2 |                     0.7 |           0.993789 |            0.180124 |                        31 |                               0.521739 |                              0.453416 | True               | True               |
| Grampian-2024-2025 |           172 |               169 |              172 |                                 0.431953 |                                0.436047 |                              0.00409385 | True           |                 1 |                15 |            0 |                          2 |                     0.7 |           0.994186 |            0.174419 |                        39 |                               0.44186  |                              0.436047 | True               | True               |

## Artifacts
- `data_processed_toc_upgrade_5docs/toc_upgrade_diagnostics/toc_upgrade_before_after_5docs.csv`
- `data_processed_toc_upgrade_5docs/toc_upgrade_diagnostics/toc_upgrade_summary.json`
- `data_processed_toc_upgrade_5docs/toc_upgrade_diagnostics/chart_toc_coverage_by_doc.png`
- `data_processed_toc_upgrade_5docs/toc_upgrade_diagnostics/chart_subsection_unknown_before_after_5docs.png`
- `data_processed_toc_upgrade_5docs/toc_upgrade_diagnostics/chart_subsection_reject_count_5docs.png`
