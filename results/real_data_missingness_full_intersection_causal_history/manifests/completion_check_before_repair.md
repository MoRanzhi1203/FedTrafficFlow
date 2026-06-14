# Completion Check Before Repair

- 输出目录：`results/real_data_missingness_full_intersection_causal_history`
- 预期 chunk 数：`61`

## Chunk 级资产

- input_chunks：`61/61`，缺失 chunk：`[]`
- masks：`61/61`，缺失 chunk：`[]`
- missing_datasets：`61/61`，缺失 chunk：`[]`

## 方法覆盖

- `zero_fill`：imputed=`61`，detail=`61`，缺失 imputed chunk=`[]`
- `forward_fill`：imputed=`61`，detail=`61`，缺失 imputed chunk=`[]`
- `historical_linear_extrapolation`：imputed=`61`，detail=`61`，缺失 imputed chunk=`[]`
- `geo_neighbor_fill`：imputed=`61`，detail=`61`，缺失 imputed chunk=`[]`
- `function_curve_fit`：imputed=`0`，detail=`0`，缺失 imputed chunk=`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]`
- `geo_func_hybrid`：imputed=`0`，detail=`0`，缺失 imputed chunk=`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60]`

## 关键文件

- generate_missing_chunk_status.csv：`true`
- impute_chunk_status.csv：`true`
- imputation_runs.csv：`true`
- imputation_quality_detail.csv：`true`
- imputation_quality_summary_all_days.csv：`false`
- imputation_quality_summary_exclude_warmup.csv：`false`
- full_intersection_missingness_validation.json：`false`
- full_intersection_missingness_audit.json：`false`
- figures：`false`

