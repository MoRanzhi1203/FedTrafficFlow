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
- `function_curve_fit`：imputed=`61`，detail=`61`，缺失 imputed chunk=`[]`
- `geo_func_hybrid`：imputed=`61`，detail=`61`，缺失 imputed chunk=`[]`

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

