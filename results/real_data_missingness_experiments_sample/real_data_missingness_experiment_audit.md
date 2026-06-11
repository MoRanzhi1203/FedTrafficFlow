# Real Data Missingness Experiment Audit

## 1. Environment

- Python path: `E:\anaconda3\envs\analysis\python.exe`
- Python version target: `3.9.23`
- Compatibility note: script avoids Python 3.10+ union syntax and avoids optional Markdown table dependencies.

## 2. Inputs

- Input directory: `data/analysis/node_intersection_flow_parquet`
- Output directory: `results/real_data_missingness_experiments_sample`
- Selected file count: `1`
- Max files: `1`
- Max rows per file: `500`
- Mechanisms: `mcar_point`
- Missing rates: `0.0, 0.05, 0.1`
- Seeds: `42`
- Impute methods: `zero_fill, forward_fill, linear_interpolation, geo_neighbor_fill, function_curve_fit, geo_func_hybrid`

## 3. Detected Columns

- target_col: `路口车流量`
- time_col: `时间段`
- node_col: `节点ID`

## 4. Actual Missing Rate Summary

```text
 mechanism  missing_rate  actual_missing_rate
mcar_point          0.00                 0.00
mcar_point          0.05                 0.05
mcar_point          0.10                 0.10
```

## 5. Imputation Quality Summary

```text
 mechanism        impute_method  missing_rate  imputation_mae  imputation_rmse  imputation_mape
mcar_point         forward_fill          0.00        0.000000         0.000000         0.000000
mcar_point         forward_fill          0.05      799.449519      1126.597747        31.380291
mcar_point         forward_fill          0.10      955.201882      1265.743961        31.861695
mcar_point   function_curve_fit          0.00        0.000000         0.000000         0.000000
mcar_point   function_curve_fit          0.05     1165.581061      1615.838263        63.213232
mcar_point   function_curve_fit          0.10     1277.587383      1677.803249        57.783156
mcar_point      geo_func_hybrid          0.00        0.000000         0.000000         0.000000
mcar_point      geo_func_hybrid          0.05      861.454103      1127.960839        41.298972
mcar_point      geo_func_hybrid          0.10     1052.947504      1293.218923        42.023939
mcar_point    geo_neighbor_fill          0.00        0.000000         0.000000         0.000000
mcar_point    geo_neighbor_fill          0.05      799.449519      1126.597747        31.380291
mcar_point    geo_neighbor_fill          0.10      955.201882      1265.743961        31.861695
mcar_point linear_interpolation          0.00        0.000000         0.000000         0.000000
mcar_point linear_interpolation          0.05     1165.581061      1615.838263        63.213232
mcar_point linear_interpolation          0.10     1277.587383      1677.803249        57.783156
mcar_point            zero_fill          0.00        0.000000         0.000000         0.000000
mcar_point            zero_fill          0.05     2089.480769      2345.923033       100.000000
mcar_point            zero_fill          0.10     2422.373091      2630.061666       100.000000
```

## 6. Output Files

- Design summary rows: `3`
- Mask summary rows: `3`
- Quality summary rows: `18`
- Figure: `figures/missing_rate_vs_imputation_rmse.png`
- Figure: `figures/missing_rate_vs_imputation_rmse.pdf`

