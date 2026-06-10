# Real Data Missingness Experiment Audit

## 1. Environment

- Python path: `E:\anaconda3\envs\analysis\python.exe`
- Python version target: `3.9.23`
- Compatibility note: script avoids Python 3.10+ union syntax and avoids optional Markdown table dependencies.

## 2. Inputs

- Input directory: `data/analysis/node_intersection_flow_parquet`
- Output directory: `results/real_data_missingness_experiments`
- Selected file count: `10`
- Max files: `10`
- Max rows per file: `500`
- Mechanisms: `mcar_point`
- Missing rates: `0.0, 0.05, 0.1, 0.2, 0.3`
- Seeds: `42, 2024, 3407, 1234, 5678`
- Impute methods: `zero_fill, forward_fill, linear_interpolation`

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
mcar_point          0.20                 0.20
mcar_point          0.30                 0.30
```

## 5. Imputation Quality Summary

```text
 mechanism        impute_method  missing_rate  imputation_mae  imputation_rmse  imputation_mape
mcar_point         forward_fill          0.00        0.000000         0.000000         0.000000
mcar_point         forward_fill          0.05      882.746620      1227.832632        47.697430
mcar_point         forward_fill          0.10      883.622199      1209.305263        60.040351
mcar_point         forward_fill          0.20      881.760766      1242.716913        54.098155
mcar_point         forward_fill          0.30      884.408533      1240.801332        50.756758
mcar_point linear_interpolation          0.00        0.000000         0.000000         0.000000
mcar_point linear_interpolation          0.05     1141.338044      1567.627303        78.482213
mcar_point linear_interpolation          0.10     1075.567729      1485.238468        84.466185
mcar_point linear_interpolation          0.20     1101.582851      1535.692964        81.765212
mcar_point linear_interpolation          0.30     1102.056858      1532.994024        75.564437
mcar_point            zero_fill          0.00        0.000000         0.000000         0.000000
mcar_point            zero_fill          0.05     2161.403979      2437.730228       100.000000
mcar_point            zero_fill          0.10     2122.336934      2401.430835       100.000000
mcar_point            zero_fill          0.20     2164.531183      2447.672326       100.000000
mcar_point            zero_fill          0.30     2163.081369      2448.517198       100.000000
```

## 6. Output Files

- Design summary rows: `250`
- Mask summary rows: `250`
- Quality summary rows: `750`
- Figure: `figures/missing_rate_vs_imputation_rmse.png`
- Figure: `figures/missing_rate_vs_imputation_rmse.pdf`

