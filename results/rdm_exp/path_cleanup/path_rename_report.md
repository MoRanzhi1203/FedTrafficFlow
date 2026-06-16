# 路径缩短报告

## 结论

- 本次处理仅执行路径缩短、路径默认值同步、文档路径引用同步。
- 原最长路径: `results\real_data_missingness_experiments\scenarios\node_subset_temporal_outage_mixed_short_mid_long\imputation\imputed_datasets\mechanism_node_subset_temporal_outage__rate_0p05__mixed_short_mid_long__seed_42__method_historical_linear_extrapolation\node_flow_chunk_000.parquet` (310)
- 新最长路径: `data\analysis\date_type_curve_method_comparison\function_cluster_visualization\M2_shape_normalized_weighted_curve_normalized_residual_distribution_by_cluster.png` (195)
- 缩短比例: `37.10%`
- 重命名条目数: `394`

## 新短路径

- `g_mcar_pt`: `results\rdm_exp\scenarios\g_mcar_pt`
- `ntb_mix`: `results\rdm_exp\scenarios\ntb_mix`
- `nso_mix`: `results\rdm_exp\scenarios\nso_mix`

## 方法缩写

- `zero_fill -> zf`
- `forward_fill -> ff`
- `historical_linear_extrapolation -> hle`
- `road_topology_neighbor_fill -> rtn`
- `function_curve_fit -> fcf`
- `topology_function_hybrid -> tfh`

## 逻辑变更审计

- 是否修改算法逻辑: 否
- 是否修改缺失生成逻辑: 否
- 是否修改补全逻辑: 否
- 是否修改指标计算逻辑: 否
- 是否修改可视化逻辑: 否
- 是否验证通过: 是
