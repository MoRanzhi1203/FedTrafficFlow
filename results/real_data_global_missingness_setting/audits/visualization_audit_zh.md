# 全局缺失值补全结果可视化审计报告

## 1. 数据源

- 主结果 summary: `E:\Jupter_Notebook\FedTrafficFlow\results\real_data_global_missingness_setting\summaries\imputation_quality_summary_exclude_warmup.csv`
- 流量组 summary: `E:\Jupter_Notebook\FedTrafficFlow\results\real_data_global_missingness_setting\summaries\imputation_quality_by_flow_group.csv`
- 是否使用 exclude_warmup 主结果: `True`

## 2. 校验结论

- 是否包含 5%、10%、20%、30%: `True`
- 是否包含 6 个方法: `True`
- 是否排除 0% masked-position error: `True`
- 是否未使用 relative-to-forward-fill 正式主图: `True`
- 是否未重新运行 impute: `True`
- 是否未重新生成 masks: `True`
- 是否未重新生成 missing_datasets: `True`

## 3. 图件输出

- multirate_rmse_by_method: `True`
- multirate_mae_by_method: `True`
- multirate_smape_or_mape_by_method: `True`
- multirate_nrmse_by_method: `True`
- multirate_rmse_by_method_nonzero_zoom: `True`
- multirate_flow_group_rmse_by_method: `True`
- method_rank_heatmap_rmse: `True`
- figure_index_csv: `True`

## 4. 说明

- 当前所有图件均基于 masked-position imputation error，不代表 FedAvg / Independent 交通流预测性能。
- forward_fill 只是 6 个普通方法之一，不作为正式主图 baseline。
- nonzero zoom 图仅用于观察非 zero_fill 方法间差异，不替代完整六方法主图。
