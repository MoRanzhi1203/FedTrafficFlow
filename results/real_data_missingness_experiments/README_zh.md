# 真实数据缺失实验统一结果目录

1. 本目录统一管理三类真实数据缺失与补全实验。
2. global_mcar_point 是完整数据全局 MCAR 点级随机缺失。
3. node_temporal_block_mixed_short_mid_long 是单节点连续时间块缺失，长度为 short/mid/long 混合。
4. node_subset_temporal_outage_mixed_short_mid_long 是节点子集连续离线缺失，长度为 short/mid/long 混合。
5. 每个 scenario 下分为 missingness_setting 和 imputation。
6. missingness_setting 存放 masks、missing_datasets、缺失设置 audit 和 manifest。
7. imputation 存放 imputed_datasets、summary、figures、补全 audit 和 manifest。
8. comparison 存放三类机制综合对比图。
9. parquet 大文件不进入 Git。
10. 当前结果是缺失值补全误差，不是交通流预测误差。

## Scenario IDs

- `global_mcar_point`
- `node_temporal_block_mixed_short_mid_long`
- `node_subset_temporal_outage_mixed_short_mid_long`

## 路径索引

- `experiment_registry.json`: `results\real_data_missingness_experiments\experiment_registry.json`
- `path_aliases.json`: `results\real_data_missingness_experiments\path_aliases.json`
- `comparison`: `results\real_data_missingness_experiments\comparison`
