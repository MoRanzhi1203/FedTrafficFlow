# 全量有效 grid cells 清单报告

## 1. 目的

本报告用于只读盘点 pooled-grid tensor 中的全部 grid cells，并识别可用于 full-cells region-client 实验的有效 cells。

## 2. 统计摘要

- total grid cells: `630`
- valid grid cells: `223`
- invalid / empty cells: `407`
- inventory csv: `E:\Jupter_Notebook\FedTrafficFlow\real_data_experiments\region_client_full_cells\full_cell_inventory.csv`
- mean source_node_count: `188.305`
- mean_total_flow mean: `458800.042`
- flow_cv mean: `0.069560`
- lag1_autocorr mean: `0.904050`

## 3. 说明

- `valid cell` 定义为：active pooled region、`source_node_count > 0`、存在可用时间窗、序列有限且非全零。
- 若坐标列在 sidecar 中缺失，则 CSV 会保留列名但不伪造坐标值。

## 4. 主要无效原因

- `inactive_region;no_source_nodes;all_zero_series`: `407`
