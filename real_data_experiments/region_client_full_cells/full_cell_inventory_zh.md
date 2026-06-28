# 新实验 3 / 4：全量有效 grid cells 清单报告

## 1. 目的

本报告用于只读盘点 pooled-grid tensor 中的全部有效 grid cells，并为新实验 3 / 新实验 4 提供 grouped-client 组织所需的基础 inventory。

## 2. 统计摘要

- total grid cells: `630`
- valid grid cells: `223`
- invalid / empty cells: `407`
- inventory csv: `E:/Jupter_Notebook/FedTrafficFlow/real_data_experiments/region_client_full_cells/full_cell_inventory.csv`
- mean source_node_count: `188.305`
- mean_total_flow mean: `458800.042`
- flow_cv mean: `0.069560`
- lag1_autocorr mean: `0.904050`

## 3. 与新编号的关系

- 本 inventory 服务于 `real_data_experiments/region_client_full_cells/` 目录。
- 该目录在新的编号体系下承接新实验 3：多个相似网格合并为一个客户端的对比实验。
- 同时为新实验 4：基于相同客户端组织方式补齐消融实验，提供数据清单与分组基础。
- 本报告只做只读盘点，不运行训练，不修改 results。

## 4. valid cell 定义

- `valid cell` 定义为：active pooled region、`source_node_count > 0`、存在可用时间窗、序列有限且非全零。
- 若坐标列在 sidecar 中缺失，则 CSV 会保留列名但不伪造坐标值。
- 本 inventory 仅用于 grouped-client 组织基础，不改变正式数据入口与 FedAvg 主线。

## 5. 主要无效原因

- `inactive_region;no_source_nodes;all_zero_series`: `407`
