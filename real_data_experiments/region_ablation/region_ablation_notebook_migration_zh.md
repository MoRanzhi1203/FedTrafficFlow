# 新实验 6：全局所有网格按相似度划分为客户端的消融 Notebook 迁移映射

## 编号重构说明

- 本目录在新的编号体系下固定对应新实验 6。
- 旧新映射：原实验 4 -> 新实验 6。
- 文档语义强调全局覆盖式客户端划分上的结构消融。

## 当前迁移状态

- 本 notebook 已完成 tensor-only Python 化迁移。
- 默认输入：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 默认客户端定义：全局覆盖式客户端划分，每个 client 包含一组互不重叠的 grid cells / regions。
- 当前实现保留 `spatial_block` 与 `flow_kmeans`；在新的编号语义中，`flow_kmeans` 对应相似度划分主线。
- 默认联邦聚合为标准样本量加权 `FedAvg`。
- 当前 smoke test 已通过，但 smoke 指标不作为论文正式结果。

## 审计对象

- Notebook：`test/区域客户端消融实验_2x2_最终版.ipynb`
- 已迁移到：`real_data_experiments/region_ablation/`
- 正式 `tensor_path`：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 正式 `regions_path`：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`

## 关键审计结论

- 原 notebook 延续了每个 client = 一组 region indices的定义；在新实验 6 中，该定义被固定为全局覆盖式客户端划分上的消融实验。
- 原 notebook 使用过 `kmeans_cluster + balance_clusters_by_size` 客户端划分思想；新的编号语义下，`flow_kmeans` 对应相似度划分主线。
- 原 notebook 的有效消融变体是四组：`Full`、`w/o Attn`、`w/o CNN`、`w/o LSTM`。
- 正式迁移版继续使用标准样本量加权 `FedAvg`，不引入非主线聚合。

## 默认主流程冻结

- `data_mode = tensor`
- `client = global partition clients`
- `partition_method = flow_kmeans` 作为相似度划分主线语义，`spatial_block` 作为保留的全局划分实现
- `split = temporal_contiguous_by_target_time`
- `FedAvg = standard sample-size weighted FedAvg`
- `variants = Full / Without Attention / Without CNN / Without LSTM`
