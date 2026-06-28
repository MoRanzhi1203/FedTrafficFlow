# 新实验 5：全局所有网格按相似度划分为客户端的 Notebook 迁移映射

## 编号重构说明

- 本目录在新的编号体系下固定对应新实验 5。
- 旧新映射：原实验 3 -> 新实验 5。
- 文档语义强调全局覆盖式客户端划分，而不是 grouped-client 的局部合并语义。

## 当前迁移状态

- 本 notebook 已完成 tensor-only Python 化迁移。
- 默认输入：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 默认客户端定义：全局所有网格按规则划分为多个 client，每个网格只能属于一个 client。
- 当前实现保留 `spatial_block` 与 `flow_kmeans`；在新的编号语义中，`flow_kmeans` 对应相似度划分主线。
- 默认联邦聚合为标准样本量加权 `FedAvg`。
- 当前 smoke test 已通过，但 smoke 指标不作为论文正式结果。

## 审计对象

- Notebook：`test/区域客户端计算_3x2_最终版.ipynb`
- 已迁移到：`real_data_experiments/region_client/`
- 正式 `tensor_path`：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 正式 `regions_path`：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`

## 关键审计结论

- 原 notebook 的客户端定义是每个 client = 一组 region indices，这一点在新实验 5 中解释为全局覆盖式客户端划分。
- 原 notebook 的默认划分方法包含基于区域时序特征的聚类与平衡；新的编号语义下，`flow_kmeans` 对应相似度划分主线。
- 原 notebook 的默认训练主流程混入了 `FedProx`、`server damping`、`personalization` 和 mixed raw-scale loss，这些不进入当前正式默认主流程。
- 原 notebook 的 `split_indices()` 先构造完整样本再随机打乱切分，不满足当前真实数据阶段必须按 target time 连续划分的要求。
- `Independent baseline` 仍保留，但要与 tensor-only / 时间切分一致。

## 默认主流程冻结

- `data_mode = tensor`
- `client = global partition clients`
- `partition_method = flow_kmeans` 作为相似度划分主线语义，`spatial_block` 作为保留的全局划分实现
- `split = temporal_contiguous_by_target_time`
- `FedAvg = standard sample-size weighted FedAvg`
- `Independent baseline = enabled`
