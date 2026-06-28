# 真实数据实验 3/5/6 Smoke 状态报告

> 生成日期：2026-06-29
> 这些是 r1e1 smoke 测试，只验证 pipeline 可运行，不作为论文 formal 结果。
> results 和 smoke_logs 目录不提交。

## 1. 运行环境

- **分支**: `feature/real-exp1-client-similarity-diagnosis`
- **Commit**: `5653445 docs(real-data): record exp1 formal result status`
- **GPU**: NVIDIA GeForce RTX 3060 Laptop GPU
- **Python**: `E:\anaconda3\envs\FedTrafficFlow\python.exe`

## 2. 实验 3 Smoke（多个相似 grid cells 组成一个 client）

- **入口**: `real_data_experiments.region_client_full_cells.rfc_core`
- **命令**:
  ```
  python -m real_data_experiments.region_client_full_cells.rfc_core \
    --tensor-path data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt \
    --partition-file real_data_experiments/region_client_full_cells/partitions/similarity_k5.json \
    --rounds 1 --local-epochs 1 --device cuda \
    --output-dir results/real_data_experiments/smoke/exp3_rfc_similarity_k5_r1e1
  ```
- **结果**: ❌ **TIMEOUT** — 运行超过 300 秒无输出
- **可能原因**: `rfc_core` 的 full-cell dataset 构建阶段耗时过长（需处理 5 组 multi-grid 数据集），或存在死循环
- **建议**: 检查 `rfc_dataset.py` 的数据构造逻辑是否有性能问题

## 3. 实验 5 Smoke（全部 grid cells 划分为多个 clients）

- **入口**: `real_data_experiments.region_client.rc_core`
- **flow_kmeans 命令**:
  ```
  python -m real_data_experiments.region_client.rc_core \
    --workflow train --partition-method flow_kmeans --num-clients 3 \
    --rounds 1 --local-epochs 1 --device cuda \
    --output-dir results/real_data_experiments/smoke/exp5_rc_flow_kmeans_k3_r1e1
  ```
- **flow_kmeans 结果**: ❌ **TIMEOUT** — 运行超过 300 秒无输出
- **spatial_block 命令**:
  ```
  python -m real_data_experiments.region_client.rc_core \
    --workflow train --partition-method spatial_block --num-clients 3 \
    --rounds 1 --local-epochs 1 --device cuda \
    --output-dir results/real_data_experiments/smoke/exp5_rc_spatial_block_k3_r1e1
  ```
- **spatial_block 结果**: ❌ **TIMEOUT** — 运行超过 300 秒无输出
- **可能原因**: `rc_core` 的 KMeans 分区或数据加载阶段耗时过长；spatial_block 也超时说明不是聚类算法本身的问题
- **建议**: 检查 `rc_core.py` 的 `build_tensor_client_data` 是否有死循环或逐 grid cell 加载优化不足

## 4. 实验 6 Smoke（全局 partition 模型结构消融）

- **入口**: `real_data_experiments.region_ablation.ra_core`
- **命令**:
  ```
  python -m real_data_experiments.region_ablation.ra_core \
    --workflow train --partition-method spatial_block --num-clients 3 \
    --variants full --rounds 1 --local-epochs 1 --device cuda \
    --output-dir results/real_data_experiments/smoke/exp6_ra_spatial_block_k3_r1e1
  ```
- **结果**: ❌ **TIMEOUT** — 运行超过 300 秒无输出
- **可能原因**: 与实验 5 共用 partition 逻辑，同源超时；模型变体初始化也可能贡献延迟
- **建议**: 先修复实验 5 的超时问题，实验 6 大概率同步解决

## 5. 超时分析

三个实验模块均在 r1e1（仅 1 轮训练，1 个 local epoch）下超时。对比实验 1 的 `sic_core` 在相同条件下仅需约 10 秒即可完成。这表明：

- 超时发生在**训练开始之前**的数据加载/分区阶段
- 三个模块都涉及全局 grid cell 分区逻辑（`region_partition.py` 或 `rfc_partition.py`）
- 分区逻辑可能在一个对所有 223 个 active grid cells 的循环中逐 cell 构建数据集，导致 O(n) 的时间复杂度×每个 cell 的 tensor slicing 开销

## 6. 下一步建议

**P0：诊断并修复超时**
1. 在 `rc_core.py` 或 `ra_core.py` 的 `build_tensor_client_data` 中添加简单的 `print` 或 `logging.info` 定位慢环节
2. 检查是否存在对所有 grid cells 的外层循环 + 每个 cell 创建完整 `GridTensorWindowDataset` 的 O(n×T) 开销
3. 考虑为 smoke（r1e1）场景跳过不必要的全量验证

**P1：实验 3/5/6 pipeline 修复后**
4. 完成实验 5/6 的 r1e1 smoke
5. 再跑 exp3 的 r1e1（similarity_k5）

**暂不进行**
- 实验 2/4 独立入口开发
- 实验 3/5/6 formal 运行
