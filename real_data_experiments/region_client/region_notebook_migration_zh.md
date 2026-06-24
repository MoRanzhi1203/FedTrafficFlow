# 区域主实验 Notebook 迁移映射

## 审计对象

- Notebook：`test/区域客户端计算_3×2_最终版.ipynb`
- 目标目录：`real_data_experiments/region_client/`
- 当前正式输入：tensor-only
- 正式 `tensor_path`：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 正式 `regions_path`：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`
- 当前主线约束：标准样本量加权 `FedAvg`，不把 `FedProx`、`server damping`、`personalization` 作为默认主流程

## 关键审计结论

- 原 notebook 的区域客户端定义是“每个 client = 一组 region indices”，这一点需要保留。
- 原 notebook 的默认划分方法是基于区域时序特征的 `kmeans_cluster + balance_clusters_by_size`。
- 原 notebook 的默认训练主流程混入了 `FedProx`、`server damping`、`personalization` 和 mixed raw-scale loss，这些不应进入当前正式默认主流程。
- 原 notebook 的 `split_indices()` 会先构造完整样本再随机打乱切分，不满足当前真实数据阶段必须按 target time 连续划分的要求。
- 原 notebook 中 “Independent baseline” 是有效实验对比逻辑，应保留，但需要改为与 tensor-only / 时间切分一致的正式版本。

## 迁移映射表

| Notebook | Cell/函数/类 | 原始功能 | 迁移目标 py 文件 | 处理方式 | 备注 |
|---|---|---|---|---|---|
| `区域客户端计算_3×2_最终版.ipynb` | Cell 0 顶部参数区 | 配置设备、轮次、学习率、客户端数等 | `region_client/rc_config.py` | 迁移并参数化 | 默认改为 smoke test 友好配置 |
| `区域客户端计算_3×2_最终版.ipynb` | `extract_region_features()` | 基于区域时序统计构造聚类特征 | `common/region_partition.py` | 迁移并泛化 | 正式版补充从 tensor `channel 0` 计算统计量 |
| `区域客户端计算_3×2_最终版.ipynb` | `kmeans_cluster()` | 对区域特征做 KMeans 聚类 | `common/region_partition.py` | 迁移为可选方法 | 作为 `flow_kmeans`，默认不启用 |
| `区域客户端计算_3×2_最终版.ipynb` | `balance_clusters_by_size()` | 平衡聚类后的样本量 / 区域数 | `common/region_partition.py` | 部分吸收 | 正式版优先保证每个 client 至少 1 个 active region |
| `区域客户端计算_3×2_最终版.ipynb` | `build_region_clients_cluster_balanced()` | 构造区域客户端集合 | `common/region_partition.py` | 迁移并扩展 | 正式版同时支持 `spatial_block` 和 `flow_kmeans` |
| `区域客户端计算_3×2_最终版.ipynb` | `RegionDataset` | 单个区域客户端内多 region 的窗口样本集 | `common/region_tensor_dataset.py` | 重写迁移 | 改为按 target time 连续切分，不做随机拆分 |
| `区域客户端计算_3×2_最终版.ipynb` | `split_indices()` | 随机拆分 train/val/test | `region_client/historical_notes_zh.md` | 仅记录，不迁移到默认流程 | 与当前时间序列约束冲突 |
| `区域客户端计算_3×2_最终版.ipynb` | `Attention` | 时序注意力模块 | `region_client/rc_core.py` | 复用 / 迁移 | 复用现有单区域实验 Attention 结构 |
| `区域客户端计算_3×2_最终版.ipynb` | `CNN_LSTM_Attention` | 主模型结构 | `region_client/rc_core.py` | 复用 / 迁移 | 保持 CNN + LSTM + Attention 主体 |
| `区域客户端计算_3×2_最终版.ipynb` | `evaluate()` | 原始尺度下评估 MSE / MAE | `region_client/rc_core.py` | 重写迁移 | 正式版统一到公共指标接口 |
| `区域客户端计算_3×2_最终版.ipynb` | `mixed_raw_loss()` | 混合 raw-scale 损失 | `region_client/historical_notes_zh.md` | 仅记录 | 不进入默认主流程 |
| `区域客户端计算_3×2_最终版.ipynb` | `fedavg_weighted()` | 按样本量做加权聚合 | `common/fedavg.py` | 用公共实现替换 | 默认主流程必须统一到标准样本量加权 `FedAvg` |
| `区域客户端计算_3×2_最终版.ipynb` | `FedClient` | 本地客户端训练，包含 FedProx | `region_client/historical_notes_zh.md` | 仅记录历史版本 | 正式版改用不带 FedProx 的标准本地训练 |
| `区域客户端计算_3×2_最终版.ipynb` | “Build clients” 代码段 | 基于 region split 构造 client loaders | `region_client/rc_core.py` | 迁移并重写 | 正式版按 active pooled regions 和连续时间窗口构造 |
| `区域客户端计算_3×2_最终版.ipynb` | “Strict synchronous federated training” | 主联邦训练循环 | `region_client/rc_core.py` | 迁移并净化 | 去除 `FedProx`、`server damping`，保留标准 FedAvg |
| `区域客户端计算_3×2_最终版.ipynb` | `train_local_from_scratch()` | Independent baseline | `region_client/rc_core.py` | 迁移 | 保留为默认对比基线 |
| `区域客户端计算_3×2_最终版.ipynb` | `finetune_personalized_earlystop()` | 个性化微调 | `region_client/historical_notes_zh.md` | 仅记录 | 不进入默认主流程 |
| `区域客户端计算_3×2_最终版.ipynb` | “Final evaluation on TEST split” | Fed vs Independent 最终指标 | `region_client/rc_core.py` | 迁移 | 正式版输出 `main_metrics.csv` 等文件 |
| `区域客户端计算_3×2_最终版.ipynb` | Cell 1 可视化 | 2×3 论文风格图 | `region_client/rc_visualization.py` | 迁移并拆图 | 改成结果目录驱动的正式图表生成 |

## 重点识别结果

### 1. 原 notebook 如何构造区域客户端

- 先基于所有 region 的时序特征构造 feature vectors。
- 通过 `kmeans_cluster()` 得到初始聚类。
- 通过 `balance_clusters_by_size()` 做样本量近似平衡。
- 最终每个 client 拥有多个 `region_ids`。

### 2. 原 notebook 如何做客户端聚类或区域划分

- 使用区域级统计特征：
  均值、标准差、分位数、IQR、差分统计、季节性 ACF、频谱峰值比。
- 划分方式是 “feature clustering + balancing”，不是纯空间分块。
- 正式迁移时保留为可选 `flow_kmeans`；默认方法改为 `spatial_block`。

### 3. 原 notebook 的模型结构

- `CNN_LSTM_Attention`
- 输入形式为 `x = (channels, T_in)`，其中 channels 对应张量通道。
- 输出为下一时刻 `channel 0` 的标量回归。

### 4. 原 notebook 的训练循环

- 客户端本地训练若干 epochs。
- 服务端同步等待所有客户端完成后再聚合。
- 聚合后再评估全局模型。

### 5. 原 notebook 的 FedAvg 实现

- 使用 `fedavg_weighted(weights, ns)`，按客户端训练样本量 `n_k` 加权。
- 这一点与当前正式主线一致，但正式迁移时统一调用 `common/fedavg.py`。

### 6. 原 notebook 中是否存在 FedProx

- 存在。
- 体现在 notebook 自定义 `FedClient` 的 proximal term。
- 不进入默认主流程，只记录到 `historical_notes_zh.md`。

### 7. 原 notebook 中是否存在 server damping

- 存在。
- 聚合后使用 `server_lr` 做 server-side damping。
- 不进入默认主流程，只记录到 `historical_notes_zh.md`。

### 8. 原 notebook 中是否存在 personalization

- 存在。
- 通过 `finetune_personalized_earlystop()` 对 federated server model 做个性化微调。
- 不进入默认主流程，只记录到 `historical_notes_zh.md`。

### 9. 原 notebook 中是否存在 loss-weighted / data-loss weighted / Proposed aggregation

- 未发现 `loss-weighted`、`data-loss weighted` 或 `Proposed aggregation` 作为明确实现函数。
- 但发现 mixed raw-scale loss 属于训练损失层面的历史探索。
- 该损失不作为当前正式默认设置。

### 10. 迁移结论

- 保留：tensor-only 区域客户端、区域划分、CNN+LSTM+Attention、FedAvg、Independent baseline、图表导出。
- 修正：随机切分改为按 target time 连续切分。
- 转入历史说明：`FedProx`、`server damping`、`personalization`、mixed raw-scale loss。

## 默认主流程冻结

- `data_mode = tensor`
- `client = region client = group of pooled grid regions`
- `partition_method = spatial_block`
- `split = temporal_contiguous_by_target_time`
- `FedAvg = standard sample-size weighted FedAvg`
- `Independent baseline = enabled`

## 不进入默认主流程的内容

- `FedProx`
- `server damping`
- `personalization`
- mixed raw-scale loss

这些内容来自真实实验 notebook，但属于历史探索或非主线设置；不进入默认主流程，不作为论文主方法。
