# 区域消融实验 Notebook 迁移映射

## 当前迁移状态

本 notebook 已完成 tensor-only Python 化迁移。

- 迁移后的默认输入为：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 默认客户端定义为：
  簇级客户端联邦学习设置；每个 client = 一组 pooled grid regions。
- 默认划分方法为：
  `spatial_block`
- 可选划分方法为：
  `flow_kmeans`
- 默认联邦聚合为：
  标准样本量加权 FedAvg。
- 当前 smoke test 已通过，但 smoke test 指标不作为论文正式结果。
- smoke test 结果不作为论文正式结果。

## 审计对象

- Notebook：`test/区域客户端消融实验_2×2_最终版.ipynb`
- 已迁移到：`real_data_experiments/region_ablation/`
- 当前正式输入：tensor-only
- 正式 `tensor_path`：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 正式 `regions_path`：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`
- 当前主线约束：标准样本量加权 `FedAvg`，不引入非主线聚合

## 关键审计结论

- 原 notebook 延续了“每个 client = 一组 region indices”的定义，正式文档中统一解释为簇级客户端设置。
- 原 notebook 同样使用 `kmeans_cluster + balance_clusters_by_size` 做客户端划分。
- 原 notebook 的数据划分修复了主实验 notebook 的随机切分问题，改为对每个簇级客户端内部按连续时间切分。
- 原 notebook 的聚合是简单等权 `fedavg()`，正式迁移时需要改为标准样本量加权 `FedAvg`。
- 原 notebook 的有效消融变体是四组：`Full`、`w/o Attn`、`w/o CNN`、`w/o LSTM`。

## 迁移映射表

| Notebook | Cell/函数/类 | 原始功能 | 迁移目标 py 文件 | 处理方式 | 备注 |
|---|---|---|---|---|---|
| `区域客户端消融实验_2×2_最终版.ipynb` | Cell 0 顶部参数区 | 配置设备、轮次、学习率、客户端数等 | `region_ablation/ra_config.py` | 迁移并参数化 | 默认改为 tensor-only smoke test 配置 |
| `区域客户端消融实验_2×2_最终版.ipynb` | `extract_region_features()` | 构造区域时序聚类特征 | `common/region_partition.py` | 迁移复用 | 与主实验共享公共划分模块 |
| `区域客户端消融实验_2×2_最终版.ipynb` | `kmeans_cluster()` | KMeans 划分区域 | `common/region_partition.py` | 迁移为可选方法 | 对应正式 `flow_kmeans` |
| `区域客户端消融实验_2×2_最终版.ipynb` | `balance_clusters_by_size()` | 平衡客户端规模 | `common/region_partition.py` | 部分吸收 | 正式版仍保证每个 client 至少一个 active region |
| `区域客户端消融实验_2×2_最终版.ipynb` | `build_region_clients_cluster_balanced()` | 构造区域客户端 | `common/region_partition.py` | 迁移并泛化 | 默认方法改为 `spatial_block` |
| `区域客户端消融实验_2×2_最终版.ipynb` | `RegionDataset` | 单 client 多 region 的窗口样本 | `common/region_tensor_dataset.py` | 重写迁移 | 正式版统一支持 target time 连续划分 |
| `区域客户端消融实验_2×2_最终版.ipynb` | `split_region_dataset_timewise()` | 按时间连续切分 train/val/test | `common/region_tensor_dataset.py` | 迁移吸收 | 这是有效逻辑，正式版保留 |
| `区域客户端消融实验_2×2_最终版.ipynb` | `Attention` | 注意力模块 | `region_ablation/ra_core.py` | 复用 / 迁移 | 可直接复用单区域 Attention |
| `区域客户端消融实验_2×2_最终版.ipynb` | `CNN_LSTM_Attention` | Full 变体 | `region_ablation/ra_core.py` | 迁移 | 作为 Full |
| `区域客户端消融实验_2×2_最终版.ipynb` | `CNN_LSTM` | 去 Attention 变体 | `region_ablation/ra_core.py` | 迁移 | 映射为 `Without Attention` |
| `区域客户端消融实验_2×2_最终版.ipynb` | `LSTM_Attention` | 去 CNN 变体 | `region_ablation/ra_core.py` | 迁移 | 映射为 `Without CNN / Spatial Encoder` |
| `区域客户端消融实验_2×2_最终版.ipynb` | `CNN_Attention` | 去 LSTM 变体 | `region_ablation/ra_core.py` | 迁移 | 映射为 `Without LSTM` |
| `区域客户端消融实验_2×2_最终版.ipynb` | `evaluate_rmse()` | 计算 MSE / RMSE / MAE | `region_ablation/ra_core.py` | 重写迁移 | 正式版统一到公共指标接口 |
| `区域客户端消融实验_2×2_最终版.ipynb` | `fedavg()` | 简单等权聚合 | `common/fedavg.py` | 用公共实现替换 | 必须改成标准样本量加权 `FedAvg` |
| `区域客户端消融实验_2×2_最终版.ipynb` | `FedClient` | 本地客户端训练 | `region_ablation/ra_core.py` | 用公共训练包装替换 | 迁移为正式标准本地训练 |
| `区域客户端消融实验_2×2_最终版.ipynb` | `federated_training_with_history()` | 轮次级联邦训练与收敛记录 | `region_ablation/ra_core.py` | 迁移并重写 | 改为标准样本量加权 `FedAvg` |
| `区域客户端消融实验_2×2_最终版.ipynb` | `models = {...}` | 4 种模型变体 | `region_ablation/ra_core.py` | 迁移 | 保留为默认消融配置 |
| `区域客户端消融实验_2×2_最终版.ipynb` | Cell 1 可视化 | 2×2 消融可视化 | `region_ablation/ra_visualization.py` | 迁移并拆图 | 改为结果目录驱动的正式图表生成 |

## 重点识别结果

### 1. 原 notebook 如何构造区域客户端

- 先按区域特征聚类，再做平衡。
- 每个 client 包含多个 `region_ids`。

### 2. 原 notebook 如何做客户端聚类或区域划分

- 与主实验 notebook 类似，使用基于区域时序统计特征的聚类划分。
- 正式迁移时保留为 `flow_kmeans` 可选项。
- 默认主流程改为 `spatial_block`。

### 3. 原 notebook 的模型结构

- `Full = CNN + LSTM + Attention`
- `w/o Attn = CNN + LSTM`
- `w/o CNN = LSTM + Attention`
- `w/o LSTM = CNN + Attention`

### 4. 原 notebook 的训练循环

- 对每个消融变体分别做联邦训练。
- 每轮收集本地模型后聚合，记录测试 RMSE 历史。

### 5. 原 notebook 的 FedAvg 实现

- 使用简单等权平均 `fedavg(weights)`。
- 这与当前正式主线不一致，正式迁移必须替换为按 `n_k / sum(n_j)` 的样本量加权 `FedAvg`。

### 6. 原 notebook 中是否存在 FedProx

- 未发现明确的 `FedProx` 实现。

### 7. 原 notebook 中是否存在 server damping

- 未发现明确的 `server damping` 实现。

### 8. 原 notebook 中是否存在 personalization

- 未发现明确的 `personalization` 实现。

### 9. 原 notebook 中是否存在 loss-weighted / data-loss weighted / Proposed aggregation

- 未发现 `loss-weighted`、`data-loss weighted`、`Proposed aggregation` 或 `adaptive aggregation` 作为主流程实现。

### 10. 原 notebook 中消融实验有哪些模型变体

- `Full (CNN+LSTM+Attn)`
- `w/o Attn (CNN+LSTM)`
- `w/o CNN (LSTM+Attn)`
- `w/o LSTM (CNN+Attn)`

正式 README 中建议映射为：

- `Full`
- `Without Attention`
- `Without CNN / Spatial Encoder`
- `Without LSTM`

## 默认主流程冻结

- `data_mode = tensor`
- `client = Cluster-level Client Setting = group of pooled grid regions`
- `partition_method = spatial_block`
- `split = temporal_contiguous_by_target_time`
- `FedAvg = standard sample-size weighted FedAvg`
- `variants = Full / Without Attention / Without CNN / Without LSTM`

## 不进入默认主流程的内容

- 原 notebook 中未发现需要单独剥离的 `FedProx` / personalization / server damping 主逻辑。
- 但历史 notebook 的 cluster-balanced split 和记法差异仍应在 `historical_notes_zh.md` 中说明，不作为论文方法创新。
