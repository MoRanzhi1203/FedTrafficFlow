# Real Data Experiment Report / 真实数据实验报告

## 1. 实验迁移说明

- 本报告服务于真实数据实验 notebook 向 Python 工程代码迁移的 Phase 0 审计与后续实现规划。
- 本次迁移严格遵守最高优先级约束：论文主线保持为标准样本量加权 `FedAvg`，不提出新的联邦聚合算法。
- 当前报告仅基于项目现有 notebook、`analysis_scripts/`、`docs/` 与 `paper_revision/` 材料整理，不编造实验结果，不修改 LaTeX 正文，不改动 `simulation_experiments/` 核心训练逻辑。

## 2. Notebook 到 py 映射

- 详见 `real_data_experiments/notebook_migration_map.md`。
- 迁移重点为 4 个真实实验 notebook：
- `test/单路口客户端计算_3×2.ipynb`
- `test/单路口客户端消融实验_2×2.ipynb`
- `test/区域客户端计算_3×2_最终版.ipynb`
- `test/区域客户端消融实验_2×2_最终版.ipynb`

## 3. 实验结构

- 目标工程结构采用“一个实验组目录 = 一个 `core.py` + 一个 `visualization.py`”。
- 通用功能沉淀到 `real_data_experiments/common/`，仅包含随机种子、指标、数据划分、标准 `FedAvg`、结果写出等基础模块。
- 单路口与区域实验分开维护；主实验与消融实验分开维护；训练与可视化分离。

## 4. 数据来源与数据划分

### 已确认的数据来源

- 真实数据基础输入来自：
- `data/raw/link_gps.v2`
- `data/raw/road_network_sub-dataset.v2`
- `data/raw/traffic_speed_sub-dataset.v2`

### 已确认的可复用中间结果

- `data/analysis/node_intersection_flow_parquet/node_flow_chunk_*.parquet`
- `data/processed/rnsd_processed.csv`

### 当前正式数据入口

- 当前正式 tensor-only 输入为 `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 配套 sidecar 为：
- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`
- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor_metadata.json`
- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_metadata.json`
- 正式 tensor shape 为 `(2, 630, 5856)`。
- 正式 `pool_mode = sum_mean`。
- 正式 `layout = standard`，即 `row = lat`、`col = lon`。

### 当前发现的问题

- 4 个真实实验 notebook 曾默认读取 `6.池化网格张量.pt`，但该命名只属于历史 notebook 临时文件。
- 当前工程化代码不再生成该文件，正式入口统一为 `node_flow_grid_tensor.pt`。
- 此前 `parquet-direct` 版本仅用于 smoke test fallback，不应再视为正式主实验输入。
- 单路口主实验 notebook 未实现严格的 `train/val/test` 拆分。
- 区域主实验 notebook 采用随机打乱再切分，存在未来信息泄漏风险。

### 迁移后的默认划分原则

- 默认使用时间顺序划分：
- `train = 0.70`
- `val = 0.15`
- `test = 0.15`
- 若某实验沿用不同划分，则必须写入配置、结果文件与中文说明中。

## 5. FedAvg 实现说明

### 必须保留的主线

- 本文采用标准样本量加权 `FedAvg`：

$$
\mathbf{w}^{t+1}
=
\sum_{k=1}^{K}
\frac{n_k}{\sum_{j=1}^{K} n_j}
\mathbf{w}_k^{t+1}
$$

**English:**
The server aggregates local model parameters using only the client training sample counts.

**中文：**
服务器仅根据各客户端训练样本数进行加权聚合。

### 迁移时必须执行的限制

- 只允许保留标准样本量加权。
- 不默认引入 `loss-weighted`。
- 不默认引入 `data-loss weighted`。
- 不默认引入 `similarity-aware`。
- 不默认引入 `quality-weighted`。
- 不默认引入 `FedProx`。
- 不默认引入 `server damping`。
- 不默认引入 `personalization`。
- 不默认引入 `adaptive aggregation`。

## 6. 单路口客户端实验

### 审计结论

- notebook 保留了“单个空间网格/路口作为客户端”的核心实验思想。
- 模型主体为 `CNN + LSTM + Attention`。
- 已包含 `FedAvg` 与 `Independent` 对比思路。
- 但当前 `FedClient` 的 `train_loader` 与 `val_loader` 共用同一份 dataset，存在验证泄漏。

### 迁移策略

- 保留模型结构与联邦/独立对比主线。
- 当前单路口主实验已切换到正式 tensor-only 输入，不再默认使用 `parquet-direct`。
- 当前 `client` 表示 `pooled-grid-region client`，默认仅使用 active pooled regions。
- 将数据读取、窗口构造、时间划分、客户端训练、指标计算和结果导出拆分到正式模块。
- 增加每客户端指标、总体平均指标、标准差、变异系数和收敛历史导出。

## 7. 单路口客户端消融实验

### 审计结论

- notebook 已实现 `Full / CNN+LSTM / LSTM+Attention / CNN+Attention` 四组对比。
- 当前连续时间划分逻辑相对规范，可作为正式工程参考。
- 需要统一消融命名到：
- `Full model`
- `Without Attention`
- `Without CNN / spatial encoder`
- `Without LSTM`

### 迁移策略

- 保留消融对比范围，不改变 `FedAvg` 聚合。
- 当前单路口消融已切换到正式 tensor-only 输入，不再默认使用 `parquet-direct`。
- 对齐单路口主实验的数据划分、训练轮数、batch size 与指标输出格式。

## 8. 区域客户端实验

### 审计结论

- notebook 保留了“基于区域/聚类的客户端构造”核心思想。
- 已实现区域特征抽取、KMeans 聚类与样本量平衡。
- 但主训练流显式包含 `FedProx`、`server damping`、`personalization`，与本次迁移约束冲突。
- 数据划分函数 `split_indices()` 采用随机打乱样本，不符合交通时序默认时间顺序划分要求。

### 迁移策略

- 保留“区域客户端构造”和“区域级指标分析”。
- 删除默认主流程中的 `FedProx`、`server damping`、`personalization`。
- 将区域划分、non-IID 分布与样本量统计单独导出为结果文件。
- 主实验默认只运行标准 `FedAvg` 与 `Independent`。

## 9. 区域客户端消融实验

### 审计结论

- notebook 已经采用“按区域内每个节点连续时间切分”的更合理划分方式。
- 4 种模型变体与区域主实验结构相匹配。
- 适合作为区域主实验的规范化迁移模板之一。

### 迁移策略

- 对齐区域主实验的区域划分结果。
- 保持相同的数据划分、训练轮次与 `FedAvg` 聚合。
- 输出区域客户端级别的消融指标、总体均值与波动统计。

## 10. 一审意见对应修改

| 一审方向 | 当前问题 | 迁移后补强 |
|---|---|---|
| 可复现性 | notebook 依赖缺失中间文件，参数分散 | 引入配置文件、运行命令记录、环境摘要、数据划分摘要 |
| 指标完整性 | notebook 指标不统一、输出不标准 | 统一到 `MSE/RMSE/MAE/MAPE/sMAPE/R2` 并同时输出 CSV/JSON |
| 客户端异质性 | 缺少规范化的样本量/目标分布导出 | 增加 `client_sample_distribution.csv`、`client_target_distribution.csv` 等 |
| 收敛性 | 仅 notebook 内部画图 | 增加 `convergence_history.csv` 与正式 `visualization.py` |
| 消融实验 | 命名与主实验结果接口不统一 | 统一变体命名、对齐数据划分与导出格式 |
| 审稿支撑文档 | 缺少正式迁移报告 | 生成中文说明、图表说明、执行说明与总报告 |

## 11. 仍需作者确认的问题

- `6.池化网格张量.pt` 的原始生成方式是否必须完全复刻，还是以 `node_intersection_flow_parquet` 为新的正式数据入口。
- 单路口客户端在真实论文中的“客户端”究竟是单节点、单网格，还是 notebook 中以索引 `k` 表示的单一空间位置。
- 区域客户端构造中是否需要固定为 `3` 个区域客户端，还是允许通过 CLI 调整。
- 当前真实实验默认预测目标是否固定为 `路口车流量`，以及是否需要保留 `路口进入流量 / 路口离开流量` 的扩展入口。
- 区域实验是否需要沿用 notebook 中的 `stride = 6` 以降低样本量，还是以完整窗口采样为默认设置。

## 12. 不进入论文主文的历史探索内容

- 以下内容若在 notebook 中出现，只作为历史探索记录，不进入主实验默认流程，也不写成论文创新：
- `FedProx`
- `server damping`
- `personalization`
- `loss-weighted aggregation`
- `data-loss weighted aggregation`
- `similarity-aware aggregation`
- `quality-weighted aggregation`
- `adaptive aggregation`
- `Proposed aggregation`

## 待修改文件清单

### 必须新增

- `real_data_experiments/common/seed.py`
- `real_data_experiments/common/metrics.py`
- `real_data_experiments/common/io_utils.py`
- `real_data_experiments/common/data_splits.py`
- `real_data_experiments/common/fedavg.py`
- `real_data_experiments/common/client.py`
- `real_data_experiments/common/trainer.py`
- `real_data_experiments/common/result_writer.py`
- `real_data_experiments/single_intersection_client/sic_config.py`
- `real_data_experiments/single_intersection_client/sic_core.py`
- `real_data_experiments/single_intersection_client/sic_visualization.py`
- `real_data_experiments/single_intersection_ablation/sia_config.py`
- `real_data_experiments/single_intersection_ablation/sia_core.py`
- `real_data_experiments/single_intersection_ablation/sia_visualization.py`
- `real_data_experiments/region_client/rc_config.py`
- `real_data_experiments/region_client/rc_core.py`
- `real_data_experiments/region_client/rc_visualization.py`
- `real_data_experiments/region_ablation/ra_config.py`
- `real_data_experiments/region_ablation/ra_core.py`
- `real_data_experiments/region_ablation/ra_visualization.py`
- `real_data_experiments/RUN_REAL_EXPERIMENTS_zh.md`

### 必须复核但不改动

- `test/单路口客户端计算_3×2.ipynb`
- `test/单路口客户端消融实验_2×2.ipynb`
- `test/区域客户端计算_3×2_最终版.ipynb`
- `test/区域客户端消融实验_2×2_最终版.ipynb`
- `analysis_scripts/preprocessing/compute_node_intersection_flow_optimized.py`
- `docs/project_pipeline.md`
- `simulation_experiments/*`

## 当前阶段结论

- Phase 0 已识别出 notebook 迁移的主要风险与模块映射关系。
- 继续推进时，应先创建 `real_data_experiments/` 工程骨架，再实现公共模块，优先完成单路口主实验的最小可运行版本。
