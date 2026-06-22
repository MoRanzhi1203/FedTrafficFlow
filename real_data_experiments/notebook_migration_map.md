# Notebook Migration Map / Notebook 迁移映射

## 约束摘要

- 主方法严格保持标准样本量加权 `FedAvg`，不得把 `Proposed`、`Loss-weighted`、`FedProx`、`server damping`、`personalization`、`adaptive aggregation` 写入真实数据主实验默认流程。
- 真实数据主线优先使用可审计的数据入口 `data/analysis/node_intersection_flow_parquet/`，不再依赖 notebook 中未纳入仓库的黑箱中间产物 `6.池化网格张量.pt`。
- 交通时序数据默认采用时间顺序划分，修复 `train/val/test` 泄漏与随机打乱导致的未来信息泄漏。
- notebook 中的临时打印、交互式展示、一次性绘图样式、paper-level 拼图代码只保留为迁移参考，不直接作为正式工程核心逻辑。

## 目标 Notebook

| Notebook | Cell 范围 | 原始功能 | 迁移到的 py 文件 | 函数/类名称 | 是否修改逻辑 | 备注 |
|---|---|---|---|---|---|---|
| `test/单路口客户端计算_3×2.ipynb` | Cell 0, 1-63 行左右 | 环境导入、随机种子、样式、`6.池化网格张量.pt` 读取 | `real_data_experiments/single_intersection_client/sic_config.py`、`real_data_experiments/common/io_utils.py`、`real_data_experiments/common/seed.py` | `ExperimentConfig`、`load_single_intersection_tensor()`、`set_global_seed()` | 是 | notebook 依赖的 `6.池化网格张量.pt` 当前仓库中不存在，需改为从 `data/analysis/node_intersection_flow_parquet/` 显式构造时序张量。 |
| `test/单路口客户端计算_3×2.ipynb` | Cell 0, 64-121 行左右 | `IntersectionDataset`、`AdaptiveSwish`、`CNN_LSTM_Attention` 模型定义 | `real_data_experiments/single_intersection_client/sic_core.py` | `IntersectionDataset`、`AdaptiveSwish`、`CNNLSTMAttentionRegressor` | 否 | 保留模型结构思路，类名与模块名按工程规范整理。 |
| `test/单路口客户端计算_3×2.ipynb` | Cell 0, 122-197 行左右 | `evaluate()`、`FedClient` 本地训练逻辑 | `real_data_experiments/common/client.py`、`real_data_experiments/common/trainer.py`、`real_data_experiments/common/metrics.py` | `evaluate_regression_metrics()`、`FedClient.train_one_round()`、`run_federated_rounds()` | 是 | 当前 notebook 将整套 dataset 同时用于训练与“验证”，存在验证泄漏，必须拆分为 `train/val/test`。 |
| `test/单路口客户端计算_3×2.ipynb` | Cell 0, 198-244 行左右 | `fedavg()` 聚合与联邦轮次训练 | `real_data_experiments/common/fedavg.py`、`real_data_experiments/common/trainer.py` | `fedavg_aggregate()`、`run_federated_training()` | 是 | 仅保留标准样本量加权 `FedAvg`，增加输入校验与不可变返回。 |
| `test/单路口客户端计算_3×2.ipynb` | Cell 0, 245-413 行左右 | Independent 训练、指标汇总、图表绘制 | `real_data_experiments/single_intersection_client/sic_core.py`、`real_data_experiments/single_intersection_client/sic_visualization.py`、`real_data_experiments/common/result_writer.py` | `run_independent_baseline()`、`export_single_intersection_results()`、`plot_single_intersection_figures()` | 是 | 训练与可视化必须拆分；图表只读取 CSV/JSON 结果。 |
| `test/单路口客户端消融实验_2×2.ipynb` | Cell 0, 1-81 行左右 | 真实数据读取、`IntersectionDataset`、Full/消融模型定义 | `real_data_experiments/single_intersection_ablation/sia_config.py`、`real_data_experiments/single_intersection_ablation/sia_core.py` | `ExperimentConfig`、`IntersectionDataset`、`build_ablation_model()` | 否 | 保留 Full / w/o Attention / w/o spatial encoder / w/o LSTM 的结构对应关系。 |
| `test/单路口客户端消融实验_2×2.ipynb` | Cell 0, 82-234 行左右 | 时间顺序划分、`fedavg()`、评估函数 | `real_data_experiments/common/data_splits.py`、`real_data_experiments/common/fedavg.py`、`real_data_experiments/common/metrics.py` | `temporal_train_val_test_split()`、`fedavg_aggregate()`、`compute_regression_metrics()` | 否 | 当前 notebook 的时序划分思路可保留，但需统一到公共模块。 |
| `test/单路口客户端消融实验_2×2.ipynb` | Cell 0, 235-537 行左右 | 联邦消融训练、收敛曲线、稳定性图 | `real_data_experiments/single_intersection_ablation/sia_core.py`、`real_data_experiments/single_intersection_ablation/sia_visualization.py` | `run_ablation_experiment()`、`plot_ablation_comparison()` | 是 | 统一输出 test-set 指标、每客户端指标、收敛历史与 2×2 图。 |
| `test/区域客户端计算_3×2_最终版.ipynb` | Cell 0, 1-262 行左右 | 区域特征抽取、KMeans 聚类、均衡区域客户端划分、`RegionDataset` | `real_data_experiments/region_client/rc_config.py`、`real_data_experiments/region_client/rc_core.py`、`real_data_experiments/common/io_utils.py` | `ExperimentConfig`、`extract_region_features()`、`build_region_clients_cluster_balanced()`、`RegionDataset` | 部分 | 保留“区域/聚类客户端”思想，但需要把区域划分结果导出为 `region_assignment.csv`、`region_summary.csv`。 |
| `test/区域客户端计算_3×2_最终版.ipynb` | Cell 0, 263-354 行左右 | 数据集索引、`split_indices()` 随机划分、评估函数 | `real_data_experiments/common/data_splits.py`、`real_data_experiments/common/metrics.py` | `temporal_train_val_test_split()`、`compute_regression_metrics()` | 是 | 当前 `split_indices()` 会随机打乱样本，破坏时间顺序，必须改为按时间顺序划分。 |
| `test/区域客户端计算_3×2_最终版.ipynb` | Cell 0, 355-605 行左右 | `mixed_raw_loss`、`fedavg_weighted()`、FedProx、本地训练、server damping、personalization | `real_data_experiments/common/fedavg.py`、`real_data_experiments/common/client.py`、`real_data_experiments/common/trainer.py`、`real_data_experiments/region_client/historical_notes_zh.md` | `fedavg_aggregate()`、`FedClient`、`run_federated_training()` | 是 | `FedProx`、`server damping`、`personalization` 不进入主实验，只在历史说明中登记。 |
| `test/区域客户端计算_3×2_最终版.ipynb` | Cell 0, 606-726 行左右 + Cell 1 | 主结果汇总、稳定性统计、2×3 可视化 | `real_data_experiments/region_client/rc_core.py`、`real_data_experiments/region_client/rc_visualization.py`、`real_data_experiments/common/result_writer.py` | `run_region_client_experiment()`、`plot_region_client_figures()` | 是 | 可视化改为只读结果文件；不得在 `visualization.py` 中重新训练。 |
| `test/区域客户端消融实验_2×2_最终版.ipynb` | Cell 0, 1-244 行左右 | 区域客户端构造、`RegionDataset`、四种模型变体 | `real_data_experiments/region_ablation/ra_config.py`、`real_data_experiments/region_ablation/ra_core.py` | `ExperimentConfig`、`RegionDataset`、`build_region_ablation_model()` | 否 | 可复用区域客户端构造逻辑，但要和区域主实验共享划分结果。 |
| `test/区域客户端消融实验_2×2_最终版.ipynb` | Cell 0, 245-397 行左右 | 连续时间划分、`fedavg()`、评估函数 | `real_data_experiments/common/data_splits.py`、`real_data_experiments/common/fedavg.py`、`real_data_experiments/common/metrics.py` | `temporal_train_val_test_split()`、`fedavg_aggregate()`、`compute_regression_metrics()` | 否 | 当前 notebook 已经修复为连续时间划分，可作为正向参考。 |
| `test/区域客户端消融实验_2×2_最终版.ipynb` | Cell 0, 398-592 行左右 + Cell 1 | 消融联邦训练、2×2 论文图 | `real_data_experiments/region_ablation/ra_core.py`、`real_data_experiments/region_ablation/ra_visualization.py` | `run_region_ablation_experiment()`、`plot_region_ablation_figures()` | 是 | 输出必须补齐 `ablation_metrics.csv`、`ablation_summary.csv`、`ablation_client_metrics.csv`。 |

## 参考 Notebook（只读参考，不并入真实实验主线）

| Notebook | 用途 | 可复用内容 | 不直接迁移内容 |
|---|---|---|---|
| `test/CCN仿真.ipynb` | 仿真 CNN-FedAvg 参考 | 模型命名、Independent/FedAvg 对比思路、收敛指标结构 | 合成数据生成、仿真数据划分、仿真结论 |
| `test/GCN仿真.ipynb` | 仿真 GCN-FedAvg 参考 | GCN 实验组织方式、结果导出结构 | 合成图结构、仿真数据逻辑 |
| `test/预处理1.ipynb` ~ `test/预处理6.ipynb` | 真实数据预处理历史来源 | 路网清洗、节点映射、网格化历史线索 | notebook 式线性处理、相对路径、未版本化中间文件 `6.池化网格张量.pt` |

## 审计发现的关键风险

### 1. 数据入口不可复现

- 4 个真实实验 notebook 默认读取 `6.池化网格张量.pt`，但当前仓库中未发现该文件。
- 因此正式 Python 工程必须改为从 `data/analysis/node_intersection_flow_parquet/` 重建输入张量和窗口样本，并记录转换参数。

### 2. 单路口主实验存在验证泄漏

- `单路口客户端计算_3×2.ipynb` 中 `FedClient` 的 `train_loader` 与 `val_loader` 都直接来自同一份完整 `dataset`。
- 该 notebook 需要显式修复为 `train/val/test` 三段式时间顺序划分。

### 3. 区域主实验存在未来信息泄漏

- `区域客户端计算_3×2_最终版.ipynb` 使用 `split_indices()` 先随机打乱样本再切分 `train/val/test`。
- 对时序交通流任务，这会造成未来信息泄漏，不符合默认时间顺序划分要求。

### 4. 区域主实验与 FedAvg 主线冲突

- `区域客户端计算_3×2_最终版.ipynb` 明确把 `FedProx`、`server damping`、`personalization` 写入主训练流程。
- 迁移时必须降级为“历史探索”，默认主工作流仅保留标准样本量加权 `FedAvg`。

### 5. 可视化与训练未分离

- 目标 notebook 目前将训练、评估、汇总、论文拼图绘制混在同一 cell 或同一 notebook 中。
- 迁移后必须统一为 `*_core.py` 负责训练与导出结果，`*_visualization.py` 只读取已有 CSV/JSON 进行绘图。

## 待新增/待修改文件清单（Phase 0 规划）

### 待新增目录

- `real_data_experiments/common/`
- `real_data_experiments/single_intersection_client/`
- `real_data_experiments/single_intersection_ablation/`
- `real_data_experiments/region_client/`
- `real_data_experiments/region_ablation/`
- `results/real_data_experiments/`

### 待新增核心文件

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
- `real_data_experiments/real_experiment_report.md`

### 优先复用的现有文件

- `analysis_scripts/preprocessing/compute_node_intersection_flow_optimized.py`
- `analysis_scripts/preprocessing/fit_node_flow_daily_curve.py`
- `docs/project_pipeline.md`
- `paper_revision/manuscript_sections_zh/current/real_data_prediction_pipeline_next_steps_zh.md`
- `paper_revision/manuscript_sections_zh/current/real_data_preprocessing_audit_zh.md`
- `simulation_experiments/cnn_fed_base/cfb_core.py`

## Phase 0 结论

- 已确认 4 个真实实验 notebook 可以迁移，但必须先替换不可复现的数据入口，并修复主实验中的数据泄漏与 `FedAvg` 主线冲突。
- 其中 `单路口客户端消融实验_2×2.ipynb` 与 `区域客户端消融实验_2×2_最终版.ipynb` 的连续时间划分思路可直接吸收为正式工程实现参考。
- 其中 `区域客户端计算_3×2_最终版.ipynb` 的区域构造思路可保留，但其主训练流程必须剥离 `FedProx / server damping / personalization`。
