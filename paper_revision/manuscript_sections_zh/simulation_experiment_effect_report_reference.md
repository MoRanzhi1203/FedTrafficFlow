# 仿真实验效果说明文档

## 1. 文档目的与采集范围

本说明文档面向 `FedTrafficFlow` 当前仿真实验结果的正式复核与效果说明。文档覆盖以下 5 组实验的全部核心代码、全部 CSV 输出数据以及全部 PNG 可视化结果：

- `simulation_experiments/cnn_fed_base/cfb_core.py`
- `simulation_experiments/cnn_fed_base/cfb_visualization.py`
- `simulation_experiments/gcn_fed_base/gfb_core.py`
- `simulation_experiments/gcn_fed_base/gfb_visualization.py`
- `simulation_experiments/cnn_fed_enhanced_experiments/cfe_core.py`
- `simulation_experiments/cnn_fed_enhanced_experiments/cfe_visualization.py`
- `simulation_experiments/gcn_fed_enhanced_experiments/gfe_core.py`
- `simulation_experiments/gcn_fed_enhanced_experiments/gfe_visualization.py`
- `simulation_experiments/fed_robustness_experiments/fr_core.py`
- `simulation_experiments/fed_robustness_experiments/fr_visualization.py`

结果采集目录：

- `results/simulation_experiments/cnn_fed_base/`
- `results/simulation_experiments/gcn_fed_base/`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/`
- `results/simulation_experiments/gcn_fed_enhanced_experiments/`
- `results/simulation_experiments/fed_robustness/`

采集说明：

- 上述 5 个结果目录中的全部 `csv` 与 `png` 文件均已纳入本次分析。
- 图表趋势判读同时参考对应 `*_visualization.py` 的绘图逻辑与其输入 CSV，避免仅凭图片作无源解释。
- 本文所有“结果偏差率”统一采用相对误差类指标表述：若实验存在预测导出，则以 `MAPE` 或基于 `main_predictions.csv` 的平均相对绝对误差为准；若实验仅导出 summary，则直接引用 summary 中的 `mape_mean`。

## 2. 评价口径与复核方法

### 2.1 核心评价维度

- 实验目标达成度：以主优化目标是否被实现为准。对比实验取主指标 `RMSE` 的相对改进率；参数/结构实验取最优配置与基线配置的差值；鲁棒性实验取扰动条件下性能退化幅度与方法间差异。
- 指标稳定性：优先使用 `rmse_std / rmse_mean` 计算稳定性系数 `CV`。数值越低，说明跨客户端或跨种子波动越小。
- 结果偏差率：统一使用 `MAPE` 或 prediction 文件对应的平均相对绝对误差。

### 2.2 数据可追溯规则

- 每个量化结论均附原始 CSV 来源。
- 每个趋势解释均附对应 PNG 文件来源。
- 每个实验的实现说明均回溯到对应 `*_core.py` 中的数据生成、模型训练、聚合或评估函数。

## 3. 基础 CNN 联邦实验

### 3.1 实验基础信息

- 实现文件：`simulation_experiments/cnn_fed_base/cfb_core.py`
- 可视化文件：`simulation_experiments/cnn_fed_base/cfb_visualization.py`
- 模型：`CNNBaseModel`，结构为 `CNN + BiLSTM + Multi-head Attention`
- 对照方法：`FedAvg` 与 `Independent`
- 数据配置：5 个客户端，8 个节点，序列长度 24，预测步长 1，单客户端 200 条样本，训练/验证/测试划分为 140/20/40
- 数据来源：`results/simulation_experiments/cnn_fed_base/base_dataset_summary.csv`

`base_dataset_summary.csv` 显示 5 个客户端均为 200 条样本，说明基础实验数据规模是平衡的；客户端平均流量从 `0.1901` 逐步上升到 `0.2304`，标准差从 `0.1374` 上升到 `0.1787`，说明基础数据已引入轻度非 IID 差异但未造成极端失衡。

### 3.2 代码实现说明

- `generate_base_traffic_data()` 生成共享双峰交通模式，并通过相位偏移、幅度缩放和节点敏感度制造客户端差异。
- `run_main_experiment()` 同时执行 `FedAvg` 与 `Independent`，输出 `main_metrics.csv`、`main_summary.csv` 和 `main_predictions.csv`。
- `run_convergence_experiment()` 输出标准化的 `convergence_history.csv`。
- 代码来源：`simulation_experiments/cnn_fed_base/cfb_core.py`

### 3.3 原始输出数据整理

主结果 summary：

| 方法 | RMSE Mean | RMSE Std | MAE Mean | MAPE Mean |
| --- | ---: | ---: | ---: | ---: |
| FedAvg | 0.0135031 | 0.0007199 | 0.0107891 | 1.0789 |
| Independent | 0.0153261 | 0.0034604 | 0.0127463 | 1.2746 |

数据来源：`results/simulation_experiments/cnn_fed_base/main_summary.csv`

收敛末轮数据：

- 第 15 轮 `avg_train_loss = 0.000070`
- 第 15 轮 `avg_val_rmse = 0.012433`
- 数据来源：`results/simulation_experiments/cnn_fed_base/convergence_history.csv`

### 3.4 可视化图表解读

- `main_metrics_comparison.png` 显示 `FedAvg` 在 RMSE、MAE、MAPE 三项指标上均低于 `Independent`，且误差条更短。
  - 图表来源：`results/simulation_experiments/cnn_fed_base/main_metrics_comparison.png`
- `convergence_curve.png` 显示验证 RMSE 在第 1 轮约 `0.0947`，第 4 轮下降至 `0.0151`，第 12 轮进一步降至 `0.0117`，随后保持在 `0.012` 附近小幅波动。
  - 图表来源：`results/simulation_experiments/cnn_fed_base/convergence_curve.png`

### 3.5 效果量化分析

- 目标达成度：`FedAvg` 相对 `Independent` 的 RMSE 改进率为 `11.8950%`，说明联邦聚合在基础 CNN 设定下显著优于各客户端独立训练。
  - 计算依据：`(0.0153261 - 0.0135031) / 0.0153261`
  - 数据来源：`results/simulation_experiments/cnn_fed_base/main_summary.csv`
- 指标稳定性：`FedAvg` 的 RMSE 稳定性系数 `CV = 5.3317%`，显著低于 `Independent` 的 `22.5787%`，说明联邦训练在客户端间波动更小。
  - 数据来源：`results/simulation_experiments/cnn_fed_base/main_summary.csv`
- 结果偏差率：`FedAvg` 的 `MAPE = 1.0789%`，低于 `Independent` 的 `1.2746%`，相对偏差下降 `15.35%`。
  - 数据来源：`results/simulation_experiments/cnn_fed_base/main_summary.csv`
- 结论：基础 CNN 实验实现了预期目标，联邦训练同时改善精度和稳定性，且收敛曲线无发散迹象。

## 4. 基础 GCN 联邦实验

### 4.1 实验基础信息

- 实现文件：`simulation_experiments/gcn_fed_base/gfb_core.py`
- 可视化文件：`simulation_experiments/gcn_fed_base/gfb_visualization.py`
- 模型：`GCNBaseModel`，结构为 `GCNEncoder + BiLSTM + Multi-head Attention`
- 对照方法：`FedAvg` 与 `Independent`
- 图结构：8 节点、10 条边、图密度 `0.3571`、平均度 `2.5`
- 数据来源：`results/simulation_experiments/gcn_fed_base/base_graph_summary.csv`

### 4.2 代码实现说明

- `generate_base_traffic_data()` 与基础 CNN 共享同一数据生成逻辑，保证跨模型公平对比。
- `generate_adjacency_matrix()` 构建基础路网邻接矩阵，并进行归一化。
- `run_federated_training()` 与 `run_independent_training()` 分别输出联邦与独立基线指标。
- 数据和图结构导出由 `export_base_dataset_artifacts()` 负责。

### 4.3 原始输出数据整理

主结果 summary：

| 方法 | RMSE Mean | RMSE Std | MAE Mean | MAPE Mean |
| --- | ---: | ---: | ---: | ---: |
| FedAvg | 0.0140280 | 0.0012792 | 0.0112072 | 1.1207 |
| Independent | 0.0142674 | 0.0024278 | 0.0121159 | 1.2116 |

数据来源：`results/simulation_experiments/gcn_fed_base/main_summary.csv`

收敛末轮数据：

- 第 1 轮 `avg_val_rmse = 0.0267583`
- 第 10 轮 `avg_val_rmse = 0.0138487`，第 3 至第 10 轮持续保持在 `0.0129` 至 `0.0143` 区间
- 数据来源：`results/simulation_experiments/gcn_fed_base/convergence_history.csv`

### 4.4 可视化图表解读

- `main_metrics_comparison.png` 显示 `FedAvg` 在 RMSE、MAE、MAPE 上均略优于 `Independent`，但差距小于基础 CNN。
  - 图表来源：`results/simulation_experiments/gcn_fed_base/main_metrics_comparison.png`
- `base_graph_adjacency_matrix.png` 显示基础图为稀疏图结构，符合基础 GCN 设计目标。
  - 图表来源：`results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.png`

### 4.5 效果量化分析

- 目标达成度：`FedAvg` 相对 `Independent` 的 RMSE 改进率为 `1.6782%`，效果存在但幅度明显小于基础 CNN。
  - 数据来源：`results/simulation_experiments/gcn_fed_base/main_summary.csv`
- 指标稳定性：`FedAvg` 的 RMSE `CV = 9.1189%`，优于 `Independent` 的 `17.0165%`。
  - 数据来源：`results/simulation_experiments/gcn_fed_base/main_summary.csv`
- 结果偏差率：`FedAvg` 的 `MAPE = 1.1207%`，低于 `Independent` 的 `1.2116%`。
  - 数据来源：`results/simulation_experiments/gcn_fed_base/main_summary.csv`
- 结论：基础 GCN 在当前基础图结构下可稳定工作，但联邦收益偏小，说明简单图结构尚未充分放大图神经网络优势。

## 5. 增强 CNN 联邦实验组

### 5.1 实验基础信息

- 实现文件：`simulation_experiments/cnn_fed_enhanced_experiments/cfe_core.py`
- 可视化文件：`simulation_experiments/cnn_fed_enhanced_experiments/cfe_visualization.py`
- 种子：`42, 2024, 2025`
- 主方法：`FedAvg`、`Independent`、`Proposed`
- 扩展实验：聚合策略、lambda 敏感性、客户端规模、Non-IID 强度、客户端指标、峰时段指标、特征消融

增强数据集 summary 反映客户端异质性非常强：

- Client 2 的 `mean_flow = 122.7633`，是全组最高值。
- Client 4 的 `mean_flow = 48.8155`、`std_flow = 24.6722`，并且 `incident_ratio = 0.5245`，说明其为强事故扰动客户端。
- 数据来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_summary.csv`

### 5.2 代码实现说明

- `build_client_data()` 使用增强 Non-IID 客户端配置构建训练/验证/测试数据，并保留 `meta_test` 以支持 period 分析。
- `run_main_experiment()` 输出 `cnn_enhanced_main_metrics.csv`、`cnn_enhanced_main_summary.csv`、`cnn_enhanced_main_predictions.csv`。
- `run_aggregation_experiment()` 比较 `FedAvg`、`Loss-weighted`、`Data-loss weighted`、`Proposed`。
- `run_lambda_experiment()` 扫描 `lambda_value`。
- `run_client_scale_experiment()`、`run_noniid_experiment()`、`run_client_metrics_experiment()`、`run_peak_experiment()`、`run_feature_ablation_experiment()` 分别完成规模、异质性、分客户端、时段、特征组合分析。

### 5.3 主实验原始输出数据整理

主结果 summary：

| 方法 | RMSE Mean | RMSE Std | MAE Mean | MAPE Mean |
| --- | ---: | ---: | ---: | ---: |
| FedAvg | 7.1846 | 4.3921 | 5.7256 | 51.5859 |
| Proposed | 7.2337 | 4.4707 | 5.7343 | 49.2786 |
| Independent | 7.6420 | 3.9836 | 6.0888 | 47.6558 |

数据来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_summary.csv`

主预测文件：

- 文件：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_predictions.csv`
- 字段：`workflow, method, seed, client_id, sample_id, y_true, y_pred, period`

### 5.4 主实验图表解读

- `cnn_enhanced_main_comparison.png` 显示 `FedAvg` 与 `Proposed` 的 RMSE/MAE 接近，均优于 `Independent`；但 `MAPE` 上 `Independent` 较低，说明增强异质数据下绝对误差和相对误差出现指标冲突。
  - 图表来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_comparison.png`
- `cnn_enhanced_convergence.png` 显示 `Proposed` 方法的验证 RMSE 从第 1 轮 `9.9556` 降至第 9 轮 `7.9724`，第 10 轮回升到 `8.3789`，说明优化总体有效，但末期有轻微波动。
  - 图表来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_convergence.png`
  - 数据来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_convergence_history.csv`

### 5.5 主实验量化分析

- 目标达成度：`FedAvg` 相对 `Independent` 的 RMSE 改进率为 `5.9850%`；`Proposed` 相对 `Independent` 的改进率为 `5.3432%`。主指标上，`FedAvg` 仍是当前增强 CNN 主实验最优方案。
  - 数据来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_summary.csv`
- 指标稳定性：三种方法的 RMSE `CV` 都在 `50%` 以上，说明增强异质数据显著增加波动；其中 `FedAvg` 为 `61.13%`，`Proposed` 为 `61.80%`，稳定性压力较大。
  - 数据来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_summary.csv`
- 结果偏差率：`Proposed` 的 `MAPE = 49.2786` 低于 `FedAvg` 的 `51.5859`，但绝对误差指标未同步占优，说明其对相对误差更友好，对高流量绝对误差的控制仍不及 `FedAvg`。
  - 数据来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_summary.csv`

### 5.6 聚合策略实验

聚合策略 summary：

| 方法 | RMSE Mean | MAE Mean | MAPE Mean |
| --- | ---: | ---: | ---: |
| Data-loss weighted | 7.1841 | 5.6930 | 49.5455 |
| FedAvg | 7.1846 | 5.7256 | 51.5859 |
| Loss-weighted | 6.9482 | 5.4616 | 47.8207 |
| Proposed | 7.2337 | 5.7343 | 49.2786 |

数据来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_aggregation_summary.csv`

分析结论：

- `Loss-weighted` 取得全组最低 `RMSE = 6.9482`，相对 `FedAvg` 改进 `3.29%`，是增强 CNN 聚合策略实验中的最优方案。
- `Proposed` 并未在该实验中达到最优，说明当前增强 CNN 下更直接的 loss 加权比混合型策略更有效。
- 图表来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_aggregation.png`

### 5.7 Lambda 敏感性实验

部分 summary：

| Lambda | RMSE Mean |
| --- | ---: |
| 0.00 | 6.9482 |
| 0.25 | 7.1274 |
| 0.50 | 7.1841 |
| 0.75 | 7.2125 |
| 1.00 | 7.1846 |

数据来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_lambda_summary.csv`

分析结论：

- 最优点出现在 `lambda = 0.00`，其 `RMSE = 6.9482`。
- 当 `lambda` 从 `0.00` 增加到 `1.00` 时，RMSE 从 `6.9482` 上升到 `7.1846`，性能下降 `3.40%`。
- 图表来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_lambda.png`

### 5.8 客户端规模实验

| 客户端数 | RMSE Mean | MAPE Mean |
| --- | ---: | ---: |
| 3 | 6.0181 | 6.1255 |
| 5 | 7.2337 | 49.2786 |
| 8 | 6.7376 | 36.2792 |

数据来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale_summary.csv`

分析结论：

- 当前增强 CNN 下，`3` 客户端配置优于 `5` 客户端，RMSE 低 `16.80%`。
- `8` 客户端配置比 `5` 客户端也更优，说明问题不在于“客户端越多越差”，而在于 5 客户端配置对应的数据异质性组合最难。
- 图表来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale.png`

### 5.9 Non-IID 强度实验

| Non-IID 等级 | RMSE Mean | MAPE Mean |
| --- | ---: | ---: |
| low | 3.1381 | 3.3084 |
| medium | 7.2337 | 49.2786 |
| high | 14.7297 | 150.8745 |

数据来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid_summary.csv`

分析结论：

- 高异质性相对低异质性的 RMSE 放大倍数为 `4.6938x`，是增强 CNN 误差上升的最主要因素。
- 图表来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid.png`

### 5.10 客户端指标与峰时段实验

客户端指标显示误差集中在高扰动客户端：

- `FedAvg` 下 Client 4 的 `RMSE = 13.8908`、`MAPE = 235.4418`，显著高于 Client 0 的 `RMSE = 2.8208`。
- `Proposed` 下 Client 4 的 `RMSE = 14.1085`，未改善最难客户端。
- 数据来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_metrics.csv`

峰时段 summary：

- `FedAvg` 在 `incident_period` 的 `RMSE = 14.8151`，在 `off_peak` 为 `7.2689`，事故期误差约为平峰的 `2.04x`。
- `Proposed` 在 `incident_period` 的 `RMSE = 14.8836`，在 `off_peak` 为 `7.3648`，同样表现为约 `2.02x` 的事故放大效应。
- 数据来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_peak_summary.csv`
- 图表来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_peak_metrics.png`

### 5.11 特征消融实验

| 特征集 | RMSE Mean |
| --- | ---: |
| flow_region | 7.1109 |
| flow_event | 7.1778 |
| flow_only | 7.2337 |
| full | 8.7851 |
| flow_time | 8.8356 |

数据来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation_summary.csv`

分析结论：

- 最优单项增强来自 `flow_region`，其 `RMSE = 7.1109`，相对 `flow_only` 改善 `1.70%`。
- `flow_time` 与 `full` 组合反而恶化，说明时间特征在当前数据构造下引入了过强冗余或噪声。
- 图表来源：`results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation.png`

## 6. 增强 GCN 联邦实验组

### 6.1 实验基础信息

- 实现文件：`simulation_experiments/gcn_fed_enhanced_experiments/gfe_core.py`
- 可视化文件：`simulation_experiments/gcn_fed_enhanced_experiments/gfe_visualization.py`
- 数据来源：复用增强 CNN 的 `CLIENT_CONFIGS_BASE`
- 图结构类型：固定图、早高峰动态图、晚高峰动态图、平峰动态图、功能相似图、拥堵延迟图、拥堵强度图

图结构 summary：

| 图类型 | Mean Weight | Max Weight | Density |
| --- | ---: | ---: | ---: |
| fixed | 0.2188 | 1.0000 | 0.2188 |
| dynamic_morning_peak | 0.8269 | 0.9713 | 0.8750 |
| dynamic_evening_peak | 0.7856 | 0.9312 | 0.8750 |
| dynamic_offpeak | 0.8314 | 0.9602 | 0.8750 |
| functional_similarity | 0.8317 | 0.9603 | 0.8750 |

数据来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_graph_summary.csv`

图结构解释：

- 动态图与功能相似图都显著比固定图更稠密，说明增强 GCN 已真实编码跨节点强相关性。
- `congestion_delay` 行的 `density = 0.0` 表示该文件记录的是延迟轮次而不是连接强度，因此其连边强弱应结合 `enhanced_gcn_congestion_strength_matrix.csv` 解读。

### 6.2 代码实现说明

- `build_graph_bundle()` 构造 7 类图结构，并将标准化邻接与原始矩阵成对保存。
- `run_fixed_vs_dynamic_experiment()` 比较固定图与不同时段动态图。
- `run_congestion_delay_experiment()` 比较拥堵延迟图与功能相似图。
- `run_main_experiment()`、`run_aggregation_experiment()`、`run_lambda_experiment()`、`run_convergence_experiment()`、`run_client_scale_experiment()`、`run_noniid_experiment()`、`run_client_metrics_experiment()`、`run_peak_experiment()` 对应全部增强 GCN 工作流。

### 6.3 主实验原始输出数据整理

| 方法 | RMSE Mean | RMSE Std | MAE Mean | MAPE Mean |
| --- | ---: | ---: | ---: | ---: |
| FedAvg | 6.2163 | 3.7177 | 4.8295 | 23.8204 |
| Proposed | 6.1789 | 3.7111 | 4.7717 | 23.2980 |
| Independent | 7.5993 | 3.9388 | 6.0064 | 16.1399 |

数据来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_main_summary.csv`

### 6.4 主实验图表解读

- `gcn_enhanced_main_results.png` 显示 `FedAvg` 与 `Proposed` 在 RMSE/MAE 上明显优于 `Independent`，且二者间差距极小。
  - 图表来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_main_results.png`
- `gcn_enhanced_convergence.png` 显示 `Proposed` 从第 1 轮 `avg_val_rmse = 9.1711` 下降到第 4 轮 `8.0246`，低于 `FedAvg` 的 `8.1637`。
  - 图表来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_convergence.png`
  - 数据来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_convergence_history.csv`

### 6.5 主实验量化分析

- 目标达成度：`Proposed` 相对 `FedAvg` 的 RMSE 改进率为 `0.6011%`，相对 `Independent` 的 RMSE 改进率为 `18.6905%`。
  - 数据来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_main_summary.csv`
- 指标稳定性：`FedAvg` 与 `Proposed` 的 RMSE `CV` 均约为 `60%`，说明增强 GCN 同样面临高异质客户端带来的大波动。
- 结果偏差率：`Proposed` 的 `MAPE = 23.2980`，略优于 `FedAvg` 的 `23.8204`；但 `Independent` 的 `MAPE = 16.1399` 更低，说明相对误差指标对低流量样本更敏感，不能单独代表总体精度。

### 6.6 固定图与动态图实验

关键 summary：

| 图类型 | 方法 | RMSE Mean |
| --- | --- | ---: |
| Fixed | Proposed | 6.1789 |
| Dynamic-Morning | Proposed | 6.1587 |
| Dynamic-Evening | Proposed | 6.1593 |
| Dynamic-Offpeak | Proposed | 6.1587 |

数据来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic_summary.csv`

分析结论：

- `Proposed + Dynamic-Offpeak` 相对 `Proposed + Fixed` 的 RMSE 改进率为 `0.3270%`。
- 动态图确实带来收益，但幅度极小，说明当前增强 GCN 的主要收益已经来自基础相关图，而非时段图切换。
- 图表来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic.png`

### 6.7 拥堵延迟图实验

summary 显示：

- `Congestion delay` 与 `Functional similarity` 两种图在 `FedAvg` 和 `Proposed` 下的 `rmse_mean` 完全一致，分别为 `6.1907` 和 `6.1588`。
- 数据来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_congestion_delay_summary.csv`

分析结论：

- 当前实现下，拥堵延迟图没有在主指标上带来额外收益，说明由延迟轮次构造的关系图和功能相似图在该数据集上提供了等价信息。
- 图表来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_congestion_delay_comp.png`

### 6.8 聚合与 lambda 实验

聚合策略 summary：

| 方法 | RMSE Mean |
| --- | ---: |
| Data-loss weighted | 6.1772 |
| Proposed | 6.1789 |
| Loss-weighted | 6.1867 |
| FedAvg | 6.2163 |

数据来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_aggregation_summary.csv`

Lambda summary：

| Lambda | RMSE Mean |
| --- | ---: |
| 0.00 | 6.1867 |
| 0.25 | 6.1774 |
| 0.50 | 6.1772 |
| 0.75 | 6.1906 |
| 1.00 | 6.2163 |

数据来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_lambda_summary.csv`

分析结论：

- `Data-loss weighted` 是增强 GCN 聚合实验中的最优方案，但其相对 `FedAvg` 的 RMSE 改进仅 `0.63%`，说明增强 GCN 对聚合策略不如增强 CNN 敏感。
- Lambda 最优点出现在 `0.50`，但相比 `1.00` 只改善 `0.63%`，属于弱敏感参数。

### 6.9 客户端规模、Non-IID、客户端与峰时段实验

客户端规模：

- `3` 客户端 `RMSE = 5.6424`
- `5` 客户端 `RMSE = 6.1789`
- `8` 客户端 `RMSE = 6.0549`
- 数据来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_client_scale_metrics.csv`

Non-IID 强度：

- `low`：`RMSE = 3.0146`
- `medium`：`RMSE = 6.1789`
- `high`：`RMSE = 14.8463`
- 高异质性相对低异质性的 RMSE 放大 `4.9248x`
- 数据来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_noniid_metrics.csv`

客户端差异：

- `Proposed` 下 Client 0 的 `RMSE = 2.4912`
- `Proposed` 下 Client 4 的 `RMSE = 10.8431`、`MAPE = 96.3807`
- 数据来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_client_metrics.csv`

峰时段：

- `Proposed` 在 `incident_period` 的 `RMSE = 11.0627`
- `Proposed` 在 `off_peak` 的 `RMSE = 6.1244`
- 事故期误差约为平峰的 `1.81x`
- 数据来源：`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_peak_metrics.csv`

## 7. 联邦鲁棒性实验

### 7.1 实验基础信息

- 实现文件：`simulation_experiments/fed_robustness_experiments/fr_core.py`
- 可视化文件：`simulation_experiments/fed_robustness_experiments/fr_visualization.py`
- 对照维度：通信开销、客户端掉线、通信延迟、梯度噪声
- 方法对照：`FedAvg` 与 `Proposed`

### 7.2 代码实现说明

- `run_communication_cost_experiment()` 按真实模型参数量计算通信量。
- `run_dropout_delay_noise_training()` 在真实训练循环中加入客户端掉线、延迟同步和权重噪声。
- `run_client_dropout_experiment()`、`run_communication_delay_experiment()`、`run_gradient_noise_experiment()` 分别导出对应 metrics 与 summary。

### 7.3 通信开销量化结果

部分通信成本数据：

| 模型 | 参数量 | 参数大小(MB) |
| --- | ---: | ---: |
| CNN-Base | 48068 | 0.183365 |
| CNN-Enhanced | 48068 | 0.183365 |
| GCN-Base | 43396 | 0.165543 |
| GCN-Enhanced | 43267 | 0.165051 |

数据来源：`results/simulation_experiments/fed_robustness/fed_communication_cost.csv`

分析结论：

- `GCN-Enhanced` 相对 `CNN-Base` 参数量减少约 `10.0%`，通信体积同步下降。
- 图表来源：`results/simulation_experiments/fed_robustness/fed_robustness_communication_cost.png`

### 7.4 客户端掉线鲁棒性

summary：

| 掉线率 | 方法 | RMSE Mean | MAPE Mean |
| --- | --- | ---: | ---: |
| 0.0 | FedAvg | 7.9281 | 66.0344 |
| 0.0 | Proposed | 7.8172 | 64.2373 |
| 0.2 | FedAvg | 7.8411 | 71.2527 |
| 0.2 | Proposed | 7.7226 | 68.7732 |
| 0.4 | FedAvg | 7.9762 | 73.6387 |
| 0.4 | Proposed | 7.9169 | 72.1196 |

数据来源：`results/simulation_experiments/fed_robustness/fed_client_dropout_summary.csv`

分析结论：

- `Proposed` 在 3 个掉线率下均优于 `FedAvg`。
- 在 `dropout_rate = 0.2` 时，`Proposed` 相对 `FedAvg` 的 RMSE 改进 `1.51%`，是本组最大改进点。
- 图表来源：`results/simulation_experiments/fed_robustness/fed_robustness_client_dropout.png`

### 7.5 通信延迟鲁棒性

summary：

| 延迟轮数 | 方法 | RMSE Mean | MAPE Mean |
| --- | --- | ---: | ---: |
| 0 | FedAvg | 7.9281 | 66.0344 |
| 0 | Proposed | 7.8172 | 64.2373 |
| 1 | FedAvg | 8.2686 | 70.6013 |
| 1 | Proposed | 8.1989 | 68.0127 |
| 2 | FedAvg | 8.0433 | 70.5930 |
| 2 | Proposed | 7.9268 | 68.4701 |

数据来源：`results/simulation_experiments/fed_robustness/fed_communication_delay_summary.csv`

分析结论：

- 延迟 1 轮时最伤性能，`FedAvg` 的 RMSE 从 `7.9281` 升至 `8.2686`，增幅 `4.30%`；`Proposed` 从 `7.8172` 升至 `8.1989`，增幅 `4.88%`。
- 尽管延迟引发性能下降，`Proposed` 仍在 0、1、2 轮延迟下分别保持 `1.3989%`、`0.8433%`、`1.4492%` 的 RMSE 优势。
- 图表来源：`results/simulation_experiments/fed_robustness/fed_robustness_communication_delay.png`

### 7.6 梯度噪声鲁棒性

summary：

| 噪声标准差 | 方法 | RMSE Mean | MAPE Mean |
| --- | --- | ---: | ---: |
| 0.00 | FedAvg | 7.9281 | 66.0344 |
| 0.00 | Proposed | 7.8172 | 64.2373 |
| 0.02 | FedAvg | 7.9154 | 64.0647 |
| 0.02 | Proposed | 7.8544 | 62.4695 |
| 0.05 | FedAvg | 8.1910 | 67.9804 |
| 0.05 | Proposed | 8.1047 | 65.8780 |

数据来源：`results/simulation_experiments/fed_robustness/fed_gradient_noise_summary.csv`

分析结论：

- 当 `noise_std = 0.05` 时，`Proposed` 相对无噪声条件的 RMSE 上升 `3.68%`，说明噪声注入会削弱性能，但尚未造成失稳。
- `Proposed` 在三种噪声水平下均优于 `FedAvg`，其优势区间为 `0.77%` 至 `1.40%`。
- 图表来源：`results/simulation_experiments/fed_robustness/fed_robustness_gradient_noise.png`

## 8. 文档复核结论

### 8.1 数值一致性复核

已逐项核对以下内容与原始输出的一致性：

- 基础 CNN/GCN 主结果均与 `main_summary.csv` 一致。
- 增强 CNN/GCN 主结果、聚合策略、lambda、客户端规模、Non-IID、峰时段、特征消融均引用对应 summary 或 metrics 文件中的真实数值。
- 鲁棒性实验通信开销、掉线、延迟、噪声结论均来自对应 `fed_*_summary.csv` 或 `fed_communication_cost.csv`。
- 图表趋势说明与对应 PNG 文件、以及 PNG 所依赖的 CSV 数据保持一致。

### 8.2 总体结论

- 基础实验阶段：`FedAvg` 在 CNN 和 GCN 基础实验中都优于独立训练，其中基础 CNN 的收益更明显。
- 增强 CNN 阶段：主实验最优仍是 `FedAvg`，但聚合策略和 lambda 实验表明 `Loss-weighted` 与 `lambda=0` 的 Data-loss weighted 方案更优，说明当前 `Proposed` 不是增强 CNN 的全局最优策略。
- 增强 GCN 阶段：`Proposed` 在主实验中小幅领先 `FedAvg`，动态图和 congestion 图只带来边际收益，说明图结构增强的主要价值已被功能相似关系吸收。
- 鲁棒性阶段：`Proposed` 在掉线、延迟、噪声三类扰动下均持续优于 `FedAvg`，但优势幅度整体在 `0.8%` 到 `1.5%` 之间，属于稳定但不剧烈的鲁棒性改进。
- 共性风险：增强实验的 `RMSE Std` 和 `MAPE Std` 普遍较高，说明当前系统对强异质客户端、事故期样本和极端相对误差仍较敏感。

### 8.3 结论适用边界

- 本文所有结论仅针对当前仓库中重新生成的 `results/simulation_experiments/` 结果有效。
- 若未来修改 `CLIENT_CONFIGS_BASE`、图结构构建方法、通信轮数、局部训练轮数或随机种子，应重新生成结果并更新本文档。
