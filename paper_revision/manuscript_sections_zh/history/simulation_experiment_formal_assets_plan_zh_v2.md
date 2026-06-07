# 仿真实验正式稿图表与公式资产计划 v2

## 1. 结构定位

本资产计划服务于 `Data analysis` 章节下的 `Synthetic Experiments` 模块重写，不直接修改任何 LaTeX 文件。其用途是为后续“中文 v2 审阅完成后，再转写为英文 LaTeX 草稿”提供一套可核验、可筛选、可追踪的图表与公式清单。所有资产处理仍以标准 FedAvg 为主线，不允许将 `Proposed`、`Loss-weighted`、`Data-loss weighted` 等历史探索策略写成主方法或主结果。

## 2. 公式资产

| 编号 | 公式名称 | 建议位置 | 证据来源 | 是否进入主文 | 备注 |
|---|---|---|---|---|---|
| Eq.1 | FedAvg 聚合公式 | `## 3. FedAvg 联邦训练流程与评价指标` | `paper_revision/formula_notes/formula_fedavg_aggregation.md` | 是 | 与项目最高优先级约束完全一致 |
| Eq.2 | 本地损失函数 | `## 3` | `paper_revision/formula_notes/formula_fedavg_aggregation.md` | 是 | 对应客户端局部优化目标 |
| Eq.3 | 本地 SGD 更新 | `## 3` | `paper_revision/formula_notes/formula_fedavg_aggregation.md` | 是 | 用于连接伪代码与训练流程 |
| Eq.4 | CNN 空间编码公式 | `### 4.1 CNN-BiLSTM-Attention` | `paper_revision/formula_notes/formula_spatial_cnn.md` | 是 | 与基础 CNN 路径叙述一致 |
| Eq.5 | GCN 图卷积公式 | `### 4.2 GCN-BiLSTM-Attention` | `paper_revision/formula_notes/formula_spatial_gcn.md` | 是 | 与基础 GCN 路径叙述一致 |
| Eq.6 | MSE 指标公式 | `## 3` | 标准指标定义 | 是 | 与结果表字段对应 |
| Eq.7 | RMSE 指标公式 | `## 3` | 标准指标定义 | 是 | 与结果表字段对应 |
| Eq.8 | MAE 指标公式 | `## 3` | 标准指标定义 | 是 | 与结果表字段对应 |
| Eq.9 | MAPE 指标公式 | `## 3` | 标准指标定义 | 是 | 与结果表字段对应 |

## 3. 表格资产

| 表号 | 表题 | 数据来源 | 是否进入主文 | 备注 |
|---|---|---|---|---|
| 表 1 | 仿真实验设置表 | `cfb_core.py`；`gfb_core.py`；`cfe_core.py`；`gfe_core.py`；`fr_core.py` | 是 | 已修正基础实验划分比例与轮次冲突 |
| 表 2 | CNN-FedAvg 与 Independent 指标表 | `results/simulation_experiments/cnn_fed_base/main_summary.csv`；`main_metrics.csv` | 是 | 主文核心结果表 |
| 表 3 | GCN-FedAvg 与 Independent 指标表 | `results/simulation_experiments/gcn_fed_base/main_summary.csv` | 是 | 主文核心结果表 |
| 表 4 | FedAvg 主文证据核验表 | `cnn_enhanced_noniid_summary.csv`；`cnn_enhanced_client_scale_summary.csv`；`cnn_enhanced_feature_ablation_summary.csv` | 是 | 用于显式标记证据缺口 |
| 表 5 | 鲁棒性实验结果表 | `fed_client_dropout_summary.csv`；`fed_communication_delay_summary.csv`；`fed_gradient_noise_summary.csv` | 是 | 仅保留 FedAvg 行 |
| 表 6 | GCN 固定图与动态图结果表 | `gcn_enhanced_fixed_vs_dynamic_summary.csv` | 是 | 结果可入主文，但需写成趋势性证据 |

## 4. 伪代码资产

| 编号 | 伪代码名称 | 建议位置 | 是否进入主文 | 备注 |
|---|---|---|---|---|
| 算法 1 | FedAvg 联邦交通流预测训练流程 | `## 3. FedAvg 联邦训练流程与评价指标` | 是 | 后续可直接转为 LaTeX `algorithm` 环境 |

## 5. 图片资产总则

Markdown 正文统一使用 PNG 作为预览图；后续 LaTeX 正式排版应优先使用同名 PDF。凡在本计划中列为“可直接进入主文”的图，必须同时满足三项条件：一是 PNG 存在，二是同名 PDF 存在，三是图中未混入 `Proposed`、`Loss-weighted` 或 `Data-loss weighted` 曲线。若缺少任一条件，均不得判定为“可直接进入主文”。

## 6. 可直接进入主文

| 图号 | 图题 | PNG 预览路径 | PDF 排版路径 | PNG 是否存在 | PDF 是否存在 | 是否进入主文 | 是否需重绘 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 图 1 | CNN-FedAvg 与 Independent 指标对比 | `../../results/simulation_experiments/cnn_fed_base/main_metrics_comparison.png` | `../../results/simulation_experiments/cnn_fed_base/main_metrics_comparison.pdf` | 是 | 是 | 是 | 否 | 基础结果图，当前数据仅涉及 `FedAvg` 与 `Independent` |
| 图 2 | CNN-FedAvg 收敛曲线 | `../../results/simulation_experiments/cnn_fed_base/convergence_curve.png` | `../../results/simulation_experiments/cnn_fed_base/convergence_curve.pdf` | 是 | 是 | 是 | 否 | 可直接支撑 CNN 收敛性分析 |
| 图 3 | GCN-FedAvg 收敛曲线 | `../../results/simulation_experiments/gcn_fed_base/convergence_curve.png` | `../../results/simulation_experiments/gcn_fed_base/convergence_curve.pdf` | 是 | 是 | 是 | 否 | 可直接支撑 GCN 收敛性分析 |
| 图 4 | 基础 GCN 邻接矩阵示意 | `../../results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.png` | `../../results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.pdf` | 是 | 是 | 是 | 否 | 不涉及聚合策略混入 |
| 图 5 | 增强 GCN 固定邻接矩阵示意 | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.png` | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.pdf` | 是 | 是 | 是 | 否 | 适合用于图结构说明，非方法对比图 |

## 7. 需要重绘 Paper-Ready 版本

| 图号 | 图题 | PNG 预览路径 | PDF 排版路径 | PNG 是否存在 | PDF 是否存在 | 是否进入主文 | 是否需重绘 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 图 6 | 客户端掉线鲁棒性图 | `../../results/simulation_experiments/fed_robustness/fed_robustness_client_dropout.png` | `../../results/simulation_experiments/fed_robustness/fed_robustness_client_dropout.pdf` | 是 | 是 | 否 | 是 | 现有图包含 `FedAvg` 与 `Proposed`，需重绘 FedAvg-only 版本 |
| 图 7 | 通信延迟鲁棒性图 | `../../results/simulation_experiments/fed_robustness/fed_robustness_communication_delay.png` | `../../results/simulation_experiments/fed_robustness/fed_robustness_communication_delay.pdf` | 是 | 是 | 否 | 是 | 现有图包含 `FedAvg` 与 `Proposed` |
| 图 8 | 梯度噪声扰动鲁棒性图 | `../../results/simulation_experiments/fed_robustness/fed_robustness_gradient_noise.png` | `../../results/simulation_experiments/fed_robustness/fed_robustness_gradient_noise.pdf` | 是 | 是 | 否 | 是 | 需重绘 FedAvg-only；正文仍应写为模拟扰动 |
| 图 9 | GCN 固定图与动态图对比图 | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic.png` | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic.pdf` | 是 | 是 | 否 | 是 | 现有图混入 `Proposed` 曲线，需重绘 FedAvg-only 版本 |

## 8. 暂不进入主文

| 图号 | 图题 | PNG 预览路径 | PDF 排版路径 | PNG 是否存在 | PDF 是否存在 | 是否进入主文 | 是否需重绘 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 图 10 | Non-IID 分层对比图 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid.pdf` | 是 | 是 | 否 | 否 | 对应汇总文件缺少 FedAvg 行，暂不进入主文 |
| 图 11 | 客户端数量扩展对比图 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale.pdf` | 是 | 是 | 否 | 否 | 汇总文件缺少 FedAvg 行，暂不进入主文 |
| 图 12 | 特征消融对比图 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation.pdf` | 是 | 是 | 否 | 否 | 汇总文件缺少 FedAvg 行，暂不进入主文 |

## 9. 仅作为补充材料

| 图号 | 图题 | PNG 预览路径 | PDF 排版路径 | PNG 是否存在 | PDF 是否存在 | 是否进入主文 | 是否需重绘 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 图 13 | 增强 GCN 动态峰时图结构 | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_dynamic_peak.png` | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_dynamic_peak.pdf` | 是 | 是 | 否 | 否 | 更适合作为补充材料展示动态图差异 |
| 图 14 | 增强数据集分布示意 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_distribution.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_distribution.pdf` | 是 | 是 | 否 | 否 | 用于作者讨论和补充说明，不建议直接放入主文 |

## 10. 资产使用建议

1. 正文优先使用表 1 至表 6、算法 1 以及图 1 至图 5，以保证篇幅集中于已核验的 FedAvg 证据。
2. 图 6 至图 9 虽然 PNG 和 PDF 均已存在，但由于当前图件混入历史探索策略，必须重绘 FedAvg-only 版本后方可进入主文。
3. 图 10 至图 12 不建议在当前阶段进入主文，因为对应汇总文件缺少 FedAvg 行；若作者后续补齐结果，应同步更新证据表和审计文件。
4. 图 13 与图 14 更适合作为补充材料，用于说明动态图结构和增强数据分布背景，而不承担主结果证明职责。
