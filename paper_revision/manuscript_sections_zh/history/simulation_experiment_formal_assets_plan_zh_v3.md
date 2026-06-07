# 仿真实验正式稿图表与公式资产计划 v3

## 1. 结构定位

本资产计划服务于 `Data analysis` 章节下的 `Synthetic Experiments` 模块中文 v3 修订，不直接修改任何 LaTeX 文件。其用途是为后续“中文 v3 审阅完成后，再转写为英文 LaTeX 草稿”提供一套可核验、可筛选、可追踪的图表与公式清单。所有资产处理继续以标准 FedAvg 为主线，不允许将 `Proposed`、`Loss-weighted`、`Data-loss weighted` 等历史探索策略写成主方法或主结果。

## 2. 公式资产

| 编号 | 公式名称 | 建议位置 | 证据来源 | 是否进入主文 | 备注 |
|---|---|---|---|---|---|
| Eq.1 | FedAvg 聚合公式 | `## 3. FedAvg 联邦训练流程与评价指标` | `paper_revision/formula_notes/formula_fedavg_aggregation.md` | 是 | 与项目最高优先级约束一致 |
| Eq.2 | 本地损失函数 | `## 3` | `paper_revision/formula_notes/formula_fedavg_aggregation.md` | 是 | 对应客户端局部优化目标 |
| Eq.3 | 本地 SGD 更新 | `## 3` | `paper_revision/formula_notes/formula_fedavg_aggregation.md` | 是 | 用于连接伪代码与训练流程 |
| Eq.4 | CNN 空间编码公式 | `### 4.1 CNN-BiLSTM-Attention` | `paper_revision/formula_notes/formula_spatial_cnn.md` | 是 | 与 CNN 路径叙述一致 |
| Eq.5 | GCN 图卷积公式 | `### 4.2 GCN-BiLSTM-Attention` | `paper_revision/formula_notes/formula_spatial_gcn.md` | 是 | 与 GCN 路径叙述一致 |
| Eq.6 | MSE 指标公式 | `## 3` | 标准指标定义 | 是 | 与结果表字段对应 |
| Eq.7 | RMSE 指标公式 | `## 3` | 标准指标定义 | 是 | 与结果表字段对应 |
| Eq.8 | MAE 指标公式 | `## 3` | 标准指标定义 | 是 | 与结果表字段对应 |
| Eq.9 | MAPE 指标公式 | `## 3` | 标准指标定义 | 是 | 与结果表字段对应 |

## 3. 表格资产

| 表号 | 表题 | 数据来源 | 是否进入主文 | 备注 |
|---|---|---|---|---|
| 表 1 | 仿真实验设置表 | `cfb_core.py`；`gfb_core.py`；`cfe_core.py`；`gfe_core.py`；`fr_core.py` | 是 | 继续沿用已核验的实验设定 |
| 表 2 | CNN-FedAvg 与 Independent 指标表 | `results/simulation_experiments/cnn_fed_base/main_summary.csv`；`main_metrics.csv` | 是 | 主文核心结果表 |
| 表 3 | GCN-FedAvg 与 Independent 指标表 | `results/simulation_experiments/gcn_fed_base/main_summary.csv` | 是 | 主文核心结果表 |
| 表 4 | 不同 Non-IID 程度下 FedAvg 的预测性能 | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.csv` | 是 | v3 新增，FedAvg-only 证据已补齐 |
| 表 5 | 不同客户端数量下 FedAvg 的预测性能 | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.csv` | 是 | v3 新增，FedAvg-only 证据已补齐 |
| 表 6 | FedAvg 框架下的特征消融结果 | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.csv` | 是 | v3 新增，FedAvg-only 证据已补齐 |
| 表 7 | 鲁棒性实验结果表（FedAvg 行） | `fed_client_dropout_summary.csv`；`fed_communication_delay_summary.csv`；`fed_gradient_noise_summary.csv` | 是 | 图表已同步为 FedAvg-only 版本 |
| 表 8 | GCN 固定图与动态图结果表（FedAvg 行） | `gcn_enhanced_fixed_vs_dynamic_summary.csv` | 是 | 可进入主文，但必须写成趋势性证据 |

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
| 图 4 | 基础 GCN 邻接矩阵示意 | `../../results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.png` | `../../results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.pdf` | 是 | 是 | 是 | 否 | 不涉及探索性聚合混入 |
| 图 5 | 增强 GCN 固定邻接矩阵示意 | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.png` | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.pdf` | 是 | 是 | 是 | 否 | 适合用于图结构说明 |
| 图 6 | FedAvg 在不同 Non-IID 程度下的结果图 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.pdf` | 是 | 是 | 是 | 否 | 已完成 FedAvg-only 重绘与风格同步 |
| 图 7 | FedAvg 在不同客户端数量下的结果图 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.pdf` | 是 | 是 | 是 | 否 | 已完成 FedAvg-only 重绘与风格同步 |
| 图 8 | FedAvg 框架下的特征消融结果图 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.pdf` | 是 | 是 | 是 | 否 | 已完成 FedAvg-only 重绘与风格同步 |
| 图 9 | FedAvg 鲁棒性：客户端掉线 | `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.png` | `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.pdf` | 是 | 是 | 是 | 否 | 已完成 FedAvg-only 重绘与风格同步 |
| 图 10 | FedAvg 鲁棒性：通信延迟 | `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_communication_delay_fedavg_only.png` | `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_communication_delay_fedavg_only.pdf` | 是 | 是 | 是 | 否 | 已完成 FedAvg-only 重绘与风格同步 |
| 图 11 | FedAvg 鲁棒性：模拟梯度扰动 | `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_gradient_noise_fedavg_only.png` | `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_gradient_noise_fedavg_only.pdf` | 是 | 是 | 是 | 否 | 图中保留非正式差分隐私说明 |
| 图 12 | FedAvg 下固定图与动态图结构对比 | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.png` | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.pdf` | 是 | 是 | 是 | 否 | 单种子趋势性证据，图中保留趋势提示 |

## 7. 不进入主文

| 图号 | 图题 | PNG 预览路径 | PDF 排版路径 | PNG 是否存在 | PDF 是否存在 | 是否进入主文 | 是否需重绘 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 图 13 | 原始 Non-IID 分层对比图 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid.pdf` | 是 | 是 | 否 | 否 | 原图口径不满足 v3 的 FedAvg-only 主文要求 |
| 图 14 | 原始客户端数量扩展对比图 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale.pdf` | 是 | 是 | 否 | 否 | 使用旧版混合图已无必要 |
| 图 15 | 原始特征消融对比图 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation.pdf` | 是 | 是 | 否 | 否 | 已被 FedAvg-only paper-ready 图替代 |
| 图 16 | 原始客户端掉线鲁棒性图 | `../../results/simulation_experiments/fed_robustness/fed_robustness_client_dropout.png` | `../../results/simulation_experiments/fed_robustness/fed_robustness_client_dropout.pdf` | 是 | 是 | 否 | 否 | 旧图混入 `Proposed` 曲线，不再进入主文 |
| 图 17 | 原始通信延迟鲁棒性图 | `../../results/simulation_experiments/fed_robustness/fed_robustness_communication_delay.png` | `../../results/simulation_experiments/fed_robustness/fed_robustness_communication_delay.pdf` | 是 | 是 | 否 | 否 | 旧图混入 `Proposed` 曲线，不再进入主文 |
| 图 18 | 原始梯度噪声鲁棒性图 | `../../results/simulation_experiments/fed_robustness/fed_robustness_gradient_noise.png` | `../../results/simulation_experiments/fed_robustness/fed_robustness_gradient_noise.pdf` | 是 | 是 | 否 | 否 | 旧图混入 `Proposed` 曲线，不再进入主文 |
| 图 19 | 原始 GCN 固定图与动态图对比图 | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic.png` | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic.pdf` | 是 | 是 | 否 | 否 | 旧图混入 `Proposed` 曲线，不再进入主文 |

## 8. 仅作为补充材料

| 图号 | 图题 | PNG 预览路径 | PDF 排版路径 | PNG 是否存在 | PDF 是否存在 | 是否进入主文 | 是否需重绘 | 备注 |
|---|---|---|---|---|---|---|---|---|
| 图 20 | 增强 GCN 动态峰时图结构 | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_dynamic_peak.png` | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_dynamic_peak.pdf` | 是 | 是 | 否 | 否 | 更适合作为补充材料展示动态图差异 |
| 图 21 | 增强数据集分布示意 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_distribution.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/enhanced_dataset_distribution.pdf` | 是 | 是 | 否 | 否 | 可用于背景说明，不承担主结果证明职责 |

## 9. 资产使用建议

1. 正文优先使用表 1 至表 8、算法 1 以及图 1 至图 12，以保证篇幅集中于已核验的 FedAvg 证据。
2. 图 6 至图 12 已同时满足 PNG/PDF 成对存在、FedAvg-only 和风格同步完成三项条件，因此可以直接进入中文正式稿和后续 LaTeX 图件清单。
3. 图 13 至图 19 代表旧版混合方法图件，即使文件存在，也不应再被作为主文图引用，以免重新引入与 FedAvg 主线冲突的视觉证据。
4. 图 20 与图 21 更适合作为补充材料，用于说明动态图结构差异和增强数据分布背景，而不承担主结果证明职责。
