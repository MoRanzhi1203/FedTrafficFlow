# 仿真实验正式稿图表与公式资产计划

## 1. 需要插入的公式

| 编号 | 公式 | 建议位置 | 来源 |
|---|---|---|---|
| Eq.1 | FedAvg 聚合公式 $\mathbf{w}^{t+1}=\sum_{k=1}^{K}\frac{n_k}{\sum_{j=1}^{K}n_j}\mathbf{w}_{k}^{t+1}$ | `## 3. 联邦训练流程与 FedAvg 聚合公式` | `paper_revision/formula_notes/formula_fedavg_aggregation.md` |
| Eq.2 | 本地损失函数 $\mathcal{L}_k(\mathbf{w})=\frac{1}{n_k}\sum_{i=1}^{n_k}\ell(\mathbf{w};x_i^k,y_i^k)$ | `## 3` | `paper_revision/formula_notes/formula_fedavg_aggregation.md` |
| Eq.3 | 本地 SGD 更新 $\mathbf{w}_k\leftarrow\mathbf{w}_k-\eta\nabla\mathcal{L}_k(\mathbf{w}_k)$ | `## 3` | `paper_revision/formula_notes/formula_fedavg_aggregation.md` |
| Eq.4 | CNN 空间卷积表示 $H_{i,t}^{(m)}=\sigma(W^{(m)}*M_{i,t}+b^{(m)})$ | `### 4.1 CNN-BiLSTM-Attention 路径` | `paper_revision/formula_notes/formula_spatial_cnn.md` |
| Eq.5 | GCN 图卷积公式 $H_t=\sigma(\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}X_tW)$ | `### 4.2 GCN-BiLSTM-Attention 路径` | `paper_revision/formula_notes/formula_spatial_gcn.md` |
| Eq.6 | MSE 指标公式 | `## 5. 对比方法、实验设置与评价指标` | 正式稿中已按标准定义写入 |
| Eq.7 | RMSE 指标公式 | `## 5` | 正式稿中已按标准定义写入 |
| Eq.8 | MAE 指标公式 | `## 5` | 正式稿中已按标准定义写入 |
| Eq.9 | MAPE 指标公式 | `## 5` | 正式稿中已按标准定义写入 |

## 2. 需要插入的表格

| 表格 | 数据来源 | 是否已可直接使用 | 是否需精简 |
|---|---|---|---|
| 表 1 仿真实验设置表 | `cfb_core.py`，`gfb_core.py`，`cfe_core.py`，`gfe_core.py`，`fr_core.py` | 是 | 否 |
| 表 2 CNN-FedAvg vs Independent 指标表 | `results/simulation_experiments/cnn_fed_base/main_summary.csv`，`main_metrics.csv` | 是 | 否 |
| 表 3 GCN-FedAvg vs Independent 指标表 | `results/simulation_experiments/gcn_fed_base/main_summary.csv` | 是 | 否 |
| 表 4 Non-IID 实验结果表 | `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid_summary.csv`，`cnn_enhanced_client_scale_summary.csv`，`cnn_enhanced_main_summary.csv` | 部分可用；仅 `cnn_enhanced_main_summary.csv` 含 FedAvg 行 | 是，需在主文中改写为证据核验表或补齐 FedAvg 结果 |
| 表 5 鲁棒性实验结果表 | `results/simulation_experiments/fed_robustness/fed_client_dropout_summary.csv`，`fed_communication_delay_summary.csv`，`fed_gradient_noise_summary.csv` | 是 | 可视篇幅合并为一张总表 |
| 表 6 特征消融结果表 | `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation_summary.csv` | 否；当前仅含 `Proposed` 行 | 是，需补齐 FedAvg 结果或保留为证据状态表 |

## 3. 需要插入的图片

| 图 | 数据或图片来源 | 当前是否存在 | 是否需要重绘 paper-ready 版本 | 备注 |
|---|---|---|---|---|
| 图 1 CNN-FedAvg 与 Independent 指标对比 | `results/simulation_experiments/cnn_fed_base/main_metrics_comparison.pdf` | 是 | 否，可直接作为占位参考 | 仅需统一论文字号和图题风格 |
| 图 2 CNN-FedAvg 收敛曲线 | `results/simulation_experiments/cnn_fed_base/convergence_curve.pdf` | 是 | 否，可直接作为占位参考 | 后续 LaTeX 中建议统一颜色和线宽 |
| 图 3 GCN-FedAvg 收敛曲线 | `results/simulation_experiments/gcn_fed_base/convergence_curve.pdf` | 是 | 否，可直接作为占位参考 | 与图 2 保持风格一致 |
| 图 4 基础 GCN 邻接矩阵示意 | `results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.pdf` | 是 | 否 | 可用于基础图结构说明 |
| 图 5 GCN 固定邻接矩阵示意 | `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.pdf` | 是 | 否 | 建议与动态图一起成组展示 |
| 图 6 Non-IID 等级对比图 | `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid.pdf` 或 `cnn_enhanced_noniid_summary.csv` | 是 | 是 | 现有图件可能含 `Proposed`，主文应重绘为 FedAvg-only 版本 |
| 图 7 鲁棒性实验图 | `results/simulation_experiments/fed_robustness/fed_robustness_client_dropout.pdf`，`fed_robustness_communication_delay.pdf`，`fed_robustness_gradient_noise.pdf` | 是 | 是 | 建议合并为一张 FedAvg-only 多子图 |
| 图 8 特征消融图 | `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation.pdf` 或 `cnn_enhanced_feature_ablation_summary.csv` | 是 | 是 | 当前结果文件缺少 FedAvg 行，不宜直接用于主文 |
| 图 9 固定图与动态图对比图 | `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic.pdf`，`gcn_enhanced_fixed_vs_dynamic_summary.csv` | 是 | 是 | 需检查是否包含 `Proposed` 曲线，如有则改绘 FedAvg-only 版本 |
| 图 10 GCN 动态图结构图 | `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_dynamic_peak.pdf`，`enhanced_gcn_dynamic_offpeak.pdf` | 是 | 否 | 用于说明早高峰、晚高峰和平峰的图结构差异 |

## 4. 伪代码

| 伪代码 | 内容 | 建议位置 |
|---|---|---|
| 算法 1 | 标准 FedAvg 联邦交通流预测训练流程 | `## 3. 联邦训练流程与 FedAvg 聚合公式` 末尾 |

## 5. 仍需作者确认的问题

1. `paper_revision/01_REVIEW_COMMENTS.md`、`02_RESPONSE_STRATEGY.md`、`03_MANUSCRIPT_EDITING_TASKS.md` 在当前目录中未找到，是否存在其他位置的替代文件需要纳入正式证据链。
2. `cnn_enhanced_noniid_summary.csv`、`cnn_enhanced_client_scale_summary.csv` 和 `cnn_enhanced_feature_ablation_summary.csv` 当前缺少 FedAvg 行，是否后续补充对应结果，或改为仅作为补充材料展示。
3. `gfe_core.py` 中增强 GCN 仅使用单个随机种子 `42`，正式论文是否只保留趋势性表述，或后续补充多种子结果。
4. 鲁棒性图、Non-IID 图和固定图/动态图对比图是否由作者后续统一重绘为论文风格矢量图。
5. 若后续转写为 LaTeX，公式编号、图表编号与 `Synthetic Experiments` 在正式论文中的章节层级是否保持为 `Data analysis` 下的一个扩展子模块。
