# 仿真实验正式稿一致性审计 v3

## 1. 已核验文件

### 1.1 v2 基础文档

- `paper_revision/manuscript_sections_zh/simulation_experiment_formal_module_zh_v2.md`
- `paper_revision/manuscript_sections_zh/simulation_experiment_evidence_table_zh_v2.md`
- `paper_revision/manuscript_sections_zh/simulation_experiment_formal_assets_plan_zh_v2.md`
- `paper_revision/manuscript_sections_zh/simulation_experiment_consistency_audit_zh.md`

### 1.2 新增日志与补全记录

- `simulation_experiments/simulation_missing_evidence_completion_log.md`
- `simulation_experiments/simulation_visualization_style_sync_log.md`

### 1.3 新增 FedAvg-only 结果文件

- `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.csv`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid_summary_fedavg.csv`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.csv`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale_summary_fedavg.csv`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.csv`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation_summary_fedavg.csv`

### 1.4 新增与基础图表资产

- `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.png`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.pdf`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.png`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.pdf`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.png`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.pdf`
- `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.png`
- `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.pdf`
- `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_communication_delay_fedavg_only.png`
- `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_communication_delay_fedavg_only.pdf`
- `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_gradient_noise_fedavg_only.png`
- `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_gradient_noise_fedavg_only.pdf`
- `results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.png`
- `results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.pdf`
- `results/simulation_experiments/cnn_fed_base/main_metrics_comparison.png`
- `results/simulation_experiments/cnn_fed_base/main_metrics_comparison.pdf`
- `results/simulation_experiments/cnn_fed_base/convergence_curve.png`
- `results/simulation_experiments/cnn_fed_base/convergence_curve.pdf`
- `results/simulation_experiments/gcn_fed_base/convergence_curve.png`
- `results/simulation_experiments/gcn_fed_base/convergence_curve.pdf`
- `results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.png`
- `results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.pdf`
- `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.png`
- `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.pdf`

## 2. v2 到 v3 的关键更新

| 更新项 | v2 状态 | v3 状态 | 证据文件 |
|---|---|---|---|
| Non-IID 分层结果 | 缺少 FedAvg 行，不能写入定量主结论 | 已补齐 FedAvg-only，可进入主文 | `cnn_enhanced_noniid_fedavg_only.csv`；`cnn_enhanced_noniid_summary_fedavg.csv` |
| 客户端数量扩展结果 | 缺少 FedAvg 行，不能写入定量主结论 | 已补齐 FedAvg-only，可进入主文 | `cnn_enhanced_client_scale_fedavg_only.csv`；`cnn_enhanced_client_scale_summary_fedavg.csv` |
| 特征消融结果 | 缺少 FedAvg 行，不能写入定量主结论 | 已补齐 FedAvg-only，可进入主文 | `cnn_enhanced_feature_ablation_fedavg_only.csv`；`cnn_enhanced_feature_ablation_summary_fedavg.csv` |
| 鲁棒性图表 | 原图混入 `Proposed`，需重绘 FedAvg-only | FedAvg-only 图已生成，可进入主文 | `fed_robustness_*_fedavg_only.png/pdf`；`simulation_visualization_style_sync_log.md` |
| GCN 固定图/动态图图表 | 原图混入 `Proposed`，需重绘 FedAvg-only | FedAvg-only 图已生成，可进入主文，但仅为趋势性证据 | `gcn_fixed_vs_dynamic_fedavg_only.png/pdf`；`gcn_enhanced_fixed_vs_dynamic_summary.csv` |
| 图表资产状态 | 新增图表尚未完成风格同步 | 新增图表已与基础主文图风格同步 | `simulation_visualization_style_sync_log.md` |
| 正文结论边界 | 仍需强调部分证据缺口 | 证据缺口已缩小，但梯度扰动与单种子 GCN 仍需克制表述 | 两份日志文件与新增 CSV/图表 |

## 3. 新增 FedAvg-only 证据核验

| 实验 | CSV | 是否存在 | 是否只含 FedAvg | 是否写入主文 |
|---|---|---|---|---|
| CNN 增强 Non-IID 分层 | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.csv` | 是 | 是 | 是 |
| CNN 增强客户端数量扩展 | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.csv` | 是 | 是 | 是 |
| CNN 增强特征消融 | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.csv` | 是 | 是 | 是 |
| CNN 增强 Non-IID 汇总补充 | `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid_summary_fedavg.csv` | 是 | 是 | 是 |
| CNN 增强客户端数量汇总补充 | `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale_summary_fedavg.csv` | 是 | 是 | 是 |
| CNN 增强特征消融汇总补充 | `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation_summary_fedavg.csv` | 是 | 是 | 是 |

说明：鲁棒性实验与 GCN 固定图/动态图图表的 paper-ready 更新主要体现在 FedAvg-only 图件层面，其底层数值仍来自已有汇总文件中的 FedAvg 行，而不是新增的 FedAvg-only CSV。

同时，根据 `simulation_missing_evidence_completion_log.md` 与 `simulation_visualization_style_sync_log.md` 的核验记录，可以确认用于支撑 v3 文档同步的前序阶段没有修改 LaTeX 文件，没有修改基础实验 core 逻辑，没有重跑真实数据实验，也没有改变 FedAvg 主线。v3 当前阶段本身仅新增 Markdown 文档，不涉及任何 Python 代码、结果文件或图件重生成操作。

## 4. PNG/PDF 图表核验

| 图号 | PNG | PDF | PNG 存在 | PDF 存在 | 是否进入主文 |
|---|---|---|---|---|---|
| 图 1 | `../../results/simulation_experiments/cnn_fed_base/main_metrics_comparison.png` | `../../results/simulation_experiments/cnn_fed_base/main_metrics_comparison.pdf` | 是 | 是 | 是 |
| 图 2 | `../../results/simulation_experiments/cnn_fed_base/convergence_curve.png` | `../../results/simulation_experiments/cnn_fed_base/convergence_curve.pdf` | 是 | 是 | 是 |
| 图 3 | `../../results/simulation_experiments/gcn_fed_base/convergence_curve.png` | `../../results/simulation_experiments/gcn_fed_base/convergence_curve.pdf` | 是 | 是 | 是 |
| 图 4 | `../../results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.png` | `../../results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.pdf` | 是 | 是 | 是 |
| 图 5 | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.png` | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.pdf` | 是 | 是 | 是 |
| 图 6 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.pdf` | 是 | 是 | 是 |
| 图 7 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.pdf` | 是 | 是 | 是 |
| 图 8 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.pdf` | 是 | 是 | 是 |
| 图 9 | `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.png` | `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.pdf` | 是 | 是 | 是 |
| 图 10 | `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_communication_delay_fedavg_only.png` | `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_communication_delay_fedavg_only.pdf` | 是 | 是 | 是 |
| 图 11 | `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_gradient_noise_fedavg_only.png` | `../../results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_gradient_noise_fedavg_only.pdf` | 是 | 是 | 是 |
| 图 12 | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.png` | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.pdf` | 是 | 是 | 是，但仅为趋势性证据 |

## 5. 仍需作者确认的问题

1. GCN 固定图与动态图结果当前仍为单种子证据，正式英文稿中是否仅保留趋势性表述，而不扩展为更强统计结论。
2. CNN 特征消融中 `Flow Only` 表现优于 `Full` 配置，这一现象在当前增强异质性设置下是可报告的，但是否需要在后续正文中进一步补充对特征分布差异的解释。
3. 新增的 7 张 FedAvg-only 图表是否全部进入主文，还是将其中部分保留到补充材料，以控制篇幅。
4. 后续转写英文 LaTeX 草稿时，是否需要在真实数据实验开头增加一句与仿真实验的衔接语，以突出“受控验证到真实部署”的逻辑。

## 6. 后续转 LaTeX 风险提醒

1. `main.tex` 当前相关段落若继续保留混合加权聚合表述，将与 v3 文档中“标准 FedAvg 为唯一主线”的口径冲突，后续转写时必须显式替换。
2. 旧版原始鲁棒性图与原始 GCN 固定图/动态图图仍然存在，若 LaTeX 阶段误用这些混入 `Proposed` 的旧图，会重新引入与主线不一致的视觉证据。
3. 梯度噪声图虽然已具备主文图件资格，但其文本解释必须始终限定为模拟梯度扰动，不能误写为正式差分隐私机制。
4. GCN 固定图/动态图 FedAvg-only 图虽然已可进入主文，但图注和正文必须同步保留“单种子趋势性证据”的限制，否则会出现结论强度与证据等级不匹配的问题。
5. v3 文档同步阶段并未修改任何代码、LaTeX 或结果文件；后续若作者继续补实验，必须同步更新证据表、资产计划与审计文件，而不能只改正文结论。
