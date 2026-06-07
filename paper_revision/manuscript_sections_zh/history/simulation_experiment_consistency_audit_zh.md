# 仿真实验正式稿一致性审计

## 1. 已核验文件

### 1.1 v1 与配套文档

- `paper_revision/manuscript_sections_zh/simulation_experiment_formal_module_zh.md`
- `paper_revision/manuscript_sections_zh/simulation_experiment_evidence_table_zh.md`
- `paper_revision/manuscript_sections_zh/simulation_experiment_formal_assets_plan_zh.md`
- `paper_revision/manuscript_sections_zh/simulation_experiment_formal_checklist_zh.md`
- `paper_revision/manuscript_sections_zh/simulation_experiment_insertion_plan_zh.md`
- `paper_revision/manuscript_sections_zh/simulation_experiment_section_zh.md`

### 1.2 约束文件

- `paper_revision/00_NATURE_SKILLS_PRECHECK.md`
- `paper_revision/01_prerequisite_constraints.md`
- `paper_revision/02_revision_strategy.md`
- `paper_revision/03_context_aware_response_strategy.md`
- `paper_revision/04_SIMULATION_EXPERIMENT_OPTIMIZATION_SCOPE.md`
- `paper_revision/05_BILINGUAL_DOCUMENTATION_CONSTRAINT.md`
- `paper_revision/project_context/aggregation_strategy_classification_and_prohibition.md`
- `paper_revision/project_context/experiment_overview_and_key_results.md`
- `paper_revision/project_context/model_architecture_overview.md`
- `paper_revision/formula_notes/formula_fedavg_aggregation.md`
- `paper_revision/formula_notes/formula_spatial_cnn.md`
- `paper_revision/formula_notes/formula_spatial_gcn.md`

### 1.3 正式论文材料

- `paper_revision/latex_source/main.tex`
- `paper_revision/latex_source/main.pdf`
- `paper_revision/original_submission/Federated Learning Approach for Multi-Regional Traffic Flow Prediction.pdf`

说明：两份 PDF 文件均已读取其二进制内容并确认存在；由于 PDF 直接文本提取不稳定，本次章节定位、结构衔接和风格参照主要依据 `main.tex` 完成。

### 1.4 仿真实验代码与结果

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
- `results/simulation_experiments/cnn_fed_base/main_summary.csv`
- `results/simulation_experiments/cnn_fed_base/main_metrics.csv`
- `results/simulation_experiments/cnn_fed_base/convergence_history.csv`
- `results/simulation_experiments/gcn_fed_base/main_summary.csv`
- `results/simulation_experiments/gcn_fed_base/main_metrics.csv`
- `results/simulation_experiments/gcn_fed_base/convergence_history.csv`
- `results/simulation_experiments/gcn_fed_base/base_graph_summary.csv`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_summary.csv`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid_summary.csv`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale_summary.csv`
- `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation_summary.csv`
- `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic_summary.csv`
- `results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_graph_summary.csv`
- `results/simulation_experiments/fed_robustness/fed_client_dropout_summary.csv`
- `results/simulation_experiments/fed_robustness/fed_communication_delay_summary.csv`
- `results/simulation_experiments/fed_robustness/fed_gradient_noise_summary.csv`

## 2. 已发现并修正的冲突

| 冲突项 | v1 表述 | 核验结果 | v2 修正 |
|---|---|---|---|
| 正式论文中 `Synthetic Experiments` 的位置 | 仅笼统说位于实验部分 | `main.tex` 显示其位于 `Data analysis` 下，并先于 `Real-World Data Analysis` | v2 按 `Data analysis` 内部子模块重写，并保持与真实数据分析衔接 |
| 论文主线中的聚合方式 | v1 与现有英文稿中混入 loss-aware / mixed aggregation 痕迹 | 约束文件明确规定主线必须是标准 FedAvg | v2 仅保留标准 FedAvg 公式与伪代码，排除混合加权聚合表述 |
| Non-IID 分层是否有 FedAvg 行 | 旧证据表曾写成有 FedAvg 数值 | `cnn_enhanced_noniid_summary.csv` 只有 `Proposed` 行 | v2 改为“缺少 FedAvg 行”，不再写分层定量主结论 |
| 客户端数量实验是否有 FedAvg 行 | 旧证据表写成 3/5/8 客户端下有 FedAvg 结果 | `cnn_enhanced_client_scale_summary.csv` 只有 `Proposed` 行 | v2 改为证据缺口，不进入主文 |
| 特征消融是否有 FedAvg 行 | 旧稿直接给出消融数值 | `cnn_enhanced_feature_ablation_summary.csv` 只有 `Proposed` 行 | v2 仅保留设计说明，不写 FedAvg 定量结论 |
| 基础实验数据划分比例 | v1 中出现过 70%/15%/15% | `cfb_core.py` 与 `gfb_core.py` 均为 70%/10%/20% | v2 统一改为 70%/10%/20% |
| 基础实验通信轮次 | v1 曾将基础实验笼统写为 10 轮 | CNN 主训练为 10 轮但收敛记录为 15 轮；GCN 为 10 轮 | v2 明确区分“CNN 收敛记录 15 轮，GCN 为 10 轮” |
| GCN 增强实验的统计力度 | 旧稿存在把动态图结果写得过强的风险 | `gfe_core.py` 明确 `SEEDS = [42]` | v2 将固定图与动态图比较限定为趋势性证据 |
| 梯度噪声表述 | 旧稿有被写成隐私机制的风险 | `fr_core.py` 只是参数扰动鲁棒性实验 | v2 明确改写为模拟梯度噪声扰动，不写成差分隐私 |
| 图像引用方式 | v1 未按 PNG/PDF 对应关系逐张核验 | 当前结果目录中大部分图同时存在 PNG/PDF，但部分图混入 `Proposed` 曲线 | v2 统一以 PNG 做 Markdown 预览，并在资产计划与审计中同步登记同名 PDF 与处理结论 |
| 含探索性聚合的图件 | v1 中部分图规划未剔除历史探索策略 | `cfe_visualization.py`、`gfe_visualization.py`、`fr_visualization.py` 当前图件会绘制 `Proposed` 等方法 | v2 将相关图件标记为“需重绘 FedAvg-only”或“暂不进入主文” |

## 3. 仍需作者确认的问题

1. 是否后续补充与 FedAvg 主线一致的 Non-IID 分层、客户端数量扩展和特征消融汇总结果；若补充，需同步更新正文、证据表和资产计划。
2. 是否在 LaTeX 正文改写阶段彻底删除 `main.tex` 中当前 `Synthetic Experiments` 段落内的混合加权聚合和损失加权表述。
3. 是否为增强 GCN 实验补充多种子结果；若不补充，则正式论文中只能维持趋势性表达。
4. 是否需要将部分动态图结构图放入补充材料，而不是主文。
5. 是否需要在真实数据实验部分增加一段与仿真实验对应的“受控验证到真实部署”的衔接句，以增强章节连贯性。

## 图片 PNG/PDF 对应关系核验

| 图号 | PNG 路径 | PDF 路径 | PNG 存在 | PDF 存在 | 处理结论 |
|---|---|---|---|---|---|
| 图 1 | `../../results/simulation_experiments/cnn_fed_base/main_metrics_comparison.png` | `../../results/simulation_experiments/cnn_fed_base/main_metrics_comparison.pdf` | 是 | 是 | 可进入主文 |
| 图 2 | `../../results/simulation_experiments/cnn_fed_base/convergence_curve.png` | `../../results/simulation_experiments/cnn_fed_base/convergence_curve.pdf` | 是 | 是 | 可进入主文 |
| 图 3 | `../../results/simulation_experiments/gcn_fed_base/convergence_curve.png` | `../../results/simulation_experiments/gcn_fed_base/convergence_curve.pdf` | 是 | 是 | 可进入主文 |
| 图 4 | `../../results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.png` | `../../results/simulation_experiments/gcn_fed_base/base_graph_adjacency_matrix.pdf` | 是 | 是 | 可进入主文 |
| 图 5 | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.png` | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_fixed_adjacency.pdf` | 是 | 是 | 可进入主文 |
| 图 6 | `../../results/simulation_experiments/fed_robustness/fed_robustness_client_dropout.png` | `../../results/simulation_experiments/fed_robustness/fed_robustness_client_dropout.pdf` | 是 | 是 | 需重绘 FedAvg-only |
| 图 7 | `../../results/simulation_experiments/fed_robustness/fed_robustness_communication_delay.png` | `../../results/simulation_experiments/fed_robustness/fed_robustness_communication_delay.pdf` | 是 | 是 | 需重绘 FedAvg-only |
| 图 8 | `../../results/simulation_experiments/fed_robustness/fed_robustness_gradient_noise.png` | `../../results/simulation_experiments/fed_robustness/fed_robustness_gradient_noise.pdf` | 是 | 是 | 需重绘 FedAvg-only |
| 图 9 | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic.png` | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic.pdf` | 是 | 是 | 需重绘 FedAvg-only |
| 图 10 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid.pdf` | 是 | 是 | 暂不进入主文 |
| 图 11 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale.pdf` | 是 | 是 | 暂不进入主文 |
| 图 12 | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation.png` | `../../results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation.pdf` | 是 | 是 | 暂不进入主文 |
| 图 13 | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_dynamic_peak.png` | `../../results/simulation_experiments/gcn_fed_enhanced_experiments/enhanced_gcn_dynamic_peak.pdf` | 是 | 是 | 仅作补充材料 |

## 4. 不进入主文的材料

- `cnn_enhanced_noniid_summary.csv` 中的低/中/高 Non-IID 分层数值，因为当前只有 `Proposed` 行。
- `cnn_enhanced_client_scale_summary.csv` 中的 3/5/8 客户端数量结论，因为当前只有 `Proposed` 行。
- `cnn_enhanced_feature_ablation_summary.csv` 中的特征消融数值，因为当前只有 `Proposed` 行。
- `cnn_enhanced_aggregation_summary.csv` 与 `gcn_enhanced_aggregation_summary.csv` 中的 `Proposed`、`Loss-weighted`、`Data-loss weighted` 聚合比较。
- 现有鲁棒性图与 GCN 固定图/动态图对比图的原始版本，因为当前图件混入探索性聚合曲线。

## 5. 后续转 LaTeX 风险提醒

1. `main.tex` 当前 `Synthetic Experiments` 段落仍保留与标准 FedAvg 不一致的旧聚合表述，后续转写时必须显式替换。
2. 若直接复用现有 PNG/PDF 图件而不筛除 `Proposed` 曲线，极易在 LaTeX 阶段重新引入与主线冲突的视觉证据。
3. 若后续作者补充 FedAvg 版 Non-IID、客户端数量或消融结果，必须同步更新证据表、资产计划和正文，不能只改正文数值。
4. GCN 增强实验目前是单种子结果，LaTeX 转写时必须继续保持“趋势性证据”措辞，避免放大为统计结论。
5. 鲁棒性中的梯度噪声只能写成模拟扰动；若在英文稿中误写为 privacy-preserving noise 或 differential privacy，将与当前代码证据冲突。
