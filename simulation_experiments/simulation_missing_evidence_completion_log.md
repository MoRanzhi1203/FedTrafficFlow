# Simulation Missing Evidence Completion Log / 仿真实验缺失证据补全日志

## 1. Modified Files / 已修改文件

| File / 文件 | Change / 修改内容 | Core Logic Changed? / 是否改变核心逻辑 |
|---|---|---|
| `simulation_experiments/cnn_fed_enhanced_experiments/cfe_core.py` | Added `--run-fedavg-missing` entry and exported FedAvg-only CSV for non-IID, client-scale, and feature-ablation experiments while preserving historical files. / 新增 `--run-fedavg-missing` 入口，补齐 Non-IID、客户端数量、特征消融三类实验的 FedAvg-only CSV，并保留历史结果文件。 | No; standard FedAvg aggregation is reused without changing the aggregation rule. / 否；复用现有标准 FedAvg 聚合，未改变聚合逻辑。 |
| `simulation_experiments/cnn_fed_enhanced_experiments/cfe_visualization.py` | Added `--paper-ready` mode to generate FedAvg-only PNG/PDF figures from paper-ready CSV. / 新增 `--paper-ready` 模式，从 paper-ready CSV 生成 FedAvg-only PNG/PDF 图。 | No / 否 |
| `simulation_experiments/fed_robustness_experiments/fr_visualization.py` | Added `--paper-ready` mode and generated FedAvg-only robustness figures; gradient noise figure includes non-DP note. / 新增 `--paper-ready` 模式，生成 FedAvg-only 鲁棒性图；梯度噪声图加入非 DP 说明。 | No / 否 |
| `simulation_experiments/gcn_fed_enhanced_experiments/gfe_visualization.py` | Added `--paper-ready` mode for FedAvg-only fixed-vs-dynamic graph figure and single-seed trend warning. / 新增 `--paper-ready` 模式，生成 FedAvg-only 固定图/动态图对比图，并输出单种子趋势性提示。 | No / 否 |

## 2. Generated FedAvg-only CSV / 生成的 FedAvg-only CSV

| File / 文件 | Rows / 行数 | Source Experiment / 来源实验 |
|---|---|---|
| `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.csv` | 3 | CNN enhanced non-IID sensitivity / CNN 增强 Non-IID 分层实验 |
| `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.csv` | 3 | CNN enhanced client-scale sensitivity / CNN 增强客户端数量扩展实验 |
| `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.csv` | 5 | CNN enhanced feature ablation / CNN 增强特征消融实验 |

## 3. Generated Paper-ready Figures / 生成的论文就绪图

| Figure / 图 | PNG / 预览图 | PDF / 排版图 | FedAvg-only? / 是否仅 FedAvg |
|---|---|---|---|
| CNN enhanced non-IID / CNN 增强 Non-IID | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.png` | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.pdf` | Yes / 是 |
| CNN enhanced client scale / CNN 增强客户端数量 | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.png` | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.pdf` | Yes / 是 |
| CNN enhanced feature ablation / CNN 增强特征消融 | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.png` | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.pdf` | Yes / 是 |
| Robustness: client dropout / 鲁棒性：客户端掉线 | `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.png` | `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.pdf` | Yes / 是 |
| Robustness: communication delay / 鲁棒性：通信延迟 | `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_communication_delay_fedavg_only.png` | `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_communication_delay_fedavg_only.pdf` | Yes / 是 |
| Robustness: gradient perturbation / 鲁棒性：梯度扰动 | `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_gradient_noise_fedavg_only.png` | `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_gradient_noise_fedavg_only.pdf` | Yes / 是 |
| GCN fixed vs dynamic / GCN 固定图与动态图 | `results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.png` | `results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.pdf` | Yes / 是 |

## 4. Verification / 验证

| Check / 检查项 | Result / 结果 |
|---|---|
| `py_compile` for modified scripts / 已修改脚本通过 `py_compile` | Passed / 通过 |
| LaTeX files modified? / 是否修改 LaTeX | No / 否 |
| Base experiment core modified? / 是否修改基础实验 core | No / 否 |
| Real-world experiments rerun? / 是否重跑真实数据实验 | No / 否 |
| FedAvg-only CSV generated? / 是否生成 FedAvg-only CSV | Yes / 是 |
| Paper-ready PNG generated? / 是否生成 paper-ready PNG | Yes / 是 |
| Matching PDF generated for each paper-ready PNG? / 每张 paper-ready PNG 是否有同名 PDF | Yes / 是 |
| Paper-ready figures contain only FedAvg? / paper-ready 图是否仅含 FedAvg | Yes, by FedAvg filtering in visualization and FedAvg-only CSV input. / 是，通过可视化过滤和 FedAvg-only CSV 保证。 |
| Proposed mixed into paper-ready figures? / paper-ready 图中是否混入 Proposed | No / 否 |
| Gradient noise figure includes non-DP note? / 梯度噪声图是否包含非 DP 说明 | Yes / 是 |
| New runner/pipeline/config/utils added? / 是否新增 runner/pipeline/config/utils | No / 否 |
| FedAvg aggregation changed? / 是否改变 FedAvg 聚合方式 | No / 否 |

## 5. Remaining Issues / 剩余问题

- The original mixed-method summary CSV files are preserved and still contain historical `Proposed` rows; only the new FedAvg-only CSV and paper-ready figures should be used for manuscript-ready assets. / 原始混合方法汇总 CSV 仍保留历史 `Proposed` 行；正式论文资产应只使用新生成的 FedAvg-only CSV 与 paper-ready 图。
- The GCN fixed-vs-dynamic result remains single-seed and should only be interpreted as trend evidence. / GCN 固定图与动态图结果仍为单种子，只能作为趋势性证据。
- If the manuscript v2 text needs to quote the newly generated FedAvg-only values, update the manuscript-facing markdown files in a later documentation phase rather than in this code-and-output phase. / 若后续需要在正式稿中引用本次新生成的 FedAvg-only 数值，应在后续文档阶段更新相关 markdown，而不是在当前代码与结果阶段直接改正文。
