# Simulation Experiment Report / 仿真实验报告

## 1. Simulation Experiment Code Structure / 仿真实验代码结构

**English:**
The project contains 5 simulation experiment groups, each with a core logic file and a visualization file. CNN Fed Base and GCN Fed Base are clean FedAvg-only mainline experiments. The three "enhanced" or "robustness" experiments mix FedAvg with historical exploration strategies (Loss-weighted, Data-loss weighted, Proposed).

**中文：**
项目包含 5 组仿真实验，每组包含一个核心逻辑文件和一个可视化文件。CNN Fed Base 和 GCN Fed Base 是纯粹的 FedAvg 主线实验。三个"增强"或"鲁棒性"实验混合了 FedAvg 与历史探索策略（Loss-weighted、Data-loss weighted、Proposed）。

| # | Experiment Group / 实验组 | Core File / 核心文件 | Visualization File / 可视化文件 | Status / 状态 |
|---|---|---|---|---|
| 1 | CNN Fed Base / CNN联邦基础 | `simulation_experiments/cnn_fed_base/cfb_core.py` | `simulation_experiments/cnn_fed_base/cfb_visualization.py` | Active - FedAvg Mainline / FedAvg主线 |
| 2 | CNN Fed Enhanced / CNN联邦增强 | `simulation_experiments/cnn_fed_enhanced_experiments/cfe_core.py` | `simulation_experiments/cnn_fed_enhanced_experiments/cfe_visualization.py` | Active - Mixed (FedAvg + Historical Exploration) / 混合(FedAvg + 历史探索) |
| 3 | Fed Robustness / 联邦鲁棒性 | `simulation_experiments/fed_robustness_experiments/fr_core.py` | `simulation_experiments/fed_robustness_experiments/fr_visualization.py` | Active - FedAvg Mainline Support / FedAvg主线支撑 |
| 4 | GCN Fed Base / GCN联邦基础 | `simulation_experiments/gcn_fed_base/gfb_core.py` | `simulation_experiments/gcn_fed_base/gfb_visualization.py` | Active - FedAvg Mainline / FedAvg主线 |
| 5 | GCN Fed Enhanced / GCN联邦增强 | `simulation_experiments/gcn_fed_enhanced_experiments/gfe_core.py` | `simulation_experiments/gcn_fed_enhanced_experiments/gfe_visualization.py` | Active - Mixed (FedAvg + Historical Exploration) / 混合(FedAvg + 历史探索) |

## 2. Core Logic Analysis / 核心逻辑分析

**English:**
Core logic across all 5 groups is consistent: data generation → model (CNN/GCN-BiLSTM-Attention) → training (FedAvg/Independent) → metrics export. The enhanced groups additionally include multi-strategy aggregation (Loss-weighted, Data-loss weighted, Proposed) which are historical explorations. The robustness group simulates communication cost, client dropout, delay, and gradient noise — but the gradient noise is simulated perturbation, NOT formal differential privacy.

**中文：**
五组实验的核心逻辑一致：数据生成 → 模型（CNN/GCN-BiLSTM-Attention）→ 训练（FedAvg/Independent）→ 指标导出。增强实验额外包含多策略聚合（Loss-weighted、Data-loss weighted、Proposed），这些均为历史探索。鲁棒性实验模拟了通信开销、客户端掉线、通信延迟和梯度噪声——但梯度噪声仅为模拟扰动，并非正式差分隐私。

| Experiment Group / 实验组 | Core Functions / 核心函数 | Aggregation Strategies / 聚合策略 | FedAvg Mainline? / FedAvg主线? | Risk Flags / 风险标记 |
|---|---|---|---|---|
| CNN Fed Base / CNN联邦基础 | 数据生成, CNN-BiLSTM-Attention模型, FedAvg/Independent训练, 指标导出 | FedAvg, Independent | **YES / 是** - Primary mainline experiment / 主要主线实验 | None / 无 |
| CNN Fed Enhanced / CNN联邦增强 | 数据生成, CNN-BiLSTM-Attention模型, 多策略聚合, 非IID, 消融, 3种子 | FedAvg, Loss-weighted, Data-loss weighted, Proposed, Independent | **PARTIAL / 部分** - Only FedAvg results usable for mainline / 仅FedAvg结果可用于主线 | ⚠️ Loss-weighted, Data-loss weighted, Proposed coded as main comparison / 被编码为主要对比 |
| Fed Robustness / 联邦鲁棒性 | 通信开销, 客户端掉线, 通信延迟, 梯度噪声 (基于CNN) | FedAvg, Proposed | **PARTIAL / 部分** - Only FedAvg results usable; noise NOT formal DP / 仅FedAvg可用; 噪声非正式差分隐私 | ⚠️ Proposed in METHOD_PALETTE; DP noise simulated but NOT formal differential privacy / Proposed在调色板中; 梯度噪声模拟但非正式差分隐私 |
| GCN Fed Base / GCN联邦基础 | 数据生成, GCN-BiLSTM-Attention模型, FedAvg/Independent训练, 邻接矩阵导出 | FedAvg, Independent | **YES / 是** - Primary mainline comparison / 主要主线对比 | None / 无 |
| GCN Fed Enhanced / GCN联邦增强 | 数据生成, GCN-BiLSTM-Attention模型, 多策略聚合, 动态图, 拥塞建模, 1种子 | FedAvg, Loss-weighted, Data-loss weighted, Proposed, Independent | **PARTIAL / 部分** - Only FedAvg results usable for mainline / 仅FedAvg结果可用于主线 | ⚠️ Loss-weighted, Data-loss weighted, Proposed coded as main comparison / 被编码为主要对比 |

## 3. Visualization Logic Analysis / 可视化逻辑分析

**English:**
Five visualization scripts produce 7–10+ plots each. Key issues: (1) `cfb_visualization.py` has no PDF output, (2) three enhanced/robustness scripts plot Loss-weighted/Data-loss weighted/Proposed as equals to FedAvg — these should be separated into paper-ready (FedAvg-only) and supplementary (full comparison) outputs.

**中文：**
五个可视化脚本各生成 7–10+ 张图表。关键问题：(1) `cfb_visualization.py` 无 PDF 输出，(2) 三个增强/鲁棒性脚本将 Loss-weighted/Data-loss weighted/Proposed 与 FedAvg 等同绘制——应分离为论文就绪版（仅FedAvg）和补充材料版（完整对比）。

| Visualization File / 可视化文件 | # Plots / 图表数 | Input CSV Dependencies / 输入CSV依赖 | PDF Saved? / PDF保存? | Style / 样式 | Issues / 问题 |
|---|---|---|---|---|---|
| `cfb_visualization.py` | 7 | base_dataset_*.csv, main_metrics.csv, main_predictions.csv, convergence_history.csv, main_summary.csv | No (PNG only / 仅PNG) | seaborn whitegrid, 300 DPI | No PDF output / 无PDF输出; `METHOD_PALETTE` includes "Proposed" but only FedAvg/Independent in data / 调色板含Proposed但数据仅FedAvg/Independent |
| `cfe_visualization.py` | 7 | enhanced_dataset_*.csv, cnn_enhanced_aggregation_metrics.csv, cnn_enhanced_ablation_metrics.csv (if exists/如存在), cnn_enhanced_non_iid_summary.csv (if exists/如存在) | Yes (PDF) / 是 | seaborn whitegrid, 300 DPI | ⚠️ Plots include Loss-weighted/Data-loss weighted/Proposed as equals to FedAvg / 图表将三者与FedAvg等同绘制 |
| `fr_visualization.py` | 4 | fed_communication_cost.csv, fed_client_dropout_summary.csv, fed_communication_delay_summary.csv, fed_gradient_noise_summary.csv | Yes (PDF) / 是 | seaborn whitegrid, 300 DPI | ⚠️ Plots show FedAvg vs Proposed as if both were main methods / 图表将Proposed视为主方法 |
| `gfb_visualization.py` | 8 | base_dataset_*.csv, base_graph_*.csv, main_metrics.csv, main_predictions.csv, convergence_history.csv, main_summary.csv | Yes (PDF) / 是 | seaborn whitegrid, 300 DPI | Clean / 干净 - only FedAvg/Independent plotted / 仅绘制FedAvg/Independent |
| `gfe_visualization.py` | 10+ | enhanced_dataset_*.csv, gcn_enhanced_aggregation_metrics.csv, enhanced_graph_*.csv, congestion_*.csv, dynamic_graph_*.csv | Yes (PDF) / 是 | seaborn whitegrid, 300 DPI | ⚠️ Plots include Loss-weighted/Data-loss weighted/Proposed as equals to FedAvg / 图表将三者与FedAvg等同绘制 |

## 4. Results File Index / 结果文件索引

**English:**
Results are stored under `results/simulation_experiments/` in 5 subdirectories. Each subdirectory contains CSV data files and PNG/PDF plot files generated by the corresponding core and visualization scripts.

**中文：**
结果存储在 `results/simulation_experiments/` 下的 5 个子目录中。每个子目录包含由对应核心脚本和可视化脚本生成的 CSV 数据文件和 PNG/PDF 图表文件。

### 4.1 CNN Fed Base / CNN联邦基础 (`results/simulation_experiments/cnn_fed_base/`)
| File / 文件 | Type / 类型 | Source / 来源 |
|---|---|---|
| base_dataset_client_timeseries.csv/png | Data + Plot / 数据+图表 | cfb_core → cfb_visualization |
| base_dataset_node_heatmap.csv/png | Data + Plot / 数据+图表 | cfb_core → cfb_visualization |
| base_dataset_split_overview.csv/png | Data + Plot / 数据+图表 | cfb_core → cfb_visualization |
| base_dataset_client_boxplot.png | Plot / 图表 | cfb_visualization |
| base_dataset_client_distribution.csv | Data / 数据 | cfb_core |
| base_dataset_client_sample_size.csv/png | Data + Plot / 数据+图表 | cfb_core → cfb_visualization |
| base_dataset_summary.csv | Data / 数据 | cfb_core |
| main_metrics.csv | Data / 数据 | cfb_core |
| main_metrics_comparison.png | Plot / 图表 | cfb_visualization |
| main_predictions.csv | Data / 数据 | cfb_core |
| main_predictions_comparison.png | Plot / 图表 | cfb_visualization |
| main_summary.csv | Data / 数据 | cfb_core |
| convergence_history.csv | Data / 数据 | cfb_core |
| convergence_curve.png | Plot / 图表 | cfb_visualization |

### 4.2 CNN Fed Enhanced / CNN联邦增强 (`results/simulation_experiments/cnn_fed_enhanced_experiments/`)
| File / 文件 | Type / 类型 | Source / 来源 |
|---|---|---|
| enhanced_dataset_client_timeseries.csv/png/pdf | CSV + Plot / 数据+图表 | cfe_core → cfe_visualization |
| enhanced_dataset_node_heatmap.csv/png/pdf | CSV + Plot / 数据+图表 | cfe_core → cfe_visualization |
| enhanced_dataset_split_overview.csv/png/pdf | CSV + Plot / 数据+图表 | cfe_core → cfe_visualization |
| enhanced_dataset_client_boxplot.png/pdf | Plot / 图表 | cfe_visualization |
| enhanced_dataset_client_distribution.csv | Data / 数据 | cfe_core |
| enhanced_dataset_client_sample_size.csv/png/pdf | Data + Plot / 数据+图表 | cfe_core → cfe_visualization |
| enhanced_dataset_summary.csv | Data / 数据 | cfe_core |
| cnn_enhanced_aggregation_metrics.csv | Data (3 seeds / 3种子) | cfe_core |
| cnn_enhanced_aggregation_comparison.png/pdf | Plot / 图表 | cfe_visualization |
| cnn_enhanced_aggregation_convergence.csv | Data / 数据 | cfe_core |
| cnn_enhanced_aggregation_convergence.png/pdf | Plot / 图表 | cfe_visualization |
| cnn_enhanced_ablation_metrics.csv (if exists / 如存在) | Data / 数据 | cfe_core |
| cnn_enhanced_non_iid_summary.csv (if exists / 如存在) | Data / 数据 | cfe_core |

### 4.3 Fed Robustness / 联邦鲁棒性 (`results/simulation_experiments/fed_robustness/`)
| File / 文件 | Type / 类型 | Source / 来源 |
|---|---|---|
| fed_communication_cost.csv | Data / 数据 | fr_core |
| fed_robustness_communication_cost.png/pdf | Plot / 图表 | fr_visualization |
| fed_client_dropout_metrics.csv | Data / 数据 | fr_core |
| fed_client_dropout_summary.csv | Data / 数据 | fr_core |
| fed_robustness_client_dropout.png/pdf | Plot / 图表 | fr_visualization |
| fed_communication_delay_metrics.csv | Data / 数据 | fr_core |
| fed_communication_delay_summary.csv | Data / 数据 | fr_core |
| fed_robustness_communication_delay.png/pdf | Plot / 图表 | fr_visualization |
| fed_gradient_noise_metrics.csv | Data / 数据 | fr_core |
| fed_gradient_noise_summary.csv | Data / 数据 | fr_core |
| fed_robustness_gradient_noise.png/pdf | Plot / 图表 | fr_visualization |

### 4.4 GCN Fed Base / GCN联邦基础 (`results/simulation_experiments/gcn_fed_base/`)
| File / 文件 | Type / 类型 | Source / 来源 |
|---|---|---|
| base_dataset_client_timeseries.csv/png/pdf | Data + Plot / 数据+图表 | gfb_core → gfb_visualization |
| base_dataset_node_heatmap.csv/png/pdf | Data + Plot / 数据+图表 | gfb_core → gfb_visualization |
| base_dataset_split_overview.csv/png/pdf | Data + Plot / 数据+图表 | gfb_core → gfb_visualization |
| base_dataset_client_boxplot.png/pdf | Plot / 图表 | gfb_visualization |
| base_dataset_client_distribution.csv | Data / 数据 | gfb_core |
| base_dataset_client_sample_size.csv/png/pdf | Data + Plot / 数据+图表 | gfb_core → gfb_visualization |
| base_dataset_summary.csv | Data / 数据 | gfb_core |
| base_graph_adjacency_matrix.csv/png/pdf | Data + Plot / 数据+图表 | gfb_core → gfb_visualization |
| base_graph_summary.csv | Data / 数据 | gfb_core |
| main_metrics.csv | Data / 数据 | gfb_core |
| main_metrics_comparison.png/pdf | Plot / 图表 | gfb_visualization |
| main_predictions.csv | Data / 数据 | gfb_core |
| main_predictions_comparison.png/pdf | Plot / 图表 | gfb_visualization |
| main_summary.csv | Data / 数据 | gfb_core |
| convergence_history.csv | Data / 数据 | gfb_core |
| convergence_curve.png/pdf | Plot / 图表 | gfb_visualization |

### 4.5 GCN Fed Enhanced / GCN联邦增强 (`results/simulation_experiments/gcn_fed_enhanced_experiments/`)
| File / 文件 | Type / 类型 | Source / 来源 |
|---|---|---|
| enhanced_dataset_client_timeseries.csv/png/pdf | Data + Plot / 数据+图表 | gfe_core → gfe_visualization |
| enhanced_dataset_client_distribution.csv | Data / 数据 | gfe_core |
| enhanced_dataset_client_sample_size.csv/png/pdf | Data + Plot / 数据+图表 | gfe_core → gfe_visualization |
| enhanced_dataset_node_heatmap.csv/png/pdf | Data + Plot / 数据+图表 | gfe_core → gfe_visualization |
| enhanced_dataset_split_overview.csv/png/pdf | Data + Plot / 数据+图表 | gfe_core → gfe_visualization |
| enhanced_dataset_summary.csv | Data / 数据 | gfe_core |
| enhanced_graph_adjacency_matrix.csv/png/pdf | Data + Plot / 数据+图表 | gfe_core → gfe_visualization |
| enhanced_graph_summary.csv | Data / 数据 | gfe_core |
| gcn_enhanced_aggregation_metrics.csv | Data (1 seed / 1种子) | gfe_core |
| gcn_enhanced_aggregation_comparison.png/pdf | Plot / 图表 | gfe_visualization |
| gcn_enhanced_aggregation_convergence.csv | Data / 数据 | gfe_core |
| gcn_enhanced_aggregation_convergence.png/pdf | Plot / 图表 | gfe_visualization |
| congestion_analysis.csv (if exists / 如存在) | Data / 数据 | gfe_core |
| dynamic_graph_metrics.csv (if exists / 如存在) | Data / 数据 | gfe_core |

## 5. FedAvg Mainline Usable Results / FedAvg主线可用结果

**English:**
Only FedAvg and Independent results enter the main paper. Loss-weighted, Data-loss weighted, and Proposed are historical explorations that may appear in supplementary or future work only.

**中文：**
仅 FedAvg 和 Independent 结果进入论文主文。Loss-weighted、Data-loss weighted 和 Proposed 为历史探索，仅可出现在补充材料或未来工作中。

| # | Experiment Group / 实验组 | Usable Content / 可用内容 | For Paper Section / 论文章节 | Notes / 备注 |
|---|---|---|---|---|
| 1 | CNN Fed Base / CNN联邦基础 | All FedAvg vs Independent results / 全部FedAvg vs Independent结果 | Experiments - Main Comparison / 实验-主对比 | **Ready / 就绪** - Clean FedAvg mainline / 干净FedAvg主线 |
| 2 | CNN Fed Enhanced / CNN联邦增强 | FedAvg rows only in aggregation metrics; ablation results (if only FedAvg); non-IID analysis (if FedAvg-only) / 聚合指标中的FedAvg行; 消融结果(仅FedAvg); 非IID分析(仅FedAvg) | Experiments - Non-IID Analysis / 实验-非IID分析 | ⚠️ Needs extraction: filter only FedAvg rows from multi-strategy data / 需提取: 从多策略数据中仅过滤FedAvg行 |
| 3 | Fed Robustness / 联邦鲁棒性 | Communication cost, dropout, delay, noise results **for FedAvg only** / 通信开销、掉线、延迟、噪声结果**仅FedAvg** | Experiments - Robustness / 实验-鲁棒性 | ⚠️ FedAvg rows valid; Proposed rows must be excluded from main paper / FedAvg行有效; Proposed行须从主论文排除 |
| 4 | GCN Fed Base / GCN联邦基础 | All FedAvg vs Independent results / 全部FedAvg vs Independent结果 | Experiments - Model Comparison / 实验-模型对比 | **Ready / 就绪** - Clean FedAvg mainline / 干净FedAvg主线 |
| 5 | GCN Fed Enhanced / GCN联邦增强 | FedAvg rows only; congestion and dynamic graph analysis (if model-agnostic) / 仅FedAvg行; 拥塞和动态图分析(如与模型无关) | Experiments - GCN Analysis / 实验-GCN分析 | ⚠️ Needs extraction; 1-seed only / 需提取; 仅1种子 |

## 6. Historical Exploration Experiment Handling / 历史探索实验处理

**English:**
Non-FedAvg aggregation strategies (Loss-weighted, Data-loss weighted, Proposed) must be clearly marked as historical exploration and excluded from the main paper. They may be retained in supplementary materials or discussed as future work only.

**中文：**
非FedAvg聚合策略（Loss-weighted、Data-loss weighted、Proposed）须明确标记为历史探索，不得进入论文主文。可保留在补充材料中或仅作为未来工作讨论。

| Content / 内容 | Source File(s) / 源文件 | Current Role / 当前角色 | Required Disposition / 要求处理 |
|---|---|---|---|
| Loss-weighted aggregation results / Loss-weighted聚合结果 | `cfe_core.py`, `gfe_core.py` | Plotted alongside FedAvg / 与FedAvg并列绘制 | Mark as "Historical Exploration" in report; do NOT enter main paper / 标记为历史探索; 不得进入主论文 |
| Data-loss weighted aggregation results / Data-loss weighted聚合结果 | `cfe_core.py`, `gfe_core.py` | Plotted alongside FedAvg / 与FedAvg并列绘制 | Mark as "Historical Exploration"; do NOT enter main paper / 标记为历史探索; 不得进入主论文 |
| Proposed aggregation results / Proposed聚合结果 | `cfe_core.py`, `gfe_core.py`, `fr_core.py` | Plotted as equal method / 作为对等方法绘制 | Mark as "Historical Exploration / Abandoned"; do NOT enter main paper / 标记为历史探索/已放弃; 不得进入主论文 |
| Proposed in robustness plots / 鲁棒性图中的Proposed | `fr_core.py`, `fr_visualization.py` | Compared vs FedAvg / 与FedAvg对比 | Strip from paper-ready charts; keep in supplementary folder only / 从论文就绪图表移除; 仅保留在补充材料文件夹 |

## 7. Chart Quality Checklist / 图表质量检查清单

**English:**
Chart quality audit across all 5 groups. Key findings: CNN Fed Base lacks PDF output; three enhanced/robustness groups show "Proposed" extensively; colorblind-friendliness needs improvement; font sizes are inconsistent across groups.

**中文：**
五组实验的图表质量审核。关键发现：CNN Fed Base 缺少 PDF 输出；三个增强/鲁棒性实验组大量显示 "Proposed"；色盲友好性需改进；各组字体大小不统一。

| Criterion / 标准 | CNN Fed Base | CNN Fed Enhanced | Fed Robustness | GCN Fed Base | GCN Fed Enhanced |
|---|---|---|---|---|---|
| DPI ≥ 300 | ✅ 300 | ✅ 300 | ✅ 300 | ✅ 300 | ✅ 300 |
| PDF vector output / PDF矢量输出 | ❌ PNG only / 仅PNG | ✅ | ✅ | ✅ | ✅ |
| Colorblind-friendly palette / 色盲友好调色板 | ⚠️ tab10 good but not CB-optimized / tab10好但非CB优化 | ⚠️ tab10 + custom / tab10+自定义 | ⚠️ tab10 | ⚠️ tab10 | ⚠️ tab10 |
| Readable font size (≥ 10pt) / 可读字体(≥10pt) | ✅ font_scale=1.2 | ✅ font_scale=1.15 | ✅ font_scale=1.15 | ✅ font_scale=1.15 | ✅ font_scale=1.15 |
| Consistent style across groups / 跨组样式一致 | ⚠️ font_scale differs (1.2 vs 1.15) / font_scale不一致 | ⚠️ font_scale differs / 不一致 | ⚠️ font_scale differs / 不一致 | ✅ | ✅ |
| Title present on all charts / 所有图表有标题 | ✅ | ✅ | ✅ | ✅ | ✅ |
| Legend not obscuring data / 图例不遮挡数据 | ⚠️ ncol=2, fontsize=8 (small/小) | 🟡 default ncol / 默认 | 🟡 default ncol / 默认 | ⚠️ ncol=2, fontsize=8 (small/小) | 🟡 default ncol / 默认 |
| Axis labels present / 坐标轴标签存在 | ✅ | ✅ | ✅ | ✅ | ✅ |
| No "Proposed" in mainline charts / 主线图表不含Proposed | ⚠️ In palette but not used in data / 在调色板中但数据中未使用 | ❌ Used extensively / 大量使用 | ❌ Used extensively / 大量使用 | ✅ Clean / 干净 | ❌ Used extensively / 大量使用 |
| No DP/gradient noise mislabeled as formal DP / 梯度噪声未被误标为正式差分隐私 | N/A | N/A | ⚠️ "Gradient Noise" without DP disclaimer / "梯度噪声"无差分隐私声明 | N/A | N/A |

## 8. Charts Requiring Optimization / 需要优化的图表

**English:**
Priority-ranked list of charts requiring changes before paper submission. HIGH priority items relate to "Proposed" appearing in mainline charts and missing PDF output. MEDIUM items relate to N-IID conflation and DP disclaimer. LOW items are style consistency issues.

**中文：**
按优先级排列的论文提交前需要修改的图表清单。高优先级项涉及 "Proposed" 出现在主线图表和缺失 PDF 输出。中优先级项涉及非IID混用和差分隐私声明。低优先级为样式一致性问题。

| Priority / 优先级 | Chart / 图表 | Current Issue / 当前问题 | Required Action / 要求操作 |
|---|---|---|---|
| **HIGH / 高** | CNN Enhanced Aggregation Comparison / CNN增强聚合对比 | Shows Loss-weighted/Data-loss weighted/Proposed as equals to FedAvg / 显示三者与FedAvg并列 | Split: FedAvg-only chart for paper; full comparison chart marked as supplementary / 拆分: 仅FedAvg版用于论文; 完整对比版标记为补充材料 |
| **HIGH / 高** | GCN Enhanced Aggregation Comparison / GCN增强聚合对比 | Shows Loss-weighted/Data-loss weighted/Proposed as equals to FedAvg / 显示三者与FedAvg并列 | Split: FedAvg-only chart for paper; full comparison chart marked as supplementary / 拆分: 仅FedAvg版用于论文; 完整对比版标记为补充材料 |
| **HIGH / 高** | Fed Robustness All 4 Charts / 联邦鲁棒性全部4图 | Shows Proposed alongside FedAvg in METHOD_PALETTE and plots / 调色板和图表中显示Proposed与FedAvg并列 | Re-plot FedAvg-only versions for paper / 为论文重新绘制仅FedAvg版本 |
| **HIGH / 高** | CNN Fed Base All Charts / CNN联邦基础全部图表 | No PDF output / 无PDF输出 | Add PDF export to _save equivalent / 添加PDF导出 |
| **MEDIUM / 中** | All Non-IID Distribution Charts / 所有非IID分布图 | May conflate FedAvg with custom aggregation / 可能混淆FedAvg与自定义聚合 | Ensure non-IID analysis uses only FedAvg or Independent / 确保非IID分析仅使用FedAvg或Independent |
| **MEDIUM / 中** | Gradient Noise Chart / 梯度噪声图 | Risk of being misread as formal DP / 可能被误读为正式差分隐私 | Add explicit note: "simulated gradient noise, NOT formal differential privacy" / 添加明确注释:"模拟梯度噪声, 非正式差分隐私" |
| **LOW / 低** | All Convergence Curves / 所有收敛曲线 | Font size inconsistency (1.2 vs 1.15) / 字体大小不一致 | Standardize to font_scale=1.15 / 统一为font_scale=1.15 |
| **LOW / 低** | Base Dataset Charts (both CNN & GCN) / 基础数据集图表(CNN和GCN) | Legend font too small (8pt) / 图例字体过小(8pt) | Increase to ≥ 10pt / 增大到≥10pt |

## 9. Core Modification Judgment / 核心修改判断

**English:**
No core files need modification. The FedAvg implementation is standard and correct. Historical strategies should be kept in the codebase for reference but excluded from paper content. All CSV outputs are consistent with visualization inputs.

**中文：**
无需修改任何核心文件。FedAvg 实现为标准实现且正确。历史策略保留在代码库中以供参考，但从论文内容中排除。所有 CSV 输出与可视化输入一致。

| Question / 问题 | Answer / 回答 |
|---|---|
| Need to modify any core file? / 是否需要修改核心文件？ | **NO / 否** - Core logic is correct for FedAvg / 核心逻辑对FedAvg正确 |
| Need to change aggregation logic? / 是否需要修改聚合逻辑？ | **NO / 否** - FedAvg implementation is standard / FedAvg实现为标准实现 |
| Need to remove historical strategies? / 是否需要删除历史策略？ | **NO / 否** - Keep for reference; just don't use in paper / 保留以供参考; 论文中不使用 |
| Need to add new experiments? / 是否需要新增实验？ | **NO / 否** - Not in this phase / 本阶段不需要 |
| Need to fix data output paths? / 是否需要修复数据输出路径？ | **NO / 否** - Paths are consistent / 路径一致 |
| Core output matches visualization input? / 核心输出与可视化输入是否匹配？ | **YES / 是** - All CSV outputs match visualization inputs / 所有CSV输出匹配可视化输入 |

## 10. Follow-up Modification Recommendations / 后续修改建议

**English:**
1. **Visualization Phase (Next)**: Add PDF to CNN Fed Base, create FedAvg-only paper-ready charts, add historical exploration disclaimers, standardize fonts, adopt colorblind-friendly palette.
2. **Data Extraction Phase**: Create filtered CSV snapshots containing only FedAvg rows.
3. **DP Clarification**: Add comment in `fr_core.py` and `fr_visualization.py` clarifying gradient noise is simulated, not formal DP.
4. **Do NOT**: Change directory structure, merge experiment groups, delete historical data, or modify core training logic.

**中文：**
1. **可视化阶段（下一步）**：为 CNN Fed Base 添加 PDF，创建仅 FedAvg 的论文就绪图表，添加历史探索声明，统一字体，采用色盲友好调色板。
2. **数据提取阶段**：创建仅含 FedAvg 行的过滤 CSV 快照。
3. **差分隐私澄清**：在 `fr_core.py` 和 `fr_visualization.py` 中添加注释，说明梯度噪声为模拟扰动而非正式差分隐私。
4. **禁止**：改变目录结构、合并实验组、删除历史数据、或修改核心训练逻辑。

---

**Status / 状态：PHASE 1 COMPLETE - Ready for Visualization Optimization Phase / 第一阶段完成 - 准备进入可视化优化阶段**

**English:**
*Generated: Phase 1 Simulation Experiment Scan & Analysis*
*No core files modified. No visualization files modified. No experiments re-run.*

**中文：**
*生成：第一阶段 仿真实验扫描与分析*
*未修改核心文件。未修改可视化文件。未重新运行实验。*