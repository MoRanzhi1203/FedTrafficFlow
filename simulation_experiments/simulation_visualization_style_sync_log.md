# Simulation Visualization Style Sync Log / 仿真实验可视化风格同步日志

## 1. Scope / 范围

**English:**
This round only synchronizes the visual style of the newly added FedAvg-only paper-ready figures so that they better match the existing simulation figures already marked as manuscript-ready. No core training logic is modified, no CSV is regenerated, and no experiment is rerun.

**中文：**
本轮仅对新增的 FedAvg-only paper-ready 图表进行风格同步，使其更接近已有已标记为可直接进入主文的仿真实验图表。不修改 core 训练逻辑，不重新生成 CSV，不重跑实验。

## 2. Modified Files / 已修改文件

| File / 文件 | Change / 修改内容 | Core Logic Changed? / 是否改变核心逻辑 |
|---|---|---|
| `simulation_experiments/cnn_fed_enhanced_experiments/cfe_visualization.py` | Unified paper-ready style, switched the client-scale figure to the same bar-chart language as the other CNN FedAvg-only figures, added value annotations, synchronized spacing and grid treatment. / 统一 paper-ready 风格，将客户端数量图改为与另外两张 CNN 图一致的柱状图语言，补充数值标注并统一边距与网格样式。 | No / 否 |
| `simulation_experiments/fed_robustness_experiments/fr_visualization.py` | Unified robustness figures to a consistent line-chart style, standardized markers and linewidth, and repositioned the simulated-perturbation note to avoid curve overlap. / 将三张鲁棒性图统一为同一折线风格，统一 marker 与线宽，并调整模拟扰动说明位置以避免遮挡曲线。 | No / 否 |
| `simulation_experiments/gcn_fed_enhanced_experiments/gfe_visualization.py` | Synchronized the fixed-vs-dynamic FedAvg-only figure with the manuscript style, changed the y-axis to MSE for clearer reading, added value labels, and moved the single-seed note out of the plotting area. / 同步固定图/动态图 FedAvg-only 图的主文风格，将 y 轴改为更易读的 MSE，增加数值标注，并将单种子提示移出主绘图区。 | No / 否 |

## 3. Regenerated Figures / 重新生成图表

| Figure / 图 | PNG / 预览图 | PDF / 排版图 | Style Synced? / 是否风格同步 |
|---|---|---|---|
| CNN enhanced non-IID / CNN 增强 Non-IID | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.png` | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_noniid_fedavg_only.pdf` | Yes / 是 |
| CNN enhanced client scale / CNN 增强客户端数量 | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.png` | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_client_scale_fedavg_only.pdf` | Yes / 是 |
| CNN enhanced feature ablation / CNN 增强特征消融 | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.png` | `results/simulation_experiments/cnn_fed_enhanced_experiments/paper_ready/cnn_enhanced_feature_ablation_fedavg_only.pdf` | Yes / 是 |
| Robustness: client dropout / 鲁棒性：客户端掉线 | `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.png` | `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_client_dropout_fedavg_only.pdf` | Yes / 是 |
| Robustness: communication delay / 鲁棒性：通信延迟 | `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_communication_delay_fedavg_only.png` | `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_communication_delay_fedavg_only.pdf` | Yes / 是 |
| Robustness: gradient perturbation / 鲁棒性：梯度扰动 | `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_gradient_noise_fedavg_only.png` | `results/simulation_experiments/fed_robustness/paper_ready/fed_robustness_gradient_noise_fedavg_only.pdf` | Yes / 是 |
| GCN fixed vs dynamic / GCN 固定图与动态图 | `results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.png` | `results/simulation_experiments/gcn_fed_enhanced_experiments/paper_ready/gcn_fixed_vs_dynamic_fedavg_only.pdf` | Yes / 是 |

## 4. Style Rules / 风格规则

**English:**
- Theme: `seaborn` with `style="whitegrid"`, `context="paper"`, `font_scale=1.15`
- Font: `DejaVu Sans`
- DPI: `figure.dpi = 150`, `savefig.dpi = 300`
- Title size: `11`; axis label size: `10`; tick label size: `9`; legend size: `10`
- Main FedAvg color: `#0072B2`
- Category palette for CNN bar figures: `#0072B2`, `#009E73`, `#D55E00`, `#CC79A7`, `#F0E442`
- Grid: y-axis only, alpha `0.35`
- Line width: `2.0`; marker size: `5`
- Export: same-name PNG + PDF, no SVG

**中文：**
- 主题：`seaborn`，`style="whitegrid"`，`context="paper"`，`font_scale=1.15`
- 字体：`DejaVu Sans`
- 分辨率：`figure.dpi = 150`，`savefig.dpi = 300`
- 标题字号：`11`；坐标轴字号：`10`；刻度字号：`9`；图例字号：`10`
- FedAvg 主色：`#0072B2`
- CNN 柱状图类别配色：`#0072B2`、`#009E73`、`#D55E00`、`#CC79A7`、`#F0E442`
- 网格：仅保留 y 轴网格，透明度 `0.35`
- 线宽：`2.0`；marker 大小：`5`
- 输出：同名 PNG + PDF，不输出 SVG

## 5. FedAvg Mainline Check / FedAvg 主线检查

**English:**
All regenerated paper-ready figures remain FedAvg-only. No figure includes `Proposed`, `Loss-weighted`, or `Data-loss weighted`. The gradient-noise figure keeps the wording "Simulated gradient perturbation; not formal differential privacy." The GCN fixed-vs-dynamic figure keeps the single-seed trend warning and does not overstate the result.

**中文：**
所有重新生成的 paper-ready 图仍然是 FedAvg-only，不包含 `Proposed`、`Loss-weighted` 或 `Data-loss weighted`。梯度噪声图继续保留 “Simulated gradient perturbation; not formal differential privacy.” 说明；GCN 固定图/动态图图继续保留单种子趋势性提醒，未夸大结果。

## 6. Verification / 验证结果

| Check / 检查项 | Result / 结果 |
|---|---|
| `cfe_visualization.py` passes `py_compile` / `cfe_visualization.py` 通过 `py_compile` | Passed / 通过 |
| `fr_visualization.py` passes `py_compile` / `fr_visualization.py` 通过 `py_compile` | Passed / 通过 |
| `gfe_visualization.py` passes `py_compile` / `gfe_visualization.py` 通过 `py_compile` | Passed / 通过 |
| All target PNG files exist / 所有目标 PNG 存在 | Yes / 是 |
| All target PDF files exist / 所有目标 PDF 存在 | Yes / 是 |
| PNG/PDF names remain matched / PNG/PDF 保持同名 | Yes / 是 |
| Paper-ready figures remain FedAvg-only / paper-ready 图仍为 FedAvg-only | Yes / 是 |
| Gradient perturbation figure keeps non-DP note / 梯度扰动图保留非 DP 说明 | Yes / 是 |
| GCN figure keeps single-seed trend note / GCN 图保留单种子趋势提示 | Yes / 是 |
| Any core file modified? / 是否修改 core 文件 | No / 否 |
| Any LaTeX file modified? / 是否修改 LaTeX 文件 | No / 否 |
| Any CSV regenerated? / 是否重新生成 CSV | No / 否 |
| Any runner/pipeline/config/utils added? / 是否新增 runner/pipeline/config/utils | No / 否 |

## 7. Remaining Issues / 剩余问题

- The current GCN fixed-vs-dynamic paper-ready figure is visually cleaner, but the underlying evidence is still single-seed only and should remain a trend-level figure in the manuscript. / 当前 GCN 固定图/动态图 paper-ready 图已更清晰，但底层证据仍是单种子结果，正文中仍应维持趋势性表述。
- The CNN paper-ready figures now use category colors instead of a single blue tone to improve readability across categories; if the author prefers a stricter monochrome manuscript style, this can be changed in a later figure-only pass without touching CSV or core logic. / CNN paper-ready 图目前使用类别配色而不是单一蓝色，以增强类别辨识度；若作者更偏好严格单色主文风格，可在后续仅图表阶段继续微调，无需改 CSV 或 core。
