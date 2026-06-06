# Simulation Visualization Optimization Plan / 仿真可视化优化计划

## Scope / 范围

**English:**
This document defines the visualization optimization plan for all 5 simulation experiment visualization files. It does NOT modify core logic, data generation, or aggregation strategy code.

**中文：**
本文档定义所有 5 个仿真实验可视化文件的可视化优化计划。不修改核心逻辑、数据生成或聚合策略代码。

## Prerequisites / 前置条件

**English:**
- `simulation_experiment_report.md` has been generated and reviewed
- All core CSV outputs are confirmed consistent with visualization inputs
- No new experiments need to be run

**中文：**
- `simulation_experiment_report.md` 已生成并审核
- 所有核心 CSV 输出已确认与可视化输入一致
- 无需重新运行实验

## Strategy: Patch Visualization Files Only / 策略：仅修改可视化文件

**English:**
All changes are directed at the 5 `*_visualization.py` files. Each file receives targeted patches for:
1. PDF export (where missing)
2. Colorblind-friendly palette
3. Font/legend consistency
4. "Proposed"/Loss-weighted suppression (FedAvg-only charts for paper-ready output)
5. DP disclaimer in gradient noise chart

**中文：**
所有修改仅针对 5 个 `*_visualization.py` 文件。每个文件接收针对性补丁以修复：
1. PDF 导出（缺失处）
2. 色盲友好调色板
3. 字体/图例一致性
4. "Proposed"/Loss-weighted 抑制（论文就绪输出仅 FedAvg 图表）
5. 梯度噪声图中的差分隐私声明

---

## Step 1: `cfb_visualization.py` — CNN Fed Base / CNN联邦基础

**English - Current issues:**
- No PDF output (PNG only, despite docstring saying "PDF for vector quality")
- `METHOD_PALETTE` includes "Proposed" (unused in data, but confusing)
- font_scale=1.2 different from other groups (1.15)
- Legend fontsize=8 too small

**中文 - 当前问题：**
- 无 PDF 输出（仅 PNG，尽管文档字符串说"PDF for vector quality"）
- `METHOD_PALETTE` 包含 "Proposed"（数据中未使用但令人困惑）
- font_scale=1.2 与其他组不同（1.15）
- 图例 fontsize=8 过小

### 1.1) Add PDF export / 添加PDF导出

**English:**
Modify each plot function to also save a `.pdf` in the same output directory. Create a local helper `_save_fig` patterned after `_save` in `cfe_visualization.py`.

**中文：**
修改每个绘图函数，使其同时在同一输出目录中保存 `.pdf`。参照 `cfe_visualization.py` 中的 `_save` 创建本地辅助函数 `_save_fig`。

```python
def _save_fig(fig, output_dir: Path, filename: str):
    png_path = ensure_dir(output_dir) / filename
    fig.savefig(png_path, bbox_inches="tight")
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
```

**English:** Replace all `fig.savefig(out_path, bbox_inches="tight")` + `plt.close(fig)` with `_save_fig(fig, output_dir, filename)`.

**中文：** 将所有 `fig.savefig(out_path, bbox_inches="tight")` + `plt.close(fig)` 替换为 `_save_fig(fig, output_dir, filename)`。

### 1.2) Remove "Proposed" from METHOD_PALETTE / 从METHOD_PALETTE移除"Proposed"

```python
METHOD_PALETTE = {
    "Independent": "#4C72B0",
    "FedAvg": "#DD8452",
}
```

### 1.3) Standardize to font_scale=1.15 / 统一为font_scale=1.15

**English:** Change `font_scale=1.2` → `font_scale=1.15`.

**中文：** 将 `font_scale=1.2` 改为 `font_scale=1.15`。

### 1.4) Increase legend font / 增大图例字体

**English:** Change `fontsize=8` → `fontsize=10`.

**中文：** 将 `fontsize=8` 改为 `fontsize=10`。

---

## Step 2: `cfe_visualization.py` — CNN Fed Enhanced / CNN联邦增强

**English - Current issues:**
- Shows Loss-weighted, Data-loss weighted, Proposed alongside FedAvg as equals
- These non-FedAvg strategies must not appear in paper-ready charts

**中文 - 当前问题：**
- 将 Loss-weighted、Data-loss weighted、Proposed 与 FedAvg 等同显示
- 这些非 FedAvg 策略不得出现在论文就绪图表中

### 2.1) Add paper-ready output directory parameter / 添加论文就绪输出目录参数

**English:**
Add a command-line argument `--paper-ready` that, when set, filters data to only FedAvg and Independent rows before plotting, and saves to a `paper_ready/` subdirectory.

**中文：**
添加命令行参数 `--paper-ready`，设置后会在绘图前过滤数据仅保留 FedAvg 和 Independent 行，并保存到 `paper_ready/` 子目录。

### 2.2) Add `_filter_paper_methods()` helper / 添加辅助函数

```python
PAPER_METHODS = {"FedAvg", "Independent"}

def _filter_paper_methods(df: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
    return df[df[method_col].isin(PAPER_METHODS)].copy()
```

### 2.3) In `--paper-ready` mode / 在论文就绪模式下

**English:**
- Filter all dataframes to only PAPER_METHODS
- Save to `output_dir / "paper_ready" /` subdirectory
- Use `METHOD_PALETTE` without Loss-weighted/Data-loss weighted/Proposed

**中文：**
- 将所有数据框过滤为仅保留 PAPER_METHODS
- 保存到 `output_dir / "paper_ready" /` 子目录
- 使用不含 Loss-weighted/Data-loss weighted/Proposed 的 `METHOD_PALETTE`

### 2.4) Keep full comparison output in default mode / 默认模式保留完整对比输出

**English:** Default mode (no `--paper-ready`) keeps current behavior for internal reference.

**中文：** 默认模式（无 `--paper-ready`）保持当前行为以供内部参考。

---

## Step 3: `fr_visualization.py` — Fed Robustness / 联邦鲁棒性

**English - Current issues:**
- `METHOD_PALETTE` includes "Proposed" 
- Gradient noise chart lacks DP disclaimer
- Charts show FedAvg vs Proposed as if both are main methods

**中文 - 当前问题：**
- `METHOD_PALETTE` 包含 "Proposed"
- 梯度噪声图缺少差分隐私声明
- 图表显示 FedAvg vs Proposed，仿佛两者都是主方法

### 3.1) Remove "Proposed" from METHOD_PALETTE / 从METHOD_PALETTE移除"Proposed"

```python
METHOD_PALETTE = {
    "FedAvg": "#DD8452",
}
```

### 3.2) Add `--paper-ready` mode like cfe_visualization / 添加类似cfe的论文就绪模式

**English:** Filter data to only FedAvg rows; save to `paper_ready/` subdirectory.

**中文：** 过滤数据仅保留 FedAvg 行；保存到 `paper_ready/` 子目录。

### 3.3) Add DP disclaimer to gradient noise function / 在梯度噪声函数中添加差分隐私声明

**English:**
In `plot_fed_robustness_gradient_noise()`, add a text annotation:

**中文：**
在 `plot_fed_robustness_gradient_noise()` 中添加文本注释：

```python
ax.text(0.5, -0.15, "Note: Simulated gradient noise perturbation — not formal differential privacy.",
        transform=ax.transAxes, ha="center", fontsize=9, fontstyle="italic")
```

**English:** This clarifies that the experiment uses simulated perturbation only and does not constitute formal differential privacy guarantees.

**中文：** 此注释澄清实验仅为模拟扰动，不构成正式差分隐私保障。

### 3.4) Update title for gradient noise chart / 更新梯度噪声图标题

**English:** Change title from "Gradient Noise Robustness" to "Model Robustness under Simulated Gradient Perturbations".

**中文：** 将标题从 "Gradient Noise Robustness" 改为 "Model Robustness under Simulated Gradient Perturbations"（模拟梯度扰动下的模型鲁棒性）。

---

## Step 4: `gfb_visualization.py` — GCN Fed Base / GCN联邦基础

**English - Current issues:**
- Clean (only FedAvg/Independent), but legend fontsize=8 too small

**中文 - 当前问题：**
- 干净（仅 FedAvg/Independent），但图例 fontsize=8 过小

### 4.1) Standardize to font_scale=1.15 / 统一为font_scale=1.15

**English:** (Already 1.15 — verify)

**中文：** （已为 1.15 — 验证确认）

### 4.2) Increase legend font / 增大图例字体

**English:** Change `fontsize=8` → `fontsize=10`.

**中文：** 将 `fontsize=8` 改为 `fontsize=10`。

### 4.3) Verify PDF output already present / 验证PDF输出已存在

**English:** (Already present via `fig.savefig(pdf_path, bbox_inches="tight")`)

**中文：** （已通过 `fig.savefig(pdf_path, bbox_inches="tight")` 实现）

---

## Step 5: `gfe_visualization.py` — GCN Fed Enhanced / GCN联邦增强

**English - Current issues:**
- Shows Loss-weighted, Data-loss weighted, Proposed alongside FedAvg
- 1-seed only results

**中文 - 当前问题：**
- 将 Loss-weighted、Data-loss weighted、Proposed 与 FedAvg 并列显示
- 仅为 1 种子结果

### 5.1) Add `--paper-ready` mode / 添加论文就绪模式

**English:** Same pattern as Step 2 for `cfe_visualization.py`.

**中文：** 与 Step 2 的 `cfe_visualization.py` 相同模式。

### 5.2) In paper-ready mode / 在论文就绪模式下

**English:**
- Filter to only PAPER_METHODS
- Save to `paper_ready/` subdirectory
- Add note about "1-seed preliminary results" in plot title or caption

**中文：**
- 过滤为仅保留 PAPER_METHODS
- 保存到 `paper_ready/` 子目录
- 在图表标题或图注中添加"单种子初步结果"说明

### 5.3) In 1-seed mode, add caption note / 单种子模式下添加说明注释

```python
if len(df["seed"].unique()) == 1:
    ax.set_title(ax.get_title() + " (single-seed, preliminary)")
```

**English:** This warns readers that results are based on a single random seed and should be interpreted as preliminary.

**中文：** 此警告告知读者结果基于单个随机种子，应视为初步结果。

---

## Step 6: Cross-cutting — Colorblind-Friendly Palette / 跨模块—色盲友好调色板

### 6.1) Define shared Wong palette / 定义共享Wong调色板

```python
WONG_PALETTE = [
    "#0072B2",  # blue / 蓝色
    "#D55E00",  # vermillion / 朱红
    "#009E73",  # green / 绿色
    "#CC79A7",  # reddish purple / 红紫
    "#F0E442",  # yellow / 黄色
    "#56B4E9",  # sky blue / 天蓝
]
```

### 6.2) Reassign METHOD_PALETTE per file / 各文件重新分配METHOD_PALETTE

**English:**
| Method / 方法 | Wong Color / Wong颜色 |
|---|---|
| FedAvg | `#0072B2` (blue / 蓝色) |
| Independent | `#D55E00` (vermillion / 朱红) |
| Loss-weighted (internal only / 仅内部) | `#009E73` (green / 绿色) |
| Data-loss weighted (internal only / 仅内部) | `#CC79A7` (purple / 紫色) |
| Proposed (internal only / 仅内部) | `#F0E442` (yellow / 黄色) |

**中文：**
FedAvg 和 Independent 使用最区分的颜色（蓝 vs 朱红），确保论文主线图表在色盲条件下仍然可辨。历史探索策略使用辅助颜色。

### 6.3) Apply in each visualization file / 在各可视化文件中应用

**English:** Replace existing METHOD_PALETTE with Wong-based assignments.

**中文：** 将现有 METHOD_PALETTE 替换为基于 Wong 的分配。

---

## Step 7: Consistency Standardization / 一致性标准化

**English:**
| Parameter / 参数 | Target Value / 目标值 |
|---|---|
| font_scale | 1.15 |
| figure.dpi | 150 |
| savefig.dpi | 300 |
| font.family | DejaVu Sans |
| seaborn style / seaborn样式 | whitegrid |
| seaborn context / seaborn上下文 | paper |
| Legend fontsize / 图例字号 | ≥ 10 |
| Title fontsize / 标题字号 | default (matplotlib rcParams) |
| PDF output / PDF输出 | Yes (always, alongside PNG) / 是（始终与PNG同时输出） |

**中文：**
以上参数确保 5 组实验的所有图表在字体、分辨率、样式和输出格式方面完全一致。

---

## Step 8: Execution Order / 执行顺序

**English:**
1. **Patch `cfb_visualization.py`** — Add PDF, fix palette, standardize font
2. **Patch `gfb_visualization.py`** — Fix legend font (cleanest files first)
3. **Patch `cfe_visualization.py`** — Add paper-ready mode
4. **Patch `gfe_visualization.py`** — Add paper-ready mode
5. **Patch `fr_visualization.py`** — Add paper-ready mode, DP disclaimer

**中文：**
1. **修改 `cfb_visualization.py`** — 添加 PDF，修复调色板，统一字体
2. **修改 `gfb_visualization.py`** — 修复图例字体（最干净的文件优先）
3. **修改 `cfe_visualization.py`** — 添加论文就绪模式
4. **修改 `gfe_visualization.py`** — 添加论文就绪模式
5. **修改 `fr_visualization.py`** — 添加论文就绪模式、差分隐私声明

---

## Step 9: Verification Checklist (post-patch) / 验证清单（补丁后）

**English:**
- [ ] All 5 visualization files import without error
- [ ] `--paper-ready` flag generates FedAvg-only charts for enhanced/robustness experiments
- [ ] All charts have PDF output alongside PNG
- [ ] Font sizes consistent across all groups
- [ ] Colorblind-friendly palette applied
- [ ] Gradient noise chart has DP disclaimer
- [ ] No "Proposed", "Loss-weighted", "Data-loss weighted" appear in paper-ready output
- [ ] Legend text ≥ 10pt in all charts

**中文：**
- [ ] 5 个可视化文件全部可正常导入无错误
- [ ] `--paper-ready` 标志为增强/鲁棒性实验生成仅 FedAvg 图表
- [ ] 所有图表均有 PDF 和 PNG 双输出
- [ ] 各组字体大小一致
- [ ] 色盲友好调色板已应用
- [ ] 梯度噪声图有差分隐私声明
- [ ] 论文就绪输出中不出现 "Proposed"、"Loss-weighted"、"Data-loss weighted"
- [ ] 所有图表图例字体 ≥ 10pt

---

## Step 10: Files to be Modified / 待修改文件

**English:**
| File / 文件 | Changes / 修改内容 |
|---|---|
| `simulation_experiments/cnn_fed_base/cfb_visualization.py` | Add PDF output, remove Proposed from palette, font_scale 1.2→1.15, legend 8→10pt, Wong palette / 添加PDF输出, 调色板移除Proposed, font_scale 1.2→1.15, 图例8→10pt, Wong调色板 |
| `simulation_experiments/cnn_fed_enhanced_experiments/cfe_visualization.py` | Add --paper-ready mode, Wong palette, filter helper / 添加--paper-ready模式, Wong调色板, 过滤辅助函数 |
| `simulation_experiments/fed_robustness_experiments/fr_visualization.py` | Add --paper-ready mode, remove Proposed from palette, DP disclaimer, Wong palette / 添加--paper-ready模式, 调色板移除Proposed, 差分隐私声明, Wong调色板 |
| `simulation_experiments/gcn_fed_base/gfb_visualization.py` | Legend 8→10pt, Wong palette / 图例8→10pt, Wong调色板 |
| `simulation_experiments/gcn_fed_enhanced_experiments/gfe_visualization.py` | Add --paper-ready mode, Wong palette, single-seed note / 添加--paper-ready模式, Wong调色板, 单种子说明 |

## Files NOT Modified / 不修改的文件

**English:**
| File / 文件 | Reason / 原因 |
|---|---|
| All `*_core.py` files / 所有核心文件 | Core logic is correct; no changes needed / 核心逻辑正确; 无需修改 |
| `simulation_experiment_report.md` | Generated report; reference only / 生成的报告; 仅参考 |
| `fed_robustness/fr_core.py` | Core logic correct; DP disclaimer added only in visualization / 核心逻辑正确; 差分隐私声明仅在可视化中添加 |

**中文：**
核心逻辑文件和报告文件均不做任何修改。DP 免责声明仅在可视化层面添加。

---

**Status / 状态：PLANNING COMPLETE — Awaiting approval to execute patches / 计划完成 — 等待批准执行补丁**

**English:**
*Generated: Phase 1 Simulation Experiment Scan & Optimization Planning*

**中文：**
*生成：第一阶段 仿真实验扫描与优化计划*