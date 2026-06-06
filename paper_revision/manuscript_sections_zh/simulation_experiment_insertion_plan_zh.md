# 仿真实验章节插入计划

## 1. 原论文实验章节位置

根据 `paper_revision/latex_source/main.tex` 的章节结构，实验相关内容分布于：

| 章节 | LaTeX 标签 | 大致行号范围 | 当前内容 |
|------|-----------|------------|---------|
| Methodology | `\section{Methodology}` | 127–424 | 框架描述 + 局部建模函数实例化 |
| Data analysis | `\section{Data analysis}` | 425–764 | 含 Synthetic Experiments 和 Real-World Data Analysis |
| Synthetic Experiments | `\subsection{Synthetic Experiments}` | 428–561 | 现有仿真实验内容 |
| Real-World Data Analysis | `\subsection{Real-World Data Analysis}` | 562–763 | 真实数据实验 |

## 2. 建议新增或替换的内容

以下建议仅作为后续 LaTeX 修改的参考，**不要求立即执行替换或编译**。

| 章节 | 建议操作 | 中文内容来源 | 后续处理 |
|------|---------|------------|---------|
| Synthetic Experiments（仿真实验） | 用新内容**整体替换**现有 subsection（lines 428–561） | `manuscript_sections_zh/simulation_experiment_section_zh.md` | 需人工审阅后逐段转换为 LaTeX |
| 实验设计概览 + 数据生成 | 新增 subsubsection | Section 1–4 | 合并现有 `\subsubsection{Synthetic Data and Experimental Settings}` |
| 基础联邦对比（CNN+GCN） | 新增两个 subsubsection | Section 5–7 | 替换现有 `\subsubsection{Synthetic Results and Analysis}` |
| 异质性与鲁棒性分析 | 新增 subsubsection | Section 8–9 | 原论文可能无此内容，全新添加 |
| 图结构与消融 | 新增 subsubsection | Section 10 | 补充 GCN 消融分析 |
| 图表引用建议 | 不写入正文 | Section 11 | 仅供作者参考 |
| 小结 | 天然置于 Conclusion 之前 | Section 12 | 可与现有 `\section{Conclusions}` 衔接 |

## 3. 建议新增图表

| 图表 | 来源文件 | 建议放置位置 | 是否需要 paper-ready 版本 | 备注 |
|------|---------|------------|------------------------|------|
| CNN-FedAvg 收敛曲线 | `cnn_fed_base/convergence_history.csv` | Section 5（基础联邦） | 是，需用 matplotlib/ggplot 绘制 | 双纵轴：训练损失 + 验证 RMSE |
| GCN-FedAvg 收敛曲线 | `gcn_fed_base/convergence_history.csv` | Section 6 | 是 | 样式同 CNN 图 |
| 非 IID 等级对比柱状图 | `cnn_enhanced_noniid_summary.csv`（仅 FedAvg 行） | Section 8 | 是 | 三级对比：低/中/高 |
| 鲁棒性测试分组柱状图 | `fed_robustness/` 下三个 summary CSV | Section 9 | 是 | 掉线/延迟/噪声三组，仅取 FedAvg 行 |
| 特征消融柱状图 | `cnn_enhanced_feature_ablation_summary.csv`（FedAvg 行） | Section 10 | 是 | 五种配置 |
| CNN FedAvg vs Independent 逐客户端表 | `cnn_fed_base/main_metrics.csv` | Section 5 | 否（直接 CSV 值制表） | 三线表 |
| GCN FedAvg vs Independent 汇总表 | `gcn_fed_base/main_summary.csv` | Section 6 | 否 | 三线表 |
| 特征消融表 | `cnn_enhanced_feature_ablation_summary.csv`（FedAvg 行） | Section 10 | 否 | 三线表 |
| 鲁棒性汇总表 | `fed_robustness/` 下三个 summary CSV（仅 FedAvg 行） | Section 9 | 否 | 可合成一张表或分三张小表 |

## 4. 需要作者确认的问题

1. **篇幅控制**：当前仿真实验章节拟定内容较长（约 12 个子节 + 4 个表格 + 建议 5 张图），添加到现有 LaTeX 后可能超出目标期刊页数限制。建议与导师确认是否保留全文还是精简为 6–8 个核心子节。

2. **图表制作责任**：上述"待制作"的 5 张图需要从 CSV 数据重新绘制为 paper-ready 格式的矢量图（PDF/EPS）。请确认由谁负责制作（作者自行绘制、寻找绘图工具、还是后续委托生成）。

3. **LaTeX 转换**：本文件中的中文实验章节需转换为英文 LaTeX 格式后才能插入 `main.tex`。请确认是否由作者自行翻译和转换，或安排后续文字润色工作。

4. **引用更新**：替换仿真实验章节后，`\section{Conclusions}` 和 `\subsection{Real-World Data Analysis}` 中可能存在的交叉引用需要同步更新。

5. **现有图表保留**：原有 `main.tex` 中仿真实验部分可能附有图片引用（`\includegraphics`），替换内容时需决定保留、替换还是删除这些图片引用。

6. **与真实数据实验的衔接**：仿真实验的新增分析（非 IID、鲁棒性等）应与真实数据实验部分保持逻辑衔接，建议在真实数据实验中也增加对应的异质性讨论段落。

---

*本文件提供的是**结构性建议**，不要求立即修改 LaTeX 源文件、不要求立即编译、不要求重跑任何实验。所有数值均来自已有 CSV。*
