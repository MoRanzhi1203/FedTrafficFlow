# FedTrafficFlow 项目进展分析报告

> 生成日期：2026-06-12 | 基于最新 commit 471c696

---

## 1. 项目总览

本项目（FedTrafficFlow）围绕**联邦交通流预测**展开，包含两条并行的实验主线：

| 主线 | 说明 | 状态 |
|------|------|------|
| **仿真实验（Simulation）** | CNN/GCN 联邦学习在合成 Non-IID 数据上的实验 | ✅ 已完成 |
| **真实数据管线（Real Data）** | 真实交通数据预处理→分析→缺失注入→插补实验→联邦预测 | 🔶 预处理完成，预测训练尚未启动 |

---

## 2. 两条主线详析

### 2.1 仿真实验（Simulation Experiments）— 已完成 ✅

基于合成数据的联邦学习实验体系已完成，包括：

| 实验类别 | 脚本 | 输出目录 |
|----------|------|----------|
| CNN 基础仿真 | simulation_experiments/cnn_fed_base.py | esults/simulation_experiments/cnn_fed_base/ |
| GCN 基础仿真 | simulation_experiments/gcn_fed_base.py | esults/simulation_experiments/gcn_fed_base/ |
| CNN 增强实验 | simulation_experiments/cnn_fed_enhanced_experiments.py | esults/simulation_experiments/cnn_fed_enhanced/ |
| GCN 增强实验 | simulation_experiments/gcn_fed_enhanced_experiments.py | esults/simulation_experiments/gcn_fed_enhanced/ |
| 鲁棒性实验 | simulation_experiments/fed_robustness_experiments.py | esults/simulation_experiments/fed_robustness/ |

**已完成的工作：**
- 模型架构：CNN-BiLSTM-Attention + GCN-BiLSTM-Attention
- 聚合策略：标准 FedAvg vs 增强聚合（Proposed）
- 消融实验：图结构（固定/动态/功能相似/拥堵延迟）
- 鲁棒性实验：客户端掉线、通信延迟、DP噪声、通信开销
- 多种子验证（5 seeds）
- 合成数据集构造总览图（2×2面板）
- 论文 LaTeX 源码已编译为 paper_revision/latex_source/main.pdf

### 2.2 真实数据管线（Real Data Pipeline）— 进行中 🔶

#### 阶段 1：数据预处理 ✅ 已完成

| 步骤 | 脚本 | 输出 | 状态 |
|------|------|------|------|
| 路段GPS清洗 | preprocessing_scripts/process_link_gps.py | data/processed/link_gps_processed.csv (1.4MB) | ✅ |
| 路网属性清洗 | preprocessing_scripts/process_rnsd.py | data/processed/rnsd_processed.csv (3.4MB) | ✅ |
| 速度数据合并 | preprocessing_scripts/merge_speed_data.py | data/processed/speed_data_chunks/*.parquet ×61 (≈188MB×61) | ✅ |

**关键指标：**
- 路段/坐标记录：45,148 条
- 拓扑/观测节点数：42,031（完全匹配）
- 日分片：61 天，每天 96 个时间片
- 总观测记录：246,133,536 条（零缺失、零重复、零异常）

#### 阶段 2：数据分析与指标派生 ✅ 已完成

| 步骤 | 脚本 | 输出 |
|------|------|------|
| 速度统计 | summarize_speed_stats.py | data/analysis/speed_class_overall_stats.csv |
| 速度直方图 | isualize_speed_hist_by_period.py | data/analysis/speed_histograms_by_period_by_class/ |
| p995标注 | dd_p995_to_speed_histogram.py | data/analysis/speed_histograms_by_class_p995.csv |
| Greenshields密度 | compute_greenshields_density.py | data/analysis/density_metrics_chunks/ |
| 节点路口流量 | compute_node_intersection_flow_optimized.py | data/analysis/node_intersection_flow_parquet/ ×61 |
| 节点日曲线拟合 | it_node_flow_daily_curve.py | data/analysis/node_flow_curve_fit/ |
| 日期类型对比 | compare_date_type_curve_methods.py | data/analysis/date_type_curve_method_comparison/ |
| 傅里叶阶数对比 | compare_node_flow_fourier_orders.py | data/analysis/node_flow_curve_fit/ |
| 函数聚类可视化 | isualize_fitted_function_clusters.py | data/analysis/date_type_curve_method_comparison/function_cluster_visualization/ |
| 空间完整性检查 | check_spatial_node_completeness.py | data/analysis/node_intersection_flow_check_reports/ |

**关键发现：**
- 速度等级分布：等级5（35.13 km/h）记录量最大（1.13亿条），偏中度拥堵
- 节点日曲线傅里叶拟合：中位数 R² ≈ 0.93，平均 R² ≈ 0.90，RMSE ≈ 37.64
- PCA + K-Means 聚类（k=3–6）：轮廓系数 0.15–0.50

#### 阶段 3：缺失值注入与插补实验 ✅ 已完成

**3a. 初始缺失实验（eal_data_missingness_experiment.py）**

| 参数 | 值 |
|------|-----|
| 机制 | MCAR（随机独立缺失） |
| 缺失率 | 0%, 5%, 10%, 20%, 30% |
| 种子 | 42, 2024, 3407, 1234, 5678 |
| 插补方法 | zero_fill, forward_fill, linear_interpolation |
| 抽样 | 10文件 × 500行 = 250 行 design, 250 行 mask, 750 行 quality |
| 输出 | esults/real_data_missingness_experiments/ |

**3b. 完整路口阶段历史因果实验（ull_intersection_missingness_pipeline.py）— 最新**

这是最近 commit 471c696 的核心内容：

| 参数 | 值 |
|------|-----|
| 约束 | **严格因果历史**：只能用目标时间点之前的数据 |
| 缺失率 | 5%（当前运行） |
| 机制 | MCAR 逐点 |
| 种子 | 42 |
| 插补方法 | zero_fill, forward_fill, historical_linear_extrapolation, geo_neighbor_fill, function_curve_fit, geo_func_hybrid |
| 历史天数 | 7 天上下文（不含未来） |
| Warmup | 前 7 天排除于主指标 |
| 分位数截断 | 节点流量的 P95 × 1.5 上限 |
| 输出 | esults/real_data_missingness_full_intersection_causal_history/ |

**已运行子实验：**
- esults/real_data_missingness_full_intersection_causal_history/ — 主实验
- esults/real_data_missingness_full_intersection_causal_history_hybridtest/ — geo_func_hybrid 测试
- esults/real_data_missingness_full_intersection_causal_history_hybridtest_small/ — 小规模混合测试
- esults/real_data_missingness_full_intersection_causal_history_smoketest/ — 冒烟测试
- esults/real_data_missingness_experiments_geo_func/ — 地理函数子实验
- esults/real_data_missingness_experiments_medium/ — 中等规模实验
- esults/real_data_missingness_experiments_sample/ — 抽样实验

#### 阶段 4：预处理审计 ✅ 已完成

nalysis_scripts/audit_real_data_preprocessing.py 已执行，输出：
- esults/real_data_preprocessing/real_data_file_inventory.csv
- esults/real_data_preprocessing/real_data_quality_summary.csv
- esults/real_data_preprocessing/real_data_preprocessing_audit.json
- esults/real_data_preprocessing/real_data_preprocessing_audit.md

---

## 3. 论文相关文件

| 类别 | 路径 | 说明 |
|------|------|------|
| LaTeX 源码 | paper_revision/latex_source/main.tex | 完整论文源码 |
| 编译后PDF | paper_revision/latex_source/main.pdf | 已编译 |
| 公式笔记 | paper_revision/formula_notes/ | 6个公式说明文件 |
| 手稿章节 | paper_revision/manuscript_sections_zh/ | 中文论文段落 |
| 历史版本 | paper_revision/manuscript_sections_zh/history/ | 回退存档 |
| 审稿材料 | paper_revision/reviewer_materials/ | 审稿相关 |
| 策略文档 | paper_revision/00-05_*.md | 修订策略与约束 |

**已完成的论文章节（中文）：**
- eal_data_preprocessing_audit_zh.md — 真实数据预处理审计
- eal_data_preprocessing_experiment_results_zh.md — 预处理与质量分析
- eal_data_missingness_experiment_design_zh.md — 缺失实验设计
- eal_data_missingness_experiment_results_zh.md — 缺失实验结果
- simulation_experiment_formal_module_zh_v4.md — 仿真实验正式模块 v4

---

## 4. 代码执行状态验证

根据 esults/code_execution_status_report_zh.md：

| 脚本 | 可编译 | 已执行 | 输出完整 |
|------|--------|--------|----------|
| udit_real_data_preprocessing.py | ✅ | ✅ | ✅ |
| eal_data_missingness_experiment.py | ✅ | ✅ | ✅ |
| ull_intersection_missingness_pipeline.py | ✅ | ✅ | ✅（最新执行） |

环境：Python 3.9.23 @ E:\anaconda3\envs\analysis\python.exe

---

## 5. 当前卡点：尚未完成的模块 🚧

如上所述，paper_revision/manuscript_sections_zh/real_data_prediction_pipeline_next_steps_zh.md 列出了 **7 个必须补齐的模块**，目前均未启动：

| # | 模块 | 说明 | 状态 |
|---|------|------|------|
| 1 | **真实数据滑动窗口构造** | 从时序数据生成 (X, Y) 训练样本 | ❌ |
| 2 | **预测任务归一化与 Scaler 保存** | 标准缩放器，保存参数供推理使用 | ❌ |
| 3 | **训练/验证/测试划分** | 时序划分（非随机打散） | ❌ |
| 4 | **联邦客户端划分** | 按节点/区域分组为联邦客户端 | ❌ |
| 5 | **真实路网邻接矩阵构造** | 从 RNSD 生成标准化 .npy 邻接矩阵 | ❌ |
| 6 | **FedAvg / Independent 训练入口** | 接入已有联邦模型到真实数据 | ❌ |
| 7 | **缺失率预测鲁棒性评估** | 在不同缺失率下评估预测性能 | ❌ |

**推荐实现顺序（来自文档）：**
`
窗口化 → 客户端划分 → 归一化与划分 → 邻接矩阵 → 接入 FedAvg/Independent → 缺失率鲁棒性
`

---

## 6. 完整文件清单（关键文件）

### 原始数据（3个，≈8GB）
`
data/raw/link_gps.v2                         (1.4 MB)
data/raw/road_network_sub-dataset.v2         (2.2 MB)
data/raw/traffic_speed_sub-dataset.v2        (8.3 GB)
`

### 预处理后数据（≈11.5 GB）
`
data/processed/link_gps_processed.csv        (1.5 MB)
data/processed/rnsd_processed.csv            (3.4 MB)
data/processed/speed_data_chunks/*.parquet   (61 chunks × ≈188 MB)
data/analysis/node_intersection_flow_parquet/*.parquet  (61 chunks)
`

### 分析脚本（14个）
`
analysis_scripts/add_p995_to_speed_histogram.py
analysis_scripts/audit_real_data_preprocessing.py
analysis_scripts/check_spatial_node_completeness.py
analysis_scripts/compare_date_type_curve_methods.py
analysis_scripts/compare_node_flow_fourier_orders.py
analysis_scripts/compute_greenshields_density.py
analysis_scripts/compute_node_intersection_flow_optimized.py
analysis_scripts/fit_node_flow_daily_curve.py
analysis_scripts/full_intersection_missingness_pipeline.py    ← 最新
analysis_scripts/real_data_missingness_experiment.py
analysis_scripts/summarize_speed_stats.py
analysis_scripts/visualize_fitted_function_clusters.py
analysis_scripts/visualize_node_flow_daily_curve_fit.py
analysis_scripts/visualize_speed_hist_by_period.py
`

### 检查脚本（5个）
`
dataset_inspection_scripts/check_density_time_order.py
dataset_inspection_scripts/inspect_density_metrics_chunks.py
dataset_inspection_scripts/inspect_node_intersection_flow.py
dataset_inspection_scripts/inspect_road_directionality.py
dataset_inspection_scripts/inspect_speed_data_chunks.py
`

---

## 7. 下一步行动建议

### 优先级 P0：启动真实数据联邦预测训练

当前全缺失实验（插补质量评估）已完成，但**联邦预测训练**尚未启动。建议按以下顺序推进：

`
Step 1: 真实数据窗口化
  输入: data/analysis/node_intersection_flow_parquet/*.parquet
  输出: (X, Y) 滑动窗口样本，例如 12步输入 → 1/3/6步预测
  参数: input_len, pred_len, stride, target_col=路口车流量

Step 2: 客户端划分
  基于节点流量聚类（已有 PCA + K-Means 结果）或区域划分
  生成 client_id → node_ids 映射

Step 3: 归一化与划分
  按客户端做 MinMax/Standard 归一化，保存 scaler
  时序划分 train/val/test（无随机打散）

Step 4: 邻接矩阵
  从 rnsd_processed.csv 的 snodeid ↔ enodeid 构造
  保存为 adj_matrix.npy 或 adj_matrix.npz

Step 5: 接入 FedAvg/Independent
  复用 simulation_experiments/ 中的模型定义
  创建 real_data 训练入口脚本

Step 6: 缺失率预测鲁棒性
  在 0%, 5%, 10%, 20% 缺失率下评估预测 RMSE/MAE
`

### 优先级 P1：论文撰写
- 将仿真实验部分翻译/整理为英文正文
- 将真实数据预处理与缺失实验结果写入论文
- 补齐真实数据来源与引用信息

### 优先级 P2：基础设施
- 保存标准化邻接矩阵文件（当前只有代码中动态构造，无持久化文件）
- 添加异常值裁剪逻辑

---

## 8. Git 提交历史（最近30条）

`
471c696 Add causal historical full-intersection missingness pipeline     ← 最新
80f5586 Add sample missingness experiment outputs
9f0f142 Add log-scale RMSE plot for sample missingness results
558a8b8 Update missingness experiment assets and add execution status report
39fee48 Validate real data missingness outputs and add next-step plan
14fd705 Fix real data missingness script for Python 3.9
4027c3e update: real data preprocessing audit and manuscript (2026-06-10)
739e48a Add real data preprocessing audit and manuscript section
...（仿真实验相关提交从 6d5426f 到 2f6136f）
`

---

## 9. 总结

| 维度 | 状态 | 完成度 |
|------|------|--------|
| 仿真实验（合成数据） | 全部完成 | 100% |
| 真实数据预处理 | 全部完成 | 100% |
| 真实数据分析与指标派生 | 全部完成 | 100% |
| 缺失注入与插补实验 | 全部完成（含最新历史因果约束） | 100% |
| 预处理审计 | 全部完成 | 100% |
| **真实数据联邦预测训练** | **尚未启动** | **0%** |
| 论文撰写 | 中文草稿已有，英文待译 | ~60% |

**当前阶段定位：从"数据准备与插补评估"向"真实数据联邦预测训练"过渡的临界点。**
