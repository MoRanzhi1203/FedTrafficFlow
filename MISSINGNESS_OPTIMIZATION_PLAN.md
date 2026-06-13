# 缺失值实验优化计划 (Missingness Experiment Optimization Plan)

> 生成日期：2026-06-12 | 基于完整项目审计

---

## 一、优化目标

将当前"单次实验验证"阶段的缺失值模块升级为**论文级完整实验体系**，补全多缺失率×多机制×多种子×多方法的交叉评估，生成可放入论文的汇总表、对比图和统计分析。

---

## 二、当前状态：两个实验系统的对比

| 维度 | 原始实验 (real_data_missingness_experiment.py) | 完整路口阶段因果实验 (full_intersection_missingness_pipeline.py) |
|------|------|------|
| **缺失率** | ✅ 0%, 5%, 10%, 20%, 30% | ❌ 仅 5% |
| **缺失机制** | ✅ MCAR | ❌ 仅 MCAR（
ode_temporal_block 未跑） |
| **种子数** | ✅ 5 seeds (42, 2024, 3407, 1234, 5678) | ❌ 仅 1 seed (42) |
| **插补方法** | 3种（zero/forward/linear_interp） | 6种（全部历史因果版本） |
| **数据规模** | 500行×10文件（抽样） | 全量61天×42031节点 |
| **因果约束** | ❌ 无（双向插值可用） | ✅ 严格历史因果 |
| **汇总表** | ✅ mean±std 表、RMSE图 | ❌ summarize 阶段未执行 |
| **流程图组分析** | ❌ | 部分（只有 
ode_flow_group_summary.csv） |
| **Warmup 排除** | ❌ | ✅ 前7天排除 |
| **分位数上限** | ❌ | ✅ P95×1.5 |
| **子实验** | medium, sample, geo_func | hybridtest, hybridtest_small, smoketest |

---

## 三、必须补齐的优化项（按优先级排列）

### P0 —— 论文必需：多缺失率全量运行
| # | 任务 | 当前状态 | 目标状态 | 预计耗时 |
|---|------|----------|----------|----------|
| 1 | **MCAR × 多缺失率 run** | 仅5%跑过 | 跑 3%, 5%, 10%, 20%, 30% | 长（全量61天×5种方法） |
| 2 | **多 seed run** | 仅 seed=42 | 追加 seed=2024, 3407（至少3个） | 长 |
| 3 | **summarize 阶段执行** | ❌ 未执行 | 生成 imputation_quality_summary*.csv、RMSE图 | 短 |
| 4 | **Mean±Std 汇总表** | ❌ 无 | 跨 seed/rate 的 mean±std 表 | 短（需先跑P0-1,2） |

### P1 —— 论文增强：block 缺失机制
| # | 任务 | 当前状态 | 目标状态 |
|---|------|----------|----------|
| 5 | **
ode_temporal_block 机制** | 代码已支持但未跑 | 跑 block=4,8,12 × rates 5%,10%,20% |
| 6 | **Block vs MCAR 对比分析** | — | 缺失模式对比（散点/连续缺失） |

### P2 —— 输出完善
| # | 任务 | 当前状态 | 目标状态 |
|---|------|----------|----------|
| 7 | **RMSE/MAE/sMAPE vs Missing Rate 曲线** | 仅原始实验有1张RMSE图 | 每种方法一条曲线，含 error bar |
| 8 | **流程图组分析图** | ❌ 无 | low/mid/high flow 三组的 RMSE 分组柱状图 |
| 9 | **Delta RMSE 热力图** | ❌ 无 | method × rate 的 RMSE 对比热力图 |
| 10 | **插补前后时序对比** | ❌ 无 | 选代表性节点，画插补前后对比时间序列 |
| 11 | **原始 vs 因果实验对比表** | ❌ 无 | 同一方法（forward_fill/zero_fill）因果有无的对比 |

### P3 —— 论文实验材料
| # | 任务 | 当前状态 | 目标状态 |
|---|------|----------|----------|
| 12 | **更新 eal_data_missingness_experiment_design_zh.md** | 已有时序块描述但未写入 | 补全实际运行参数与实验规模 |
| 13 | **更新 eal_data_missingness_experiment_results_zh.md** | 只有原始实验结果 | 加入因果约束实验结果 + 对比讨论 |
| 14 | **生成英文版 results** | — | 用于论文投稿 |

### P4 —— 代码与流程加固
| # | 任务 | 说明 |
|---|------|------|
| 15 | **summarize 阶段多实验聚合脚本** | 跨多个 output_dir 合并 summary，生成统一对比表 |
| 16 | **sMAPE 指标确认** | 当前 detail CSV 有 sum_smape 但需要确认聚合逻辑正确 |
| 17 | **零流量节点处理** | zero_fill 的 MAPE=100% 问题需要在论文中合理解释 |

---

## 四、推荐执行顺序

`
Phase 1: 环境复查 + summarize 补执行
  ├── 确认 Python 环境可用（E:\anaconda3\envs\analysis\python.exe）
  ├── 对已有 main 实验执行 summarize 阶段（获取第一份完整 summary）
  └── 确认 sMAPE 指标正确

Phase 2: MCAR × 多缺失率全量运行
  ├── 准备 rates: 0.03, 0.05, 0.10, 0.20, 0.30
  ├── 每 rate 跑 generate_missing → impute（6种方法）→ summarize
  ├── seed=42 单 seed 先跑通，验证可行性
  └── 追加 seed=2024, 3407

Phase 3: Block 缺失机制
  ├── node_temporal_block × rates 0.05, 0.10, 0.20
  ├── block_lengths=4,8,12
  └── summarize + 与 MCAR 对比

Phase 4: 可视化与表格
  ├── 跨实验汇总表（mean±std）
  ├── RMSE/MAE/sMAPE vs Rate 曲线（含 error bar）
  ├── 流程图组分析柱状图
  └── 插补前后时序对比图

Phase 5: 论文材料更新
  ├── 更新设计章节
  ├── 更新结果章节
  └── 更新 LaTeX 正文
`

---

## 五、关键命令参考

### 5.1 对已有 main 实验执行 summarize
`powershell
E:\anaconda3\envs\analysis\python.exe analysis_scripts/full_intersection_missingness_pipeline.py 
    --stage summarize 
    --output_dir results/real_data_missingness_full_intersection_causal_history 
    --missing_rates 0.05 --mechanism mcar_point --seed 42 
    --impute_methods zero_fill,forward_fill,historical_linear_extrapolation,geo_neighbor_fill,function_curve_fit,geo_func_hybrid 
    --causal_history_only True --warmup_days 7 --exclude_warmup_from_main_metrics True 
    --period 96
`

### 5.2 MCAR × 新缺失率（例：10%）
`powershell
# Step 1: generate_missing
E:\anaconda3\envs\analysis\python.exe analysis_scripts/full_intersection_missingness_pipeline.py 
    --stage generate_missing 
    --missing_rates 0.10 --mechanism mcar_point --seed 42 
    --output_dir results/real_data_missingness_full_intersection_causal_history_rate0p10 
    --max_chunks 0 --max_rows 0 
    --save_masks True --write_missing_datasets True --write_imputed_datasets False 
    --causal_history_only True --warmup_days 7

# Step 2: impute
E:\anaconda3\envs\analysis\python.exe analysis_scripts/full_intersection_missingness_pipeline.py 
    --stage impute 
    --missing_rates 0.10 --mechanism mcar_point --seed 42 
    --impute_methods zero_fill,forward_fill,historical_linear_extrapolation,geo_neighbor_fill,function_curve_fit,geo_func_hybrid 
    --output_dir results/real_data_missingness_full_intersection_causal_history_rate0p10 
    --max_chunks 0 --max_rows 0 
    --write_missing_datasets False --write_imputed_datasets True --save_masks False 
    --causal_history_only True --warmup_days 7 --resume True --skip_existing True

# Step 3: summarize
E:\anaconda3\envs\analysis\python.exe analysis_scripts/full_intersection_missingness_pipeline.py 
    --stage summarize 
    --output_dir results/real_data_missingness_full_intersection_causal_history_rate0p10 
    --missing_rates 0.10 --mechanism mcar_point --seed 42 
    --impute_methods zero_fill,forward_fill,historical_linear_extrapolation,geo_neighbor_fill,function_curve_fit,geo_func_hybrid 
    --causal_history_only True --warmup_days 7 --exclude_warmup_from_main_metrics True
`

### 5.3 node_temporal_block 机制
`powershell
# generate_missing with block mechanism
E:\anaconda3\envs\analysis\python.exe analysis_scripts/full_intersection_missingness_pipeline.py 
    --stage generate_missing 
    --missing_rates 0.05 --mechanism node_temporal_block --seed 42 
    --block_lengths 4,8,12 
    --output_dir results/real_data_missingness_full_intersection_causal_history_block 
    --max_chunks 0 --max_rows 0 
    --save_masks True --write_missing_datasets True --write_imputed_datasets False 
    --causal_history_only True --warmup_days 7
`

---

## 六、风险与注意事项

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| **全量61天×全方法计算开销大** | function_curve_fit(傅里叶) 和 geo_func_hybrid 很慢 | 可先用 10 chunks 验证，再跑全量 |
| **多 rate 磁盘开销** | 每个 rate 产生 ≈30-50 GB masks + imputed | 清理中间产物，只保留 summaries + 关键 datasets |
| **node_temporal_block 算法正确性** | 代码已写但未验证 | 先跑 smoketest（1 chunk）验证 mask 结构 |
| **sMAPE 除零** | 零流量节点除零问题 | 需确认代码中 sMAPE 处理逻辑 |
| **zero_fill MAPE=100%** | 零填充在流量数据上 MAPE 恒定 100% | 在论文中作为 naive baseline 呈现，不追求优化 |

---

## 七、当前输出文件完整性检查

| 输出目录 | 缺失集 | 插补集 | 掩码 | 汇总 | 图表 | 审计 |
|----------|--------|--------|------|------|------|------|
| eal_data_missingness_experiments/ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| ...full_intersection_causal_history/ | ✅ | 部分 | ✅ | ❌(仅node_flow_group) | ❌ | ❌ |
| ...full_intersection_causal_history_hybridtest/ | ❌ | ✅ | ❌ | ✅ | ❌ | ✅ |
| ...full_intersection_causal_history_hybridtest_small/ | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ |
| ...full_intersection_causal_history_smoketest/ | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ |
| ...missingness_experiments_geo_func/ | — | — | — | ❌ | ❌ | ❌ |
| ...missingness_experiments_medium/ | — | — | — | ❌ | ❌ | ❌ |
| ...missingness_experiments_sample/ | — | — | — | ❌ | ❌ | ❌ |

---

## 八、与论文其他模块的关系

- **仿真实验 (Simulation)**：仿真中有一系列 FedAvg 增强聚合实验，真实数据缺失实验应该与之对齐——即在补齐缺失数据后，用同样的联邦训练流程评估预测效果（这也是 eal_data_prediction_pipeline_next_steps_zh.md 中第7步的内容）
- **论文主干**：缺失实验是论文中「数据质量」→「插补方法对比」→「最终联邦预测」三点链的第二个环节
- **LaTeX 主体**：paper_revision/latex_source/main.tex 中需要补充真实数据缺失实验的 subsection

---

## 九、预期交付物

完成后预期产出：

| 类别 | 文件 |
|------|------|
| **汇总表** | imputation_quality_summary_exclude_warmup.csv (per experiment) |
| **Mean±Std表** | imputation_quality_mean_std_table.md (cross-experiment) |
| **RMSE图** | mse_vs_missing_rate.pdf (6 methods × error bars) |
| **MAE图** | mae_vs_missing_rate.pdf |
| **sMAPE图** | smape_vs_missing_rate.pdf |
| **流程图组图** | mse_by_flow_group.pdf (low/mid/high × methods) |
| **时序对比** | imputation_before_after_sample.pdf (representative nodes) |
| **MCAR vs Block对比** | mcar_vs_block_rmse_comparison.pdf |
| **论文草稿** | 更新 eal_data_missingness_experiment_design_zh.md 和 eal_data_missingness_experiment_results_zh.md |

