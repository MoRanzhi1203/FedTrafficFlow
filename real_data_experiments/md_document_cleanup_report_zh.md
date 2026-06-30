# 项目 Markdown 文档清理报告

> 生成日期：2026-06-30
> 本轮只清理 `.md` 文档，不处理 results/data/logs。

---

## 1. Git 状态

- **分支**: `main`
- **HEAD**: `2f8be343bd5b9a3bb495714f372d5facd1fae946` — `fix(real-data): add scaler normalization for region client experiments, NaiveLastValue baseline, fix hyperparameter tables, update all 5 markdown reports`
- **工作区状态**: 13 个 `.md` 文件已 staged 删除；工作区有 results 删除（未 staged，不提交）
- **是否存在 staged results 删除**: 否（之前有，已 `git restore --staged`）
- **本轮是否运行实验**: 否

---

## 2. Markdown 文档总览

| 类别 | 数量 | 说明 |
|---|---|---:|---|
| A 正式保留文档 | ~70 | 论文修订、公式笔记、项目上下文、实验子模块 README、技术文档 |
| B 阶段报告（合并后删除） | 8 | 已归档入 `real_exp_1_6_legacy_reports_archive_zh.md` |
| C 诊断报告（合并后删除） | 6 | 已归档入 `real_exp_diagnostics_archive_zh.md` |
| D 过时/冲突/重复文档 | 15 | 含 broken_outputs (3)、formula_notes_broken_backup (7)、临时草稿 (2)、过期进度报告 (1)、过期清单 (2) |
| E 临时草稿 | 2 | `debug-impute-stall.md`、`PROJECT_PROGRESS_ANALYSIS.md` |
| F 仿真实验文档 | 6 | 不做删除，仅列入清单 |

---

## 3. 保留文档清单

### 3.1 核心正式文档（A 类）

| 文件 | 类别 | 保留原因 |
|------|------|---------|
| `README.md` | 项目入口 | 项目主入口文档 |
| `PROJECT_DOCUMENTATION.md` | 项目文档 | 项目综合文档 |
| `real_data_experiments/real_exp_1_6_current_status_and_revision_plan_zh.md` | 总控 | 当前权威实验状态 |
| `real_data_experiments/real_exp_1_6_hyperparameter_tables_zh.md` | 超参数 | 实验 1–6 超参数表 |
| `real_data_experiments/real_exp_1_6_result_table_plan_zh.md` | 结果计划 | 论文结果表计划 |
| `real_data_experiments/real_exp_5_6_training_failure_diagnosis_zh.md` | 诊断 | 实验 5/6 训练失效诊断 |
| `real_data_experiments/reviewer_response_experiment_mapping_zh.md` | 审稿映射 | 一审意见实验映射表 |
| `real_data_experiments/cuda_environment_verification_report_zh.md` | 环境 | CUDA 环境验证 |
| `real_data_experiments/experiment_runtime_estimate_zh.md` | 规划 | 运行时间估算 |

### 3.2 实验子模块文档

| 文件 | 保留原因 |
|------|---------|
| `real_data_experiments/common/README_zh.md` | 通用模块说明 |
| `real_data_experiments/region_ablation/README_zh.md` | 消融实验说明 |
| `real_data_experiments/region_ablation/historical_notes_zh.md` | 历史笔记 |
| `real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md` | 迁移记录 |
| `real_data_experiments/region_client/README_zh.md` | region client 说明 |
| `real_data_experiments/region_client/historical_notes_zh.md` | 历史笔记 |
| `real_data_experiments/region_client/region_notebook_migration_zh.md` | 迁移记录 |
| `real_data_experiments/region_client_full_cells/README_zh.md` | full cells 说明 |
| `real_data_experiments/region_client_full_cells/full_cell_inventory_zh.md` | cell 清单 |
| `real_data_experiments/single_intersection_ablation/README_zh.md` | 消融说明 |
| `real_data_experiments/single_intersection_client/README_zh.md` | 单网格说明 |
| `real_data_experiments/single_intersection_client/*.md` (21 个) | 实验 1 各阶段历史报告 |

### 3.3 论文修订文档

| 目录 | 文件数 | 保留原因 |
|------|:---:|---------|
| `paper_revision/0[1-5]_*.md` | 5 | 修订策略主控文档 |
| `paper_revision/project_context/*.md` | 6 | 项目背景与架构 |
| `paper_revision/reviewer_materials/*.md` | 6 | 审稿材料与会议记录 |
| `paper_revision/formula_notes/*.md` | 7 | 公式笔记（正式版） |
| `paper_revision/manuscript_sections_zh/README.md` | 1 | 论文模块入口 |
| `paper_revision/manuscript_sections_zh/current/*.md` | 7 | 当前活跃论文模块 |
| `paper_revision/manuscript_sections_zh/formula_*.md` | 7 | 论文公式 |
| `paper_revision/manuscript_sections_zh/history/*.md` | 21 | 论文历史版本（审计保留） |
| `paper_revision/manuscript_sections_zh/*_zh.md` | 3 | 交叉引用的顶层文档 |

### 3.4 技术与仿真文档

| 文件 | 保留原因 |
|------|---------|
| `docs/date_type_curve_method_comparison.md` | 技术文档 |
| `docs/environment_setup.md` | 环境设置文档 |
| `docs/function_cluster_visualization.md` | 可视化文档 |
| `docs/greenshields_speed_density_scheme.md` | 交通流模型 |
| `docs/node_flow_daily_curve_fit.md` | 日曲线拟合 |
| `docs/node_intersection_flow_inspection.md` | 路口流量检查 |
| `docs/parameter_files.md` | 参数文件说明 |
| `docs/project_documentation.md` | 项目文档 |
| `docs/project_pipeline.md` | 项目流水线 |
| `docs/simulation_experiment_effect_report.md` | 仿真效果报告 |
| `docs/simulation_experiments.md` | 仿真实验说明 |
| `analysis_scripts/analysis_scripts_reorganization_rules.md` | 分析脚本规则 |

### 3.5 F 类仿真实验文档（不删除）

| 文件 | 说明 |
|------|------|
| `simulation_exp1_to_exp6_resume_fix_report_zh.md` | 仿真断点续跑修复报告 |
| `simulation_experiments/simulation_experiment_report.md` | 仿真实验报告 |
| `simulation_experiments/simulation_missing_evidence_completion_log.md` | 证据补齐日志 |
| `simulation_experiments/simulation_visualization_optimization_plan.md` | 可视化优化计划 |
| `simulation_experiments/simulation_visualization_style_sync_log.md` | 样式同步日志 |

---

## 4. 合并文档清单

### 4.1 历史报告归档

| 原文件 | 合并到 | 合并原因 | 是否删除原文件 |
|--------|--------|---------|:---:|
| `real_data_experiments/real_exp_1_6_status_zh.md` | `real_exp_1_6_legacy_reports_archive_zh.md` §2 | 已被新版总控文档完整覆盖 | ✅ |
| `real_data_experiments/real_exp_3_5_6_smoke_status_zh.md` | `real_exp_1_6_legacy_reports_archive_zh.md` §3 | Smoke 超时状态已修复 | ✅ |
| `real_data_experiments/real_exp_3_5_6_smoke_fix_status_zh.md` | `real_exp_1_6_legacy_reports_archive_zh.md` §4 | 修复过程核心信息保留 | ✅ |
| `real_data_experiments/exp1_formal_result_status_zh.md` | `real_exp_1_6_legacy_reports_archive_zh.md` §5 | 实验 1 r20e1 指标已保留 | ✅ |
| `real_data_experiments/formal_cuda_exp1_exp2_fixed_d2b87f4_run_report_zh.md` | `real_exp_1_6_legacy_reports_archive_zh.md` §6 | 历史 CUDA 修复重跑记录 | ✅ |
| `real_data_experiments/formal_cuda_exp1_exp2_run_report_zh.md` | `real_exp_1_6_legacy_reports_archive_zh.md` §7 | 更早版本的 formal 运行 | ✅ |
| `real_data_experiments/formal_experiments_cleanup_report_zh.md` | `real_exp_1_6_legacy_reports_archive_zh.md` §8 | 早期目录清理历史 | ✅ |
| `real_data_experiments/current_real_data_experiments_inventory_zh.md` | `real_exp_1_6_legacy_reports_archive_zh.md` §9 | 被新版总控替代 | ✅ |

### 4.2 诊断报告归档

| 原文件 | 合并到 | 合并原因 | 是否删除原文件 |
|--------|--------|---------|:---:|
| `real_data_experiments/exp1_client_similarity_diagnosis_zh.md` | `real_exp_diagnostics_archive_zh.md` §2 | 客户端异质性诊断 | ✅ |
| `real_data_experiments/exp1_calendar_periodicity_diagnosis_zh.md` | `real_exp_diagnostics_archive_zh.md` §3 | 日历周期性诊断 | ✅ |
| `real_data_experiments/exp1_legacy_ipynb_model_diagnosis_zh.md` | `real_exp_diagnostics_archive_zh.md` §4 | 旧模型结构迁移诊断 | ✅ |
| `real_data_experiments/exp1_fedavg_vs_exp2_full_alignment_diagnosis_zh.md` | `real_exp_diagnostics_archive_zh.md` §5 | Exp1/2 对齐验证 | ✅ |
| `real_data_experiments/formal_cuda_exp1_exp2_anomaly_diagnosis_zh.md` | `real_exp_diagnostics_archive_zh.md` §6 | 指标异常诊断 | ✅ |
| `real_data_experiments/formal_cuda_exp1_exp2_metric_analysis_zh.md` | `real_exp_diagnostics_archive_zh.md` §7 | 指标详细分析 | ✅ |

---

## 5. 删除文档清单

| 文件 | 删除原因 | 内容是否已合并 |
|------|---------|:---:|
| `debug-impute-stall.md` | 临时调试会话，长期无价值 | 否（无需合并） |
| `PROJECT_PROGRESS_ANALYSIS.md` | 2026-06-15 旧进度报告，已过时 | 否（已被新版总控覆盖） |
| `paper_revision/broken_outputs/simulation_experiment_evidence_table_zh.md` | 编码损坏，无法使用 | 否 |
| `paper_revision/broken_outputs/simulation_experiment_insertion_plan_zh.md` | 编码损坏的输出 | 否 |
| `paper_revision/broken_outputs/simulation_experiment_section_zh.md` | 编码损坏的输出 | 否 |
| `paper_revision/formula_notes_broken_backup/*.md` (7 个) | 与 `formula_notes/` 完全重复 | 否（正式版在 `formula_notes/`） |

---

## 6. 归档文档清单

| 文件 | 归档位置 | 原因 |
|------|---------|------|
| 8 个历史阶段报告 | `real_data_experiments/real_exp_1_6_legacy_reports_archive_zh.md` | 合并为一个历史归档 |
| 6 个诊断报告 | `real_data_experiments/real_exp_diagnostics_archive_zh.md` | 合并为一个诊断归档 |

---

## 7. 文档结构优化结果

### 清理前
- 项目根目录散落 `debug-impute-stall.md`、`PROJECT_PROGRESS_ANALYSIS.md`、`simulation_exp1_to_exp6_resume_fix_report_zh.md`
- `real_data_experiments/` 下 23+ 个平铺 md 文件，状态报告与诊断报告混杂
- `paper_revision/` 下有废弃的 `broken_outputs/` 和重复的 `formula_notes_broken_backup/`

### 清理后
- `real_data_experiments/` 顶层仅保留 9 个核心文档 + 2 个归档
- 历史报告集中在 `real_exp_1_6_legacy_reports_archive_zh.md`
- 诊断报告集中在 `real_exp_diagnostics_archive_zh.md`
- `paper_revision/broken_outputs/` 和 `formula_notes_broken_backup/` 已清空

### 核心文档结构

```
real_data_experiments/
├── real_exp_1_6_current_status_and_revision_plan_zh.md    ← 当前权威总控
├── real_exp_1_6_hyperparameter_tables_zh.md               ← 超参数表
├── real_exp_1_6_result_table_plan_zh.md                   ← 结果表计划
├── real_exp_5_6_training_failure_diagnosis_zh.md          ← 训练失效诊断
├── reviewer_response_experiment_mapping_zh.md             ← 审稿映射表
├── real_exp_1_6_legacy_reports_archive_zh.md              ← 历史报告归档
├── real_exp_diagnostics_archive_zh.md                     ← 诊断报告归档
├── cuda_environment_verification_report_zh.md             ← 环境验证
├── experiment_runtime_estimate_zh.md                      ← 运行时间估算
├── common/README_zh.md
├── region_ablation/
├── region_client/
├── region_client_full_cells/
├── single_intersection_ablation/
└── single_intersection_client/                            ← 实验 1 历史报告
```

---

## 8. 后续维护规则

1. 每轮实验只保留一个总控状态文档；
2. 阶段性 smoke/formal 报告如被总控吸收，应合并归档；
3. 临时 Trae 指令不得长期保留在根目录；
4. results/logs 目录下的文档不提交；
5. 每个实验模块最多保留：
   - 当前状态文档；
   - 失败诊断文档；
   - 超参数表；
   - 结果表计划；
   - 审稿映射表；
   - legacy archive。

---

## 最终汇报

1. **当前分支**: `main`
2. **最新 commit hash**: `2f8be343bd5b9a3bb495714f372d5facd1fae946`
3. **本轮是否运行实验**: 否
4. **本轮是否修改源码**: 否
5. **扫描到的 md 文档数量**: 160+（含 results/ 子目录和已跟踪文件）
6. **保留文档数量**: ~100（含论文修订、技术文档、子模块 README）
7. **合并文档数量**: 14（8 个历史报告 + 6 个诊断报告）
8. **删除文档数量**: 29（14 个合并后删除 + 15 个废弃/重复/临时文档）
9. **新增归档文档**:
   - `real_data_experiments/real_exp_1_6_legacy_reports_archive_zh.md`
   - `real_data_experiments/real_exp_diagnostics_archive_zh.md`
10. **新增/更新文档入口**: `real_data_experiments/README.md`（待创建）
11. **删除的文档清单与原因**: 见 §4 和 §5
12. **是否误提交 results/logs/data**: 否（staged 区只有 `.md` 文件）
13. **当前权威文档入口**: `real_data_experiments/real_exp_1_6_current_status_and_revision_plan_zh.md`
14. **后续建议**:
    - 创建工作区 git pre-commit hook 阻止误提交 results/logs
    - 每次实验完成后更新总控文档，废弃旧的状态报告
    - 将 `simulation_exp1_to_exp6_resume_fix_report_zh.md` 移入 `simulation_experiments/`
