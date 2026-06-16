# 缺失值设置与补全内容清理报告

## 1. 清理目的

移除项目中所有缺失值设置、缺失注入、mask 生成、miss_data、imp_data、插补补全、插补评估和相关运行计划/报告/图件内容，同时保留真实数据预处理、交通流预测主任务、FedAvg / Independent 与仿真实验代码。

## 2. 已停止相关进程

- 已发现并停止运行中的 missingness 相关生成任务。
- 本轮不再继续任何 `generate_missing`、`impute` 或缺失实验审计工作。

## 3. 已删除脚本

- `analysis_scripts/full_intersection_missingness_pipeline.py`
- `analysis_scripts/real_data_missingness_experiment.py`
- `analysis_scripts/audit_missing_rate_scope.py`
- `analysis_scripts/audit_global_mcar_missingness.py`
- `analysis_scripts/check_full_missingness_completion.py`
- `analysis_scripts/rebuild_full_missingness_manifests.py`
- `analysis_scripts/inventory_real_missingness_assets.py`
- `analysis_scripts/inventory_missingness_cleanup_targets.py`

## 4. 已删除结果目录

- `results/real_data_missingness_experiments`
- `results/real_data_missingness_experiments_sample`
- `results/real_data_missingness_full_intersection_causal_history`
- `results/real_data_missingness_full_intersection_causal_history_global_mcar`
- `results/real_data_missingness_full_intersection_causal_history_hybridtest`
- `results/real_data_missingness_full_intersection_causal_history_hybridtest_small`
- `results/real_data_missingness_full_intersection_causal_history_smoketest`
- `results/real_data_missingness_inventory`

## 5. 已删除文档与报告

- `MISSINGNESS_OPTIMIZATION_PLAN.md`
- `MISSINGNESS_PHASE_RUNBOOK.md`
- `results/code_execution_status_report_zh.md`
- `paper_revision/manuscript_sections_zh/real_data_missingness_experiment_design_zh.md`
- `paper_revision/manuscript_sections_zh/real_data_missingness_experiment_results_zh.md`
- `paper_revision/manuscript_sections_zh/current/real_data_missingness_experiment_design_zh.md`
- `paper_revision/manuscript_sections_zh/current/real_data_missingness_experiment_results_zh.md`

## 6. 已编辑文档

- `PROJECT_PROGRESS_ANALYSIS.md`
  - 删除缺失注入、插补实验、missingness 结果与相关提交记录，仅保留真实数据预处理、仿真实验和真实预测主任务。
- `paper_revision/manuscript_sections_zh/real_data_prediction_pipeline_next_steps_zh.md`
  - 删除缺失率鲁棒性与插补相关描述，仅保留真实数据联邦预测后续步骤。
- `paper_revision/manuscript_sections_zh/current/real_data_prediction_pipeline_next_steps_zh.md`
  - 删除缺失率鲁棒性与插补相关描述，仅保留真实数据联邦预测后续步骤。
- `paper_revision/manuscript_sections_zh/README.md`
  - 删除对已清理 missingness 章节的目录说明。
- `.gitignore`
  - 删除过度具体的 missingness 结果目录忽略规则，保留通用大文件保护与现有项目规则。

## 7. 明确保留内容

- 原始真实数据与 `data/raw/`
- `data/analysis/node_intersection_flow_parquet`
- `data/processed/rnsd_processed.csv`
- 真实数据预处理脚本
- 交通流预测模型与真实预测准备文档
- `simulation_experiments/` 下的 FedAvg / Independent / 联邦仿真实验代码
- 普通数据清洗、数据审计与真实数据预处理审计内容

## 8. 允许残留项

- `results/cleanup_missingness_removal_plan.csv`
- `results/cleanup_missingness_removal_plan_zh.md`
- 本报告与其 JSON 版本
- `analysis_scripts/audit_real_data_preprocessing.py` 中的 `missing_rate_per_column`
- `results/real_data_preprocessing/real_data_preprocessing_audit.json` 中的 `missing_rate_per_column`

以上残留不属于缺失实验主流程，而是普通预处理数据质量检查或本轮清理记录。

## 9. 未删除原因

- `analysis_scripts/audit_real_data_preprocessing.py`
  - 保留原因：真实数据预处理审计脚本，不属于 missingness / imp 实验流程。
- `results/real_data_preprocessing/real_data_preprocessing_audit.json`
  - 保留原因：真实数据预处理审计结果，`missing_rate_per_column` 仅表示普通列缺失率统计。

## 10. 清理验证结论

- missingness / imp 主流程脚本已移除。
- `MCAR`、`mask_scope`、`generate_missing`、`impute` 等实验入口已从项目主流程中移除。
- `zero_fill`、`forward_fill`、`geo_neighbor_fill`、`function_curve_fit`、`geo_func_hybrid` 等插补结果目录已删除。
- 全局搜索后允许的剩余关键词命中仅来自清理计划、清理报告以及普通预处理质量统计。
