# 当前真实数据缺失实验状态审计报告

## 1. 审计时间与环境

- 审计时间：2026-06-14T23:57:11
- Python：`E:\anaconda3\envs\analysis\python.exe`
- Python 版本：`3.9.23`
- pandas：`2.3.3`
- pyarrow：`21.0.0`

## 2. 项目路径

- `E:\Jupter_Notebook\FedTrafficFlow`

## 3. Git 状态

- 当前分支：`main`
- `git status --short`：

```text
M analysis_scripts/full_intersection_missingness_pipeline.py
 M analysis_scripts/inventory_real_missingness_assets.py
 M results/real_data_missingness_full_intersection_causal_history/manifests/completion_check_before_repair.csv
 M results/real_data_missingness_full_intersection_causal_history/manifests/imputation_runs.csv
 M results/real_data_missingness_full_intersection_causal_history/manifests/impute_chunk_status.csv
 M results/real_data_missingness_full_intersection_causal_history/manifests/impute_stage_summary.csv
 M results/real_data_missingness_full_intersection_causal_history/summaries/imputation_quality_detail.csv
 M results/real_data_missingness_inventory/current_missingness_state_audit.json
 M results/real_data_missingness_inventory/current_missingness_state_audit.md
 M results/real_data_missingness_inventory/current_missingness_workflow_summary_zh.md
 M results/real_data_missingness_inventory/inventory_code_files.csv
 M results/real_data_missingness_inventory/inventory_document_files.csv
 M results/real_data_missingness_inventory/inventory_result_directories.csv
 M results/real_data_missingness_inventory/inventory_run_configs.csv
 M results/real_data_missingness_inventory/inventory_summary_tables.csv
 M results/real_data_missingness_inventory/inventory_visualization_files.csv
 M results/real_data_missingness_inventory/missingness_experiment_run_matrix.csv
 M results/real_data_missingness_inventory/open_issues_and_next_steps_zh.md
?? analysis_scripts/rebuild_full_missingness_manifests.py
?? results/real_data_missingness_full_intersection_causal_history/figures/
?? results/real_data_missingness_full_intersection_causal_history/full_intersection_missingness_audit.json
?? results/real_data_missingness_full_intersection_causal_history/full_intersection_missingness_audit.md
?? results/real_data_missingness_full_intersection_causal_history/full_intersection_missingness_validation.json
?? results/real_data_missingness_full_intersection_causal_history/full_intersection_missingness_validation.md
?? results/real_data_missingness_full_intersection_causal_history/summaries/batch_processing_report.json
?? results/real_data_missingness_full_intersection_causal_history/summaries/batch_processing_report.md
?? results/real_data_missingness_full_intersection_causal_history/summaries/imputation_quality_by_flow_group.csv
?? results/real_data_missingness_full_intersection_causal_history/summaries/imputation_quality_summary_all_days.csv
?? results/real_data_missingness_full_intersection_causal_history/summaries/imputation_quality_summary_exclude_warmup.csv
```

## 4. Python 环境

- 解释器：`E:\anaconda3\envs\analysis\python.exe`
- 版本：`3.9.23`

## 5. 发现的代码文件

- `analysis_scripts/audit_missingness_mechanism.py`：缺失机制审计脚本
- `analysis_scripts/audit_real_data_preprocessing.py`：真实数据预处理资产审计脚本
- `analysis_scripts/check_full_missingness_completion.py`：缺失或插补相关脚本
- `analysis_scripts/check_spatial_node_completeness.py`：未找到明确证据，需要人工确认
- `analysis_scripts/compare_date_type_curve_methods.py`：函数曲线或形态建模相关脚本
- `analysis_scripts/compare_node_flow_fourier_orders.py`：未找到明确证据，需要人工确认
- `analysis_scripts/compute_greenshields_density.py`：未找到明确证据，需要人工确认
- `analysis_scripts/compute_node_intersection_flow_optimized.py`：未找到明确证据，需要人工确认
- `analysis_scripts/fit_node_flow_daily_curve.py`：函数曲线或形态建模相关脚本
- `analysis_scripts/full_intersection_missingness_pipeline.py`：完整路口真实数据缺失构造、历史因果补全、验证与汇总流水线
- `analysis_scripts/inventory_real_missingness_assets.py`：缺失或插补相关脚本
- `analysis_scripts/real_data_missingness_experiment.py`：早期样本级真实数据人工缺失注入与简单插补评估脚本
- `analysis_scripts/rebuild_full_missingness_manifests.py`：缺失或插补相关脚本
- `analysis_scripts/visualize_fitted_function_clusters.py`：函数曲线或形态建模相关脚本
- `analysis_scripts/visualize_node_flow_daily_curve_fit.py`：函数曲线或形态建模相关脚本
- `dataset_inspection_scripts/inspect_node_intersection_flow.py`：检查或审计相关脚本
- `dataset_inspection_scripts/inspect_road_directionality.py`：检查或审计相关脚本
- `preprocessing_scripts/process_link_gps.py`：未找到明确证据，需要人工确认
- `preprocessing_scripts/process_rnsd.py`：未找到明确证据，需要人工确认

## 6. 发现的文档文件

- `MISSINGNESS_OPTIMIZATION_PLAN.md`：实验设计
- `MISSINGNESS_PHASE_RUNBOOK.md`：实验结果
- `PROJECT_DOCUMENTATION.md`：实验设计
- `PROJECT_PROGRESS_ANALYSIS.md`：实验设计
- `README.md`：实验结果
- `docs/date_type_curve_method_comparison.md`：实验结果
- `docs/environment_setup.md`：实验结果
- `docs/function_cluster_visualization.md`：实验结果
- `docs/greenshields_speed_density_scheme.md`：实验设计
- `docs/node_flow_daily_curve_fit.md`：实验设计
- `docs/node_intersection_flow_inspection.md`：实验结果
- `docs/project_documentation.md`：实验结果
- `docs/project_pipeline.md`：实验结果
- `docs/simulation_experiment_effect_report.md`：实验设计
- `docs/simulation_experiments.md`：实验结果
- `paper_revision/00_NATURE_SKILLS_PRECHECK.md`：实验设计
- `paper_revision/01_prerequisite_constraints.md`：实验设计
- `paper_revision/02_revision_strategy.md`：实验结果
- `paper_revision/04_SIMULATION_EXPERIMENT_OPTIMIZATION_SCOPE.md`：实验设计
- `paper_revision/broken_outputs/simulation_experiment_evidence_table_zh.md`：未找到明确证据，需要人工确认

## 7. 发现的结果目录

- `results/real_data_missingness_experiments`：exists=True，scope=未找到明确证据，需要人工确认，status=summaries_present_without_validation
- `results/real_data_missingness_experiments_geo_func`：exists=True，scope=geo_function_subexperiment，status=summaries_present_without_validation
- `results/real_data_missingness_experiments_medium`：exists=True，scope=medium，status=summaries_present_without_validation
- `results/real_data_missingness_experiments_sample`：exists=True，scope=sample，status=summaries_present_without_validation
- `results/real_data_missingness_full_intersection`：exists=False，scope=full_intersection，status=missing
- `results/real_data_missingness_full_intersection_causal_history`：exists=True，scope=61_chunk_main_directory，status=validated_and_summarized
- `results/real_data_missingness_full_intersection_causal_history/historical_test`：exists=True，scope=8_chunk_historical_test，status=validated_and_summarized
- `results/real_data_missingness_full_intersection_causal_history/smoke_test`：exists=True，scope=1_chunk_smoke_test，status=validated_and_summarized
- `results/real_data_missingness_full_intersection_causal_history_hybridtest`：exists=True，scope=hybridtest，status=validated_and_summarized
- `results/real_data_missingness_full_intersection_causal_history_hybridtest_small`：exists=True，scope=hybridtest_small，status=validated_and_summarized
- `results/real_data_missingness_full_intersection_causal_history_smoketest`：exists=True，scope=1_chunk_smoke_test，status=summaries_present_without_validation
- `results/real_data_missingness_mechanism_audit`：exists=False，scope=未找到明确证据，需要人工确认，status=missing
- `results/real_data_preprocessing`：exists=True，scope=preprocessing_audit，status=directory_present_no_complete_stage_evidence

## 8. 发现的数据资产

- `data/analysis/node_intersection_flow_parquet`：exists=True，rows=246133536，columns=["节点ID", "时间段", "路口进入流量", "路口离开流量", "路口车流量"]
- `data/processed/rnsd_processed.csv`：exists=True，rows=45148，columns=["路段ID", "宽度", "方向", "起始节点ID", "结束节点ID", "长度", "速度等级", "车道数", "start_lat", "start_lon", "end_lat", "end_lon"]

## 9. 发现的可视化文件

- `results/real_data_missingness_experiments/figures/missing_rate_vs_imputation_rmse.pdf`：plot_type=overall，single_rate=False，paper_ready=True
- `results/real_data_missingness_experiments/figures/missing_rate_vs_imputation_rmse.png`：plot_type=overall，single_rate=False，paper_ready=True
- `results/real_data_missingness_experiments_sample/figures/missing_rate_vs_imputation_rmse.pdf`：plot_type=overall，single_rate=False，paper_ready=False
- `results/real_data_missingness_experiments_sample/figures/missing_rate_vs_imputation_rmse.png`：plot_type=overall，single_rate=False，paper_ready=False
- `results/real_data_missingness_experiments_sample/figures/missing_rate_vs_imputation_rmse_log_y.png`：plot_type=overall，single_rate=False，paper_ready=False
- `results/real_data_missingness_full_intersection_causal_history/figures/single_rate_0p05_delta_rmse_relative_to_forward_fill.pdf`：plot_type=delta，single_rate=True，paper_ready=True
- `results/real_data_missingness_full_intersection_causal_history/figures/single_rate_0p05_delta_rmse_relative_to_forward_fill.png`：plot_type=delta，single_rate=True，paper_ready=True
- `results/real_data_missingness_full_intersection_causal_history/figures/single_rate_0p05_flow_group_rmse.pdf`：plot_type=other，single_rate=True，paper_ready=True
- `results/real_data_missingness_full_intersection_causal_history/figures/single_rate_0p05_flow_group_rmse.png`：plot_type=other，single_rate=True，paper_ready=True
- `results/real_data_missingness_full_intersection_causal_history/figures/single_rate_0p05_rmse_by_method.pdf`：plot_type=other，single_rate=True，paper_ready=True
- `results/real_data_missingness_full_intersection_causal_history/figures/single_rate_0p05_rmse_by_method.png`：plot_type=other，single_rate=True，paper_ready=True
- `results/real_data_missingness_full_intersection_causal_history/historical_test/figures/missing_rate_vs_imputation_rmse.pdf`：plot_type=overall，single_rate=True，paper_ready=False
- `results/real_data_missingness_full_intersection_causal_history/historical_test/figures/missing_rate_vs_imputation_rmse.png`：plot_type=overall，single_rate=True，paper_ready=False
- `results/real_data_missingness_full_intersection_causal_history/historical_test/figures/rmse_difference_relative_to_forward_fill.pdf`：plot_type=delta，single_rate=True，paper_ready=True
- `results/real_data_missingness_full_intersection_causal_history/historical_test/figures/rmse_difference_relative_to_forward_fill.png`：plot_type=delta，single_rate=True，paper_ready=True
- `results/real_data_missingness_full_intersection_causal_history/historical_test/figures/zoom_historical_geo_function_rmse.pdf`：plot_type=zoom，single_rate=True，paper_ready=True
- `results/real_data_missingness_full_intersection_causal_history/historical_test/figures/zoom_historical_geo_function_rmse.png`：plot_type=zoom，single_rate=True，paper_ready=True
- `results/real_data_missingness_full_intersection_causal_history/smoke_test/figures/missing_rate_vs_imputation_rmse.pdf`：plot_type=overall，single_rate=True，paper_ready=False
- `results/real_data_missingness_full_intersection_causal_history/smoke_test/figures/missing_rate_vs_imputation_rmse.png`：plot_type=overall，single_rate=True，paper_ready=False
- `results/real_data_missingness_full_intersection_causal_history/smoke_test/figures/rmse_difference_relative_to_forward_fill.pdf`：plot_type=delta，single_rate=True，paper_ready=False

## 10. 运行配置解析

- `results/real_data_missingness_experiments/run_config.json`：mechanism=["mcar_point"]，missing_rates=["0.0", "0.05", "0.1", "0.2", "0.3"]，methods=["zero_fill", "forward_fill", "linear_interpolation"]
- `results/real_data_missingness_full_intersection_causal_history/historical_test/run_config.json`：mechanism=mcar_point，missing_rates=["0.05"]，methods=["forward_fill", "historical_linear_extrapolation", "function_curve_fit", "geo_neighbor_fill", "geo_func_hybrid", "zero_fill"]
- `results/real_data_missingness_full_intersection_causal_history/run_config.json`：mechanism=mcar_point，missing_rates=["0.05"]，methods=["geo_func_hybrid"]
- `results/real_data_missingness_full_intersection_causal_history/smoke_test/run_config.json`：mechanism=mcar_point，missing_rates=["0.05"]，methods=["zero_fill", "forward_fill", "historical_linear_extrapolation", "geo_neighbor_fill", "function_curve_fit", "geo_func_hybrid"]
- `results/real_data_missingness_full_intersection_causal_history_hybridtest/run_config.json`：mechanism=mcar_point，missing_rates=["0.05"]，methods=["geo_func_hybrid"]
- `results/real_data_missingness_full_intersection_causal_history_hybridtest_small/run_config.json`：mechanism=mcar_point，missing_rates=["0.05"]，methods=["geo_func_hybrid"]
- `results/real_data_missingness_full_intersection_causal_history_smoketest/run_config.json`：mechanism=mcar_point，missing_rates=["0.05"]，methods=["forward_fill", "historical_linear_extrapolation", "geo_neighbor_fill", "function_curve_fit", "geo_func_hybrid"]

## 11. 实验矩阵

- `results/real_data_missingness_experiments`：generate=False，impute=False，summarize=True，validate=True
- `results/real_data_missingness_experiments_geo_func`：generate=False，impute=False，summarize=False，validate=False
- `results/real_data_missingness_experiments_medium`：generate=False，impute=False，summarize=False，validate=False
- `results/real_data_missingness_experiments_sample`：generate=False，impute=False，summarize=True，validate=True
- `results/real_data_missingness_full_intersection`：generate=False，impute=False，summarize=False，validate=False
- `results/real_data_missingness_full_intersection_causal_history`：generate=True，impute=True，summarize=True，validate=True
- `results/real_data_missingness_full_intersection_causal_history/historical_test`：generate=True，impute=True，summarize=True，validate=True
- `results/real_data_missingness_full_intersection_causal_history/smoke_test`：generate=True，impute=True，summarize=True，validate=True
- `results/real_data_missingness_full_intersection_causal_history_hybridtest`：generate=True，impute=True，summarize=False，validate=True
- `results/real_data_missingness_full_intersection_causal_history_hybridtest_small`：generate=True，impute=True，summarize=True，validate=True
- `results/real_data_missingness_full_intersection_causal_history_smoketest`：generate=True，impute=True，summarize=False，validate=False
- `results/real_data_missingness_mechanism_audit`：generate=False，impute=False，summarize=False，validate=False
- `results/real_data_preprocessing`：generate=False，impute=False，summarize=False，validate=False

## 12. 方法与机制矩阵

- `zero_fill`：observed=True，uses_future_data=False
- `forward_fill`：observed=True，uses_future_data=True
- `historical_linear_extrapolation`：observed=True，uses_future_data=False
- `geo_neighbor_fill`：observed=True，uses_future_data=False
- `function_curve_fit`：observed=True，uses_future_data=False
- `geo_func_hybrid`：observed=True，uses_future_data=False
- `linear_interpolation`：observed=True，uses_future_data=True

## 13. 当前完成度判断

- `historical_test`：证据完整，已完成 generate_missing、impute、summarize、validate。
- 61 chunk 主目录：存在部分或阶段性输出，但未找到完整完成证据，不能直接写“全量完成”。
- `node_temporal_block`：代码支持，未找到正式运行证据。

## 14. 不能下结论的部分

- 61 chunk 全量 generate_missing、impute、summarize、validate 是否全部完成。
- 多缺失率正式主实验是否完成。
- 多 seed 真实数据缺失实验是否完成。
- FedAvg / Independent 真实预测训练输出是否完成。

## 15. 需要人工确认的部分

- 风险关键词命中的文档语境是否需要修正。
- 61 chunk 主目录中各方法的完整 chunk 覆盖情况。
- `node_temporal_block` 是否在未纳入本次目录的外部位置运行过。

附：风险关键词命中 74 条。

