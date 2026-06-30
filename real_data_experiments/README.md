# 真实数据实验文档入口

> 最后更新：2026-06-30

---

## 当前权威文档

| # | 文档 | 说明 |
|---|------|------|
| 1 | [real_exp_1_6_current_status_and_revision_plan_zh.md](real_exp_1_6_current_status_and_revision_plan_zh.md) | 实验 1–6 当前状态与修订计划 |
| 2 | [real_exp_1_6_hyperparameter_tables_zh.md](real_exp_1_6_hyperparameter_tables_zh.md) | 实验 1–6 超参数与可复现设置表 |
| 3 | [real_exp_1_6_result_table_plan_zh.md](real_exp_1_6_result_table_plan_zh.md) | 实验 1–6 论文结果表重构建议 |
| 4 | [real_exp_5_6_training_failure_diagnosis_zh.md](real_exp_5_6_training_failure_diagnosis_zh.md) | 实验 5/6 训练失效诊断报告 |
| 5 | [reviewer_response_experiment_mapping_zh.md](reviewer_response_experiment_mapping_zh.md) | 一审意见与实验 1–6 修改映射表 |

---

## 历史归档

| # | 文档 | 说明 |
|---|------|------|
| 1 | [real_exp_1_6_legacy_reports_archive_zh.md](real_exp_1_6_legacy_reports_archive_zh.md) | 实验 1–6 历史报告归档（smoke 状态、formal 记录、清理历史） |
| 2 | [real_exp_diagnostics_archive_zh.md](real_exp_diagnostics_archive_zh.md) | 真实数据实验诊断报告归档（client similarity、calendar periodicity、legacy model、anomaly 等） |

---

## 辅助文档

| # | 文档 | 说明 |
|---|------|------|
| 1 | [cuda_environment_verification_report_zh.md](cuda_environment_verification_report_zh.md) | CUDA 环境验证报告 |
| 2 | [experiment_runtime_estimate_zh.md](experiment_runtime_estimate_zh.md) | 实验运行时间估算 |
| 3 | [md_document_cleanup_report_zh.md](md_document_cleanup_report_zh.md) | 本次 Markdown 文档清理报告 |

---

## 实验子模块

| 模块 | 目录 | README |
|------|------|--------|
| 实验 1 | `single_intersection_client/` | [README_zh.md](single_intersection_client/README_zh.md) |
| 实验 2 | `single_intersection_ablation/` | [README_zh.md](single_intersection_ablation/README_zh.md) |
| 实验 3 | `region_client_full_cells/rfc_core.py` | 多相似 grid cells 主实验 |
| 实验 4 | `region_client_full_cells/rfc_ablation_core.py` + `rfc_ablation_config.py` | 多相似 grid cells 消融实验 |
| 实验 5 | `region_client/` | [README_zh.md](region_client/README_zh.md) |
| 实验 6 | `region_ablation/` | [README_zh.md](region_ablation/README_zh.md) |
| 通用模块 | `common/` | [README_zh.md](common/README_zh.md) |

---

## 维护规则

1. **当前状态以总控文档为准**：[real_exp_1_6_current_status_and_revision_plan_zh.md](real_exp_1_6_current_status_and_revision_plan_zh.md)
2. 历史文档只作溯源，不作为当前状态依据
3. 每轮实验完成后更新总控文档，过期状态报告移入 archive
4. 临时 Trae 指令、调试日志、命令输出不得长期保留在此目录
5. results/logs 目录下的文件不提交到 Git
