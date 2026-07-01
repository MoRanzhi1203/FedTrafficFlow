# 文档清理报告

> 生成日期：2026-07-01
> 清理范围：`real_data_experiments/` 目录下全部 .md 文件
> 当前基准 commit：e91e2cd

## 1. 删除的文件（4 份）

| 文件名 | 删除原因 |
|--------|---------|
| `real_exp_1_6_existence_check_zh.md` | 过时的存在性检查，不再需要 |
| `md_document_cleanup_report_zh.md` | 旧清理报告，已被本轮报告替代 |
| `cuda_environment_verification_report_zh.md` | 过时的 CUDA 环境检查 |
| `experiment_runtime_estimate_zh.md` | 过时的运行时间估算 |

## 2. 移动并归档的文件（1 份）

| 原路径 | 新路径 | 处理 |
|--------|--------|------|
| `real_exp_5_6_training_failure_diagnosis_zh.md` | `archive_legacy_docs/real_exp_5_6_training_failure_diagnosis_zh.md` | 移至历史归档目录，顶部添加历史文档声明（Exp5/6 scaler 问题已在 e91e2cd 前的提交中修复） |

## 3. 重写的文件（2 份）

| 文件名 | 主要变更 |
|--------|---------|
| `real_exp_1_6_calendar_holiday_handling_check_zh.md` | HEAD 更新为 e91e2cd；修正 calendar 进入神经网络输入结论（从"否"改为"是，仅 Exp1 CalendarFeatureFedAvg v2 diagnostic + federated mechanism diagnostic"）；新增 grouped_metrics_by_calendar 说明；更新"不能写"列表 |
| `real_exp_1_6_current_status_and_revision_plan_zh.md` | HEAD 更新为 e91e2cd；删除"本轮不运行实验/仅静态分析"声明；补充 Exp1 federated mechanism diagnostic 结果与机制结论；Exp5/6 标注 scaler 已修复（不再写"训练失效"）；更新下一步优先级；添加 FedProx 分析任务 |

## 4. 新建的文件（2 份）

| 文件名 | 用途 |
|--------|------|
| `REAL_DATA_EXPERIMENTS_CURRENT_DOCS_zh.md` | 当前有效文档索引，统一口径，禁止使用的旧结论列表 |
| `docs_cleanup_report_zh.md` | 本文档，记录本轮清理的完整操作 |

## 5. 添加历史声明的文件（1 份）

| 文件名 | 处理 |
|--------|------|
| `real_exp_diagnostics_archive_zh.md` | 顶部添加历史文档声明 banner |

## 6. 检查后无需修改的文件（2 份）

| 文件名 | 检查结果 |
|--------|---------|
| `reviewer_response_experiment_mapping_zh.md` | 无 stale HEAD 引用，无需修改 |
| `real_exp_1_6_result_table_plan_zh.md` | 无 stale HEAD 引用，无需修改 |

## 7. 验证

- Python 3.9 环境 (`E:\anaconda3\envs\FedTrafficFlow\python.exe`) 通过 `py_compile` 验证
- 本次文档清理操作未修改任何 .py 文件
- 本次文档清理操作未运行任何实验
- 未提交任何 results/logs/data
- 归档目录 `archive_legacy_docs/` 已创建
