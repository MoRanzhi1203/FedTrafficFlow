# 真实数据实验当前有效文档索引

> 最后更新：2026-06-30
> 当前基准 commit：e91e2cd — feat(real-data): add Exp1 federated mechanism diagnostic
> 当前分支：feature/real-exp4-rfc-ablation

## 1. 当前可信主文档

| 文档 | 用途 | 状态 |
|---|---|---|
| real_exp_1_6_current_status_and_revision_plan_zh.md | 实验1-6当前状态与修订计划 | 当前有效 |
| real_exp_1_6_calendar_holiday_handling_check_zh.md | 日历/节假日处理检查报告 | 当前有效 |
| real_exp_1_6_hyperparameter_tables_zh.md | 超参数表 | 当前有效 |
| reviewer_response_experiment_mapping_zh.md | 审稿意见-实验映射 | 当前有效 |
| real_exp_1_6_result_table_plan_zh.md | 论文结果表计划 | 当前有效 |

## 2. 当前可信结果汇总

| 文档 | 结果类型 | 是否 formal | 是否可写入主文 |
|---|---|---|---|
| exp1_federated_mechanism_advantage_summary_zh.md | Exp1 federated mechanism diagnostic | 否（diagnostic） | 否（仅参考） |
| exp1_long_horizon_diagnostic_summary_zh.md | Exp1 long-horizon diagnostic | 否（diagnostic） | 否（仅参考） |

## 3. 历史追溯文档

| 文档 | 历史用途 | 注意事项 |
|---|---|---|
| archive_legacy_docs/real_exp_5_6_training_failure_diagnosis_zh.md | Exp5/6 scaler 缺失根因诊断 | 已修复；代表历史状态 |
| real_exp_diagnostics_archive_zh.md | 诊断历史归档 | 不含当前结果 |
| real_exp_1_6_legacy_reports_archive_zh.md | 旧报告归档索引 | 不含当前结果 |

## 4. 已删除文档

| 文档 | 处理方式 | 原因 |
|---|---|---|
| real_exp_1_6_status_zh.md | 已删除 | 严重过时，被 current_status 替代 |
| real_exp_1_6_existence_check_zh.md | 已删除 | 已过期，不再需要 |
| md_document_cleanup_report_zh.md | 已删除 | 旧清理报告，已被本轮报告替代 |
| cuda_environment_verification_report_zh.md | 已删除 | 过时的环境检查 |
| experiment_runtime_estimate_zh.md | 已删除 | 过时运行时间估算 |

## 5. 当前统一口径

### 5.1 数据划分
当前修订默认：80/10/10 chronological split。
历史 Exp1 formal r20e1 使用 70/15/15，只作为 sensitivity check。
不使用随机 split。

### 5.2 Exp1 状态
Level 2 diagnostic。包含：
- 历史 formal r20e1 (70/15/15) 作为 sensitivity check
- CalendarFeatureFedAvg v2 residual-gated diagnostic
- Long-horizon seq96_h4/h12/h24 diagnostic
- Federated mechanism seq96_h12 r5 diagnostic

### 5.3 Calendar/holiday 状态
calendar/holiday 目前只在 Exp1 的 CalendarFeatureFedAvg v2 diagnostic 和 federated mechanism diagnostic 中进入神经网络辅助分支；Exp3/Exp5 仅作为 CalendarProfileNaive baseline；Exp2/4/6 不使用 calendar 特征。

### 5.4 联邦机制状态
当前重点是 FedAvg shared initialization、FedProx non-IID regularization、Local fine-tuning personalization、CentralizedUpperBound as oracle。

### 5.5 Formal/Diagnostic
r5/r1/smoke/long-horizon/mechanism eval 均为 diagnostic，不能写 formal。只有明确 r20 formal 且结果目录存在才写 formal。

## 6. 禁止使用的旧结论

- "Exp1 日历特征进入模型输入：否"
- "calendar/holiday 是否进入神经网络输入：否"
- "所有实验均未使用 calendar 特征进入神经网络"
- "裸 FedAvg 已经全面超过 Independent"
- "本轮不运行实验 / 仅静态分析"

## 7. 下一步建议
1. 修 grouped_metrics 覆盖范围
2. 跑 seq96_h12 r5 quick rerun 验证一致性
3. 通过后进入 r10 diagnostic
