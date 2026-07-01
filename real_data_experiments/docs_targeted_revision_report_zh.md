# 真实数据实验文档定点修正报告

> 生成日期：2026-07-01
> 范围：real_data_experiments/ 下当前有效 Markdown 文档
> 说明：本次只修改文档，不修改代码，不运行实验

## 1. 本次修改文件

| 文件 | 修改原因 | 修改内容 |
|---|---|---|
| `reviewer_response_experiment_mapping_zh.md` | 旧口径：CalendarFeatureFedAvg 尚未实现、calendar baseline 不足、formal 结果可信度笼统 | 修正 CalendarFeatureFedAvg v2 口径；修正 Exp5 scaler 修复后状态；细化论文 formal 结果可信度逐实验说明；新增"当前禁止写入论文的旧结论"节 |
| `real_exp_1_6_calendar_holiday_handling_check_zh.md` | formal/diagnostic 混写（"formal: diagnostic已运行"）、Exp5 小节前后矛盾（先写"calendar baseline:无"又写"Level 1"）、grouped_metrics 描述不精确 | 修正 Git 状态 formal→diagnostic 分离；修正 Exp5 小节删除"calendar baseline：无"改为"CalendarProfileNaive 已新增"；修正 grouped_metrics_by_calendar 中的 CalendarProfileNaive 描述；新增3条不可写结论 |
| `real_exp_1_6_hyperparameter_tables_zh.md` | 未区分 formal 默认参数与 diagnostic 参数、归一化状态写为全✅但旧 formal 不自动可信、70/15/15 未区分历史/当前 | 拆分 input_length/prediction_horizon 为"主 formal 默认"与"diagnostic"两档；数据划分表区分历史已运行与当前推荐、标注旧 formal 不可直接使用；修正归一化状态表述；补充 convergence 不可写 formal 的风险提示 |
| `exp1_federated_mechanism_advantage_summary_zh.md` | RMSE 解释错误：把 FedAvg(82,523)<NaiveLastValue(94,259) 写成"弱于"（RMSE越低越好） | 修正第6条核心发现为"FedAvg 裸模型优于 NaiveLastValue，但仍弱于 Independent 和周期性基线"；新增"写作边界"节 |
| `exp1_long_horizon_diagnostic_summary_zh.md` | 缺少 seasonal baseline horizon 风险提示和后续检查任务 | 新增"解释风险与待核查项"节（4项）和"后续检查任务"节（4项） |
| `docs_cleanup_report_zh.md` | "未修改任何 .py 文件"与前序代码提交产生表面矛盾 | 开头新增范围说明；验证部分改为"本次文档清理操作未修改/未运行" |
| `REAL_DATA_EXPERIMENTS_CURRENT_DOCS_zh.md` | 缺少本次修正条目、禁止旧结论列表不全 | 补全可信主文档/结果汇总状态说明；禁止旧结论列表新增7条 |

## 2. 修正的关键问题

- reviewer_response_experiment_mapping_zh.md 中 CalendarFeatureFedAvg v2 旧口径已修正；
- real_exp_1_6_calendar_holiday_handling_check_zh.md 中 Exp5 calendar baseline 自相矛盾已修正（删除"calendar baseline：无"，统一为 Level 1 + CalendarProfileNaive）；
- real_exp_1_6_hyperparameter_tables_zh.md 已区分主 formal 默认参数与 Exp1 long-horizon/mechanism diagnostic 参数；
- exp1_federated_mechanism_advantage_summary_zh.md 已修正 RMSE 越低越好的解释错误；
- exp1_long_horizon_diagnostic_summary_zh.md 已补充 seasonal baseline horizon 风险提示；
- docs_cleanup_report_zh.md 已补充范围限定，避免与前序代码/实验操作冲突；
- REAL_DATA_EXPERIMENTS_CURRENT_DOCS_zh.md 已补充禁止旧结论。

## 3. 当前统一口径

| 项目 | 当前口径 |
|---|---|
| 数据划分 | 当前修订默认 80/10/10 chronological split；历史 70/15/15 仅 sensitivity check |
| formal/diagnostic | r5/r1/smoke/long-horizon/mechanism eval 均为 diagnostic，不写 formal |
| Exp1 calendar | Level 2 diagnostic，CalendarFeatureFedAvg v2 已接入辅助分支，但未 formal |
| Exp3/Exp5 calendar | Level 1，有 CalendarProfileNaive baseline，但未进入神经网络输入 |
| Exp2/4/6 calendar | Level 0，无 calendar baseline，无 calendar 模型输入 |
| Exp5/6 scaler | 历史问题已修复，但旧 formal 不自动可信，需重跑 |
| Exp1 mechanism | FedAvg 优于 NaiveLastValue，但弱于 Independent 和周期性 baseline；LocalFT 后显著改善 |

## 4. 全局搜索验证结果

| 搜索项 | 结果 |
|---|---|
| CalendarFeature-FedAvg 尚未实现 / 尚未接入 FedAvg 训练链路 | 仅在"禁止使用"列表中 |
| Exp5 calendar baseline：无 | 0 匹配（已全部清除） |
| FedAvg 弱于 NaiveLastValue (94,259) | 0 匹配（已修正） |
| federated mechanism diagnostic 已运行 (在 formal 行) | 0 匹配（已改为 diagnostic 行） |
| 70/15/15 作为当前默认 | 0 匹配（全部标注为 historical/sensitivity check） |

## 5. 后续仍需处理

- 检查并必要时修正 seasonal baselines 在 h4/h12/h24 下的 evaluator；
- 重跑 Exp5 scaler-fixed r20 formal；
- 补齐 Exp6 without_attention / without_cnn / without_lstm；
- 恢复或重跑 Exp2 formal；
- Exp3/4 从 smoke 推进到 r5 diagnostic，再决定是否 r20 formal；
- CalendarFeatureFedAvg v2 从 r5 diagnostic 推进到 r10/r20 formal candidate。
