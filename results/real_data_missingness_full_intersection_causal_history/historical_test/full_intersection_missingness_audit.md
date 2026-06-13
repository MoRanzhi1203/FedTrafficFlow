# Full Intersection Missingness Audit

## 1. Historical Causal Constraint

- 本轮补全实验采用历史因果约束。对于目标日期 D 和目标时间片 t，补全方法仅允许使用 D 日 t 之前的观测以及 D 日之前的历史观测，不使用 D 日 t 之后、D+1 或更晚日期的数据。
- causal_history_only: `true`
- history_days: `7`
- context_days_after: `0`
- uses_future_days: `false`
- uses_same_day_future_slots: `false`
- uses_bfill: `false`
- uses_bidirectional_interpolation: `false`
- warmup_days: `7`
- main_metrics_exclude_warmup: `true`

## 2. Global Time Index

- 全局时间索引构造方式：`global_time_index = day_index * 96 + time_slot`
- `day_index` 来自输入 chunk 顺序。
- `time_slot` 来自 `时间段` 字段解析；若原字段不是 0-95 整数，则按日内排序映射。

## 3. Summaries

- `imputation_quality_summary_all_days.csv` 行数：`6`
- `imputation_quality_summary_exclude_warmup.csv` 行数：`6`
- `imputation_quality_by_flow_group.csv` 行数：`18`

## 4. Batch Coverage

- 选中的 chunk 总数：`8`
- `generate_missing` 已完成或跳过的 chunk 数：`8`
- `impute` 已完成或跳过的 chunk 数：`48`
- `generate_missing` 是否覆盖全部 chunk：`true`
- `impute` 是否覆盖全部 chunk：`true`
