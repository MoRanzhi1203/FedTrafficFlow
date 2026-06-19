# 全局 MCAR 缺失补全因果审计报告

## 1. 范围

- 本轮只基于已有 `masks` 与 `missing_datasets` 执行补全。
- 未重新生成缺失设置。
- 未修改原始 `input_dir`。
- 本轮结果只代表缺失值补全误差，不代表交通预测误差。

## 2. 严格历史因果约束

- causal_history_only: `True`
- history_days: `7`
- context_days_before: `7`
- context_days_after: `0`
- uses_future_days: `False`
- uses_same_day_future_slots: `False`
- uses_bfill: `False`
- uses_bidirectional_interpolation: `False`
- warmup_days: `7`
- main_metrics_exclude_warmup: `True`

## 3. 评价口径

- evaluation_only_on_mask_positions: `True`
- non_mask_positions_preserved: `True`
- road_topology_neighbor_fill 使用的是 `rnsd_processed.csv` 中的路网拓扑邻接与道路长度权重，不是经纬度距离近邻。

## 4. 方法变更

- added_methods: `['mean_fill']`
- removed_methods: `['zero_fill']`

## 5. Fallback 策略

- mean_fill: `same_slot_7day_mean -> node_7day_mean -> slot_7day_mean -> global_7day_mean -> current_day_forward_fill`
- forward_fill: `use_previous_slot_or_previous_day_last_slot_then_global_safe_fallback_zero`
- historical_linear_extrapolation: `fallback_to_current_day_forward_fill_when_history_is_insufficient`
- function_curve_fit: `fallback_to_current_day_forward_fill_when_no_history_profile_is_available`
- road_topology_neighbor_fill: `fallback_to_current_day_forward_fill_when_no_topology_history_is_available`
- correlation_topology_neighbor_fill: `same-time positive-correlation topology neighbors -> mean_fill`

## 6. 输出完整性

- rate=0.05, method=mean_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.05, method=forward_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.05, method=historical_linear_extrapolation, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.05, method=function_curve_fit, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.05, method=road_topology_neighbor_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.05, method=correlation_topology_neighbor_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.10, method=mean_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.10, method=forward_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.10, method=historical_linear_extrapolation, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.10, method=function_curve_fit, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.10, method=road_topology_neighbor_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.10, method=correlation_topology_neighbor_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.20, method=mean_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.20, method=forward_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.20, method=historical_linear_extrapolation, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.20, method=function_curve_fit, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.20, method=road_topology_neighbor_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.20, method=correlation_topology_neighbor_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.30, method=mean_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.30, method=forward_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.30, method=historical_linear_extrapolation, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.30, method=function_curve_fit, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.30, method=road_topology_neighbor_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.30, method=correlation_topology_neighbor_fill, imputed_chunk_count=61, expected=61, is_complete=True
