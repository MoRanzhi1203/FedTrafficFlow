# 结构化连续缺失补全因果审计报告

## 1. 范围

- 本轮只基于已有 `node_temporal_block` 的 `masks` 与 `missing_datasets` 执行补全。
- 本轮未处理 `node_subset_temporal_outage`。
- 未重新生成缺失设置。
- 未修改原始 `input_dir`。
- 本轮结果只代表缺失值补全误差，不代表交通预测误差。

## 2. 严格历史因果约束

- mechanism: `node_temporal_block`
- scenario_tag: `mixed_short_mid_long`
- causal_history_only: `True`
- context_days_before: `7`
- history_days: `7`
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
- length_group_metrics_enabled: `True`
- road_topology_neighbor_fill 使用的是 `rnsd_processed.csv` 中的路网拓扑邻接与道路长度权重，不是经纬度距离近邻。

## 4. Fallback 策略

- zero_fill: `direct_zero_fill_no_fallback`
- forward_fill: `use_global_safe_fallback_zero_when_no_causal_history_exists`
- historical_linear_extrapolation: `fallback_to_current_day_forward_fill_when_history_is_insufficient`
- road_topology_neighbor_fill: `fallback_to_current_day_forward_fill_when_no_topology_history_is_available`
- function_curve_fit: `fallback_to_current_day_forward_fill_when_no_history_profile_is_available`
- topology_function_hybrid: `blend_topology_and_function_primary_predictions_or_fallback_to_current_day_forward_fill`

## 5. 输出完整性

- rate=0.05, method=zero_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.05, method=forward_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.05, method=historical_linear_extrapolation, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.05, method=road_topology_neighbor_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.05, method=function_curve_fit, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.05, method=topology_function_hybrid, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.10, method=zero_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.10, method=forward_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.10, method=historical_linear_extrapolation, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.10, method=road_topology_neighbor_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.10, method=function_curve_fit, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.10, method=topology_function_hybrid, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.20, method=zero_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.20, method=forward_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.20, method=historical_linear_extrapolation, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.20, method=road_topology_neighbor_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.20, method=function_curve_fit, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.20, method=topology_function_hybrid, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.30, method=zero_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.30, method=forward_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.30, method=historical_linear_extrapolation, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.30, method=road_topology_neighbor_fill, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.30, method=function_curve_fit, imputed_chunk_count=61, expected=61, is_complete=True
- rate=0.30, method=topology_function_hybrid, imputed_chunk_count=61, expected=61, is_complete=True
