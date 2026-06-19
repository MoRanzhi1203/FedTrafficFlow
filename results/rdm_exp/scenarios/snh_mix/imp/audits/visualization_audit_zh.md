# snh_mix 第一阶段可视化审计

- Scenario: `spatial_neighbor_holdout (snh_mix)`
- Rates: 5%, 10%, 20%, 30%
- snh_mix uses the same formal summary framework as the previous three mechanisms.
- Formal summary dimensions: `overall`, `flow_group`, `length_group`.
- `neighbor_coverage` and `constraint_level` are not required formal summary dimensions in this version.
- Evaluation: `masked-position imputation error`
- methods_phase: `phase_1_six_baseline_methods`
- methods_count: `6`
- methods: `mean_fill, forward_fill, historical_linear_extrapolation, function_curve_fit, road_topology_neighbor_fill, correlation_topology_neighbor_fill`
- removed_methods: `adaptive_spatio_temporal_fill`
- Methods: six baseline methods only.
- No adaptive methods.
- 允许使用当前时刻邻居观测值。
- 不允许使用目标节点当前真实值。
- 不允许使用未来时间片或未来日期。
- `none` 等级单独展示，不作为空间优势证明。
- 本结果不是 traffic prediction error，也不是 forecasting accuracy。
