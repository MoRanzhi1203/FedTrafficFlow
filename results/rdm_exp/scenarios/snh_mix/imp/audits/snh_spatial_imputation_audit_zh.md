# snh_mix 空间插补审计

- scenario_id: `snh_mix`
- mechanism: `spatial_neighbor_holdout`
- evaluation_protocol: `online_spatial_interpolation`
- mask_scope: `global`
- snh_mix uses the same formal summary framework as g_mcar_pt / ntb_mix / nso_mix.
- Formal summary dimensions: `overall`, `flow_group`, `length_group`.
- `neighbor_coverage` and `constraint_level` are optional spatial diagnostics, not required formal summary outputs.
- 允许使用目标节点缺失时刻的邻居观测。
- 不允许使用目标节点当前真实值。
- 不允许使用未来时间片或未来天。
- 当前指标仅在人工 mask 位置计算，不是交通流预测误差。
- methods_phase: `phase_1_six_baseline_methods`
- methods: `mean_fill, forward_fill, historical_linear_extrapolation, function_curve_fit, road_topology_neighbor_fill, correlation_topology_neighbor_fill`
- methods_count: `6`
- removed_methods: `adaptive_spatio_temporal_fill`
- `none` 等级单独统计，只用于补足全局缺失计数，不用于空间优势结论。
