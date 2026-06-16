# Comprehensive Visualization Comparison Audit

## Summary

- output_root: `results\real_data_missingness_visual_comparison`
- figures_dir: `results\real_data_missingness_visual_comparison\figures`
- tables_dir: `results\real_data_missingness_visual_comparison\tables`
- audits_dir: `results\real_data_missingness_visual_comparison\audits`

## Checks

- 已读取 global MCAR point summary: 是
- 已读取 node_temporal_block summary: 是
- 已读取 node_subset_temporal_outage summary: 是
- 已包含 5%、10%、20%、30%: 是
- 已包含 6 个方法: 是
- 已生成每个机制内部图: 是
- 已生成 length_group 图: 是
- 已生成三机制横向对比图: 是
- 未重新运行 impute: 是
- 未重新生成 masks / missing_datasets: 是
- 未生成 imputed_datasets: 是
- 图件只代表 masked-position imputation error: 是
- 没有把 forward_fill 作为 baseline: 是
- 没有生成 relative-to-forward-fill 正式主图: 是

## Note

- Zoom figures exclude zero fill only to reveal differences among non-zero-fill methods and do not replace the formal six-method figures.
- road_topology_neighbor_fill 表示基于路网拓扑邻接关系的补全，不表示经纬度距离近邻。
