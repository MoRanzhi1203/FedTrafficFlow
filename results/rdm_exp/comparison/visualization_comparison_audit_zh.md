# 正式综合可视化审计

## 说明

- output_root: `results\rdm_exp\comparison`
- figures_dir: `results\rdm_exp\comparison\figures`
- tables_dir: `results\rdm_exp\comparison\tables`
- audits_dir: `results\rdm_exp\comparison\audits`
- 本轮只重新生成可视化和对比表。
- 未重新生成缺失。
- 未重新运行补全。
- 未重新生成 imputed_datasets。
- 当前图件表示 masked-position imputation error，不是交通流预测误差。
- zero_fill 已从正式可视化中移除。
- mean_fill 已纳入三类机制正式对比。

## 检查

- 已读取 global MCAR point summary: 是
- 已读取 node_temporal_block summary: 是
- 已读取 node_subset_temporal_outage summary: 是
- 已包含 5%、10%、20%、30%: 是
- 已包含 6 个方法: 是
- 已生成每个机制内部图: 是
- 已生成 flow_group 图: 是
- 已生成 length_group 图: 是
- 已生成三机制横向对比图: 是
- 未重新运行 impute: 是
- 未重新生成 masks / missing_datasets: 是
- 未生成 imputed_datasets: 是
- 图件只代表 masked-position imputation error: 是
- 没有把 forward_fill 作为 baseline: 是
- 没有生成 relative-to-forward-fill 正式主图: 是

## 备注

- road_topology_neighbor_fill 表示基于路网拓扑邻接关系的补全，不表示经纬度距离近邻。
