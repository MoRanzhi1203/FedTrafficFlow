# 真实数据缺失值注入实验结果

## 1. 实验口径

完整路口阶段历史因果补全结果仅对应真实数据上的插补质量评估，而不是交通流预测性能评估。本轮结果使用完整路口阶段真实数据，在完整真实数据基础上人为注入缺失值，并在严格历史约束下执行缺失恢复。

## 2. 历史因果补全约束下的完整路口阶段结果

完整路口阶段结果必须满足以下解释口径：

1. 本轮结果使用完整路口阶段真实数据；
2. 缺失是在完整真实数据基础上人为注入；
3. 补全阶段只使用历史数据；
4. 不使用未来日期和目标点之后的观测；
5. 传统双向 `linear_interpolation` 已被替换为 `historical_linear_extrapolation`；
6. 前 `history_days` 天作为 warmup，主结果可以排除；
7. 结果为插补质量评估，不是交通流预测结果。

对于目标日期 `D` 和目标时间片 `t`，补全方法仅允许使用 `D` 日 `t` 之前的已观测数据以及 `D` 日之前的历史数据，不使用 `D` 日 `t` 之后、`D+1` 或更晚日期的数据。因此，结果中的误差应被解释为人为遮蔽位置上的插补误差，而不能写成预测误差。

## 3. 结果文件口径

历史因果完整路口阶段结果默认输出到：

```text
results/real_data_missingness_full_intersection_causal_history
```

其中主结果表优先使用：

```text
summaries/imputation_quality_summary_exclude_warmup.csv
```

完整统计表保留为：

```text
summaries/imputation_quality_summary_all_days.csv
```

此外，还应同时检查：

```text
summaries/imputation_quality_by_flow_group.csv
summaries/node_flow_group_summary.csv
full_intersection_missingness_audit.md
full_intersection_missingness_audit.json
```

## 4. 方法解释

历史因果完整路口阶段中，各方法的含义应统一为：

1. `zero_fill`：不使用任何未来信息，直接填零；
2. `forward_fill`：仅使用节点历史有效观测的历史前向填补；
3. `historical_linear_extrapolation`：仅基于历史点执行局部线性趋势外推；
4. `geo_neighbor_fill`：仅使用邻居节点在目标时刻之前的历史观测；
5. `function_curve_fit`：仅使用目标节点在当前缺失位置之前的历史时间序列；
6. `geo_func_hybrid`：仅融合历史地理邻近与历史函数曲线拟合结果。

在该口径下，不应再将任何方法描述为依赖未来观测，也不应继续使用“双向插值”“全局最优插补”等表述。

## 5. warmup 与主表解释

由于前 `history_days` 天无法获得完整历史窗口，warmup 阶段的结果会更多依赖回退策略。因此，论文主表应优先使用排除 warmup 的汇总结果；如果需要报告包含 warmup 的全量统计，则必须明确说明前几天历史不足、回退比例较高，不能与稳定阶段结果直接混合解释。

## 6. 小流量节点结果解释

为检验历史因果补全在不同流量尺度下的稳定性，实验将节点划分为 `low_flow`、`mid_flow` 与 `high_flow` 三组，并分别报告 MAE、RMSE 与 sMAPE。该部分用于说明方法是否对小流量节点存在过度平滑或过度放大问题，而不是用于推断下游预测性能。

## 7. 图件标题与说明

主图标题应写为：

```text
Imputation RMSE under Artificial Missing Rates
Full Intersection-stage Real Data, Historical Causal Setting
```

局部放大图标题应写为：

```text
Zoomed RMSE Comparison of Historical Geo and Functional Imputation
```

相对差值图标题应写为：

```text
RMSE Difference Relative to Causal Forward Fill
```

## 8. 结论口径

完整路口阶段历史因果结果只能得出“在严格历史约束下，不同插补方法对人为缺失位置的恢复质量存在差异”这一类结论。文中不得将该结果写成使用未来观测补全、双向插值补全、联邦预测性能提升或独立预测性能提升。
