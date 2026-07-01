# 输入特征审计报告

> 生成日期：2026-07-01
> 审计范围：所有 real_data_experiments 数据输入来源

## 1. 当前真实实验使用的字段

### 1.1 已进入模型输入的字段

| 字段 | 来源 | 使用方式 | 适用实验 |
|---|---|---|---|
| total_flow (channel 0) | `node_flow_grid_tensor.pt[:, 0, :]` | CNN 输入通道 | 全部实验 |
| mean_flow (channel 1) | `node_flow_grid_tensor.pt[:, 1, :]` | CNN 输入通道 | 全部实验 |
| sin_time_of_day | `calendar_features_15min.csv` | CalendarFeatureFedAvg 辅助分支 | 仅 Exp1 |
| cos_time_of_day | `calendar_features_15min.csv` | CalendarFeatureFedAvg 辅助分支 | 仅 Exp1 |
| sin_day_of_week | `calendar_features_15min.csv` | CalendarFeatureFedAvg 辅助分支 | 仅 Exp1 |
| cos_day_of_week | `calendar_features_15min.csv` | CalendarFeatureFedAvg 辅助分支 | 仅 Exp1 |
| is_holiday | `calendar_features_15min.csv` | CalendarFeatureFedAvg 辅助分支 | 仅 Exp1 |
| is_weekend | `calendar_features_15min.csv` | CalendarFeatureFedAvg 辅助分支 | 仅 Exp1 |
| is_effective_workday | `calendar_features_15min.csv` | CalendarFeatureFedAvg 辅助分支 | 仅 Exp1 |
| is_adjusted_workday | `calendar_features_15min.csv` | CalendarFeatureFedAvg 辅助分支 | 仅 Exp1 |
| days_to_nearest_holiday | `calendar_features_15min.csv` | CalendarFeatureFedAvg 辅助分支 | 仅 Exp1 |

### 1.2 仅用于预处理/partition 的字段

| 字段 | 来源 | 使用方式 |
|---|---|---|
| centroid_lon | `node_flow_grid_regions.csv` | flow_kmeans / spatial_block 分区 |
| centroid_lat | `node_flow_grid_regions.csv` | flow_kmeans / spatial_block 分区 |
| pooled_row | `node_flow_grid_regions.csv` | spatial_block 蛇形排序 |
| pooled_col | `node_flow_grid_regions.csv` | spatial_block 蛇形排序 |
| is_active_region | `node_flow_grid_regions.csv` | 过滤无效 grid cells |
| source_node_count | `node_flow_grid_regions.csv` | full_cell_inventory 有效性检查 |
| flow_mean / flow_std / flow_cv | 从 tensor 计算 | similarity_partition 特征矩阵 |
| lag1_autocorr | 从 tensor 计算 | similarity_partition 特征矩阵 |
| is_effective_workday (train split) | `calendar_features_15min.csv` | CalendarProfileNaive profile 构建 |
| slot_of_day | `calendar_features_15min.csv` | CalendarProfileNaive profile 构建 |

### 1.3 可扩展但未进入模型的字段

| 字段 | 来源/获取方式 | 是否可稳定对齐 | 备注 |
|---|---|---|---|
| road length | Q-Traffic 原始表 | 未知 | 道路静态属性 |
| road width | Q-Traffic 原始表 | 未知 | 道路静态属性 |
| lane number | Q-Traffic 原始表 | 未知 | 道路静态属性 |
| speed class | Q-Traffic 原始表 | 未知 | 道路类型信息 |
| road degree (in/out) | Q-Traffic 拓扑表 | 未知 | 需道路网络匹配 |
| grid adjacency | 从 grid 空间关系构建 | 可 | 用于 GCN |
| speed observations | Q-Traffic 原始表 | 可能 | 当前未使用 |
| segment coordinates | Q-Traffic 原始表 | 可能 | 已聚合为 grid |

### 1.4 不可获取的字段

| 字段 | 原因 |
|---|---|
| 天气 (降水/温度/风速) | 无气象数据接入 |
| 交通事故/施工事件 | 无事件数据源 |
| 信号控制参数 | 无信号控制数据 |

## 2. 当前输入组合总结

| 组合 | 适用实验 | 字段 |
|---|---|---|
| traffic_only | 全部 | total_flow + mean_flow |
| traffic_calendar (CalendarFeatureFedAvg) | Exp1 diagnostic | traffic_only + 9 calendar features (residual-gated) |
| traffic_calendar (CalendarProfileNaive) | Exp1/3/5 | traffic_only + calendar as independent baseline |

## 3. 建议的栏目选择实验

### 3.1 最小对比

```text
A. traffic_only = [total_flow, mean_flow]          # 当前默认
B. traffic_calendar = A + calendar branch            # CalendarFeatureFedAvg
```

### 3.2 扩展对比（如果静态字段可对齐）

```text
C. traffic_calendar_static = B + road static covariates (grid-level)
```

扩张优先级：B > C（calendar 特征已有代码，静态字段需额外数据对齐）

## 4. 论文写作指导

- 论文数据描述段应诚实列出模型输入字段（当前仅 total_flow + mean_flow）
- 在 limitations 中说明未使用 speed/static attributes 的原因
- calendar features 的使用范围需精确描述（Exp1 diagnostic only）
