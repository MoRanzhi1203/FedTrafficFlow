# 路口流量分片检查与使用说明

## 概述

本文档说明 `data/analysis/node_intersection_flow_parquet/` 的字段结构、分片规则、检查入口与使用建议。

说明：

- 本文档已不再维护“固定文件大小、固定行数、固定统计值”一类易失效的快照数据
- 旧版文档中的静态数值视为历史检查记录，当前以脚本实时检查结果为准

## 数据来源

生成脚本：

- `analysis_scripts/compute_node_intersection_flow_optimized.py`

上游输入：

- `data/analysis/density_metrics_chunks/`
- `data/processed/rnsd_processed.csv`

输出目录：

- `data/analysis/node_intersection_flow_parquet/`

## 命名与分片规则

默认文件命名模式：

- `node_flow_chunk_000.parquet`
- `node_flow_chunk_001.parquet`
- ...

当前主流程下，分片规则为：

- 每个文件覆盖 1 天，即 96 个连续 `时间段`
- `时间段` 使用全局连续编号，而不是每天从 0 重新开始
- 若上游日期范围仍为 61 天，则通常会生成 61 个分片

因此，更稳妥的理解是：

```text
节点流量分片数量 = 上游可用日期数
每个分片覆盖 96 个连续时间段
```

## 字段结构

默认输出包含以下 5 列：

| 列名 | 说明 |
| --- | --- |
| `节点ID` | 路网节点标识符 |
| `时间段` | 全局时间段编号 |
| `路口进入流量` | 聚合到节点的进入方向流量 |
| `路口离开流量` | 聚合到节点的离开方向流量 |
| `路口车流量` | 节点综合流量结果 |

## 排序与时间解释

当前脚本的目标输出顺序为：

- 主排序键：`时间段` 升序
- 次排序键：`节点ID` 升序

时间解释规则：

- 每个 `时间段` 对应 15 分钟
- `0-95` 表示第 1 天的 96 个日内时间段
- `96-191` 表示第 2 天
- 以此类推

该规则也是后续：

- `fit_node_flow_daily_curve.py`
- `compare_date_type_curve_methods.py`

将 `时间段 % 96` 转为 `日内时间段` 的基础。

## 推荐检查方式

### 快速结构检查

脚本：

- `dataset_inspection_scripts/inspect_node_intersection_flow.py`

可用于查看：

- 文件数和样例文件
- 列名和数据类型
- 空值统计
- 前若干行、后若干行样例

### 时间顺序与连续性检查

若要确认上游分片和下游节点流量分片是否连续、是否存在乱序，建议结合：

- `dataset_inspection_scripts/check_density_time_order.py`
- `dataset_inspection_scripts/inspect_node_intersection_flow.py`

### 方向性与节点映射检查

若怀疑节点进入流量/离开流量存在异常，可结合：

- `dataset_inspection_scripts/inspect_road_directionality.py`
- `analysis_scripts/compute_node_intersection_flow_optimized.py`

## 使用建议

### 下游分析用途

该目录主要服务于以下脚本：

- `analysis_scripts/fit_node_flow_daily_curve.py`
- `analysis_scripts/compare_date_type_curve_methods.py`

常见用途包括：

- 构造节点级 96 点日内平均流量曲线
- 做傅里叶拟合和拟合质量分析
- 比较不同日期类型处理方法的聚类表现

### 数据质量判断

以下现象通常不应直接视为错误：

- 某些节点在部分时间段 `路口进入流量` 或 `路口离开流量` 为 0
- 某些节点存在明显单向流量特征

这些现象可能来自：

- 单向路段主导
- 路网局部拓扑特征
- 方向映射差异

真正需要优先检查的是：

- 文件内排序是否混乱
- 某些时间段是否缺失
- 某些节点是否在部分文件中突然消失
- `路口车流量` 是否出现异常负值

## 已废除的旧写法

以下内容已不建议继续写入正式文档：

- “每个文件固定约 54-55 MB”
- “每个文件固定 4,034,976 行”
- “唯一节点数固定为某个历史值”
- “最大流量、均值、中位数始终为某组静态数字”

原因是这些数值会随着：

- 上游过滤规则
- 参数修正
- 输出类型调整
- 数据重生成

而发生变化，更适合作为一次性检查日志，而不是长期说明文档。

## 相关文件

- `analysis_scripts/compute_node_intersection_flow_optimized.py`
- `dataset_inspection_scripts/inspect_node_intersection_flow.py`
- `dataset_inspection_scripts/inspect_road_directionality.py`
- `analysis_scripts/fit_node_flow_daily_curve.py`
- `analysis_scripts/compare_date_type_curve_methods.py`
- `docs/node_flow_daily_curve_fit.md`
