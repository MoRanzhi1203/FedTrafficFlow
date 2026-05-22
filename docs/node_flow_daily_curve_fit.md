# 路口节点日内车流量傅里叶拟合说明

## 概述

本文档说明 `analysis_scripts/fit_node_flow_daily_curve.py` 和
`analysis_scripts/visualize_node_flow_daily_curve_fit.py` 的输入、处理流程、输出结构与可视化结果。

目标是针对每个 `节点ID` 构建 96 个日内时间段上的平均 `路口车流量` 曲线，并使用傅里叶基函数进行最小二乘拟合。

## 输入数据

输入目录：

- `data/analysis/node_intersection_flow_parquet/`

输入文件命名模式：

- `node_flow_chunk_000.parquet`
- `node_flow_chunk_001.parquet`
- ...
- `node_flow_chunk_060.parquet`

每个分片包含的字段中，本流程只使用以下 3 列：

- `节点ID`
- `时间段`
- `路口车流量`

以下字段不会参与拟合：

- `路口进入流量`
- `路口离开流量`

## 日内平均曲线构造

### 1. 全局时间段转日内时间段

项目中的 `时间段` 为全局连续编号，每 96 个时间段表示 1 天，每个时间段对应 15 分钟。

因此日内时间段定义为：

```text
日内时间段 = 时间段 % 96
```

取值范围为：

```text
0, 1, 2, ..., 95
```

### 2. 数据清洗

脚本会在读取后进行如下清洗：

- 删除 `节点ID` 为空的记录
- 删除 `时间段` 为空的记录
- 删除 `路口车流量` 为空的记录
- 删除 `路口车流量 < 0` 的记录

### 3. 按节点构造 96 点曲线

对于每个 `节点ID` 和每个 `日内时间段`，在所有日期上计算平均 `路口车流量`：

```text
q_bar_v(tau)
```

其中：

- `v` 表示节点
- `tau ∈ {0, 1, ..., 95}`

如果某个节点缺少部分日内时间段，导致不足 96 个点，则该节点会被跳过，不参与拟合。

## 傅里叶拟合模型

### 1. 自变量定义

对每个日内时间段 `tau`，令：

```text
x_tau = tau / 96
```

### 2. 傅里叶基函数

对节点 `v`，拟合函数为：

```text
f_v(x) = a0 + sum_{h=1}^{H} [a_h cos(2*pi*h*x) + b_h sin(2*pi*h*x)]
```

默认谐波数：

```text
H = 8
```

### 3. 求解方法

脚本为每个节点构建设计矩阵，并使用：

- `numpy.linalg.lstsq`

进行最小二乘求解，得到：

- `a0`
- `a1, b1`
- `a2, b2`
- ...
- `a8, b8`

### 4. 拟合后非负截断

由于实际车流量不应为负，因此拟合完成后会执行：

```text
fitted = max(fitted, 0)
```

即对负的拟合值截断为 `0`。

## 输出结果

输出目录：

- `data/analysis/node_flow_curve_fit/`

### 输出文件 1

文件名：

- `node_flow_fitted_daily_curves.parquet`

字段包括：

| 字段名 | 说明 |
|---|---|
| `节点ID` | 节点标识 |
| `日内时间段` | `0-95` 的日内时间段 |
| `平均路口车流量` | 该节点该日内时间段在所有日期上的平均流量 |
| `拟合路口车流量` | 傅里叶拟合得到的流量值，已截断到非负 |
| `残差` | `平均路口车流量 - 拟合路口车流量` |

### 输出文件 2

文件名：

- `node_flow_curve_coefficients.parquet`

字段包括：

| 字段名 | 说明 |
|---|---|
| `节点ID` | 节点标识 |
| `a0` | 常数项 |
| `a1` ~ `a8` | 余弦项系数 |
| `b1` ~ `b8` | 正弦项系数 |
| `RMSE` | 均方根误差 |
| `MAE` | 平均绝对误差 |
| `R2` | 拟合优度 |
| `平均流量` | 96 点平均曲线的均值 |
| `最大流量` | 96 点平均曲线的最大值 |
| `最小流量` | 96 点平均曲线的最小值 |

## 拟合质量指标

脚本对每个节点计算以下指标：

- `RMSE`
- `MAE`
- `R2`

其中：

- `RMSE` 越小，拟合误差越低
- `MAE` 越小，平均绝对偏差越小
- `R2` 越接近 `1`，说明拟合效果越好

若某节点 96 点平均曲线完全为常数，则 `R2` 会按特殊情况处理，避免除以 0。

## 可视化脚本

脚本：

- `analysis_scripts/visualize_node_flow_daily_curve_fit.py`

### 输入

- `data/analysis/node_flow_curve_fit/node_flow_fitted_daily_curves.parquet`
- `data/analysis/node_flow_curve_fit/node_flow_curve_coefficients.parquet`

### 输出目录

- `data/analysis/node_flow_curve_fit/plots/`

### 输出图片

#### 1. 拟合质量总览图

文件：

- `node_flow_daily_curve_fit_metrics.png`

图中包括：

- `RMSE` 分布
- `MAE` 分布
- `R2` 分布
- `平均流量 vs R2` 散点图

#### 2. 样本节点曲线对比图

文件：

- `node_flow_daily_curve_fit_samples.png`

图中包括：

- 选中节点的 `平均路口车流量` 曲线
- 对应的 `拟合路口车流量` 曲线
- 残差区域

默认会从系数结果中选择平均流量较高的一批节点进行展示，便于优先查看核心交通节点的拟合效果。

## 推荐使用顺序

```bash
python analysis_scripts/compute_node_intersection_flow_optimized.py
python analysis_scripts/fit_node_flow_daily_curve.py
python analysis_scripts/visualize_node_flow_daily_curve_fit.py
```

## 相关文件

- `analysis_scripts/compute_node_intersection_flow_optimized.py`
- `analysis_scripts/fit_node_flow_daily_curve.py`
- `analysis_scripts/visualize_node_flow_daily_curve_fit.py`
- `dataset_inspection_scripts/inspect_node_intersection_flow.py`
- `docs/node_intersection_flow_inspection.md`
