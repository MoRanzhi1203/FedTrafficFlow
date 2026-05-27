# 拟合函数聚类可视化说明

## 概述

本文档说明 `analysis_scripts/visualize_fitted_function_clusters.py` 的使用前提、输入文件、核心参数、输出图表与推荐阅读方式。

该脚本的定位是：

- 不重新聚类
- 不改变已有 `cluster_id`
- 不修改已有 parquet 输出
- 仅基于 `compare_date_type_curve_methods.py` 已生成的结果，从“拟合函数曲线形态”角度解释聚类

因此，它属于日期类型方法比较实验的后处理可视化步骤，而不是新的建模或聚类流程。

## 输入前提

运行该脚本前，应先完成：

```bash
python analysis_scripts/compare_date_type_curve_methods.py
```

默认情况下，脚本会读取：

- `data/analysis/date_type_curve_method_comparison/<method>/fitted_curves.parquet`
- `data/analysis/date_type_curve_method_comparison/<method>/cluster_labels.parquet`
- `data/analysis/date_type_curve_method_comparison/<method>/cluster_summary.parquet`
- `data/analysis/date_type_curve_method_comparison/<method>/cluster_centers.parquet`
- `data/analysis/date_type_curve_method_comparison/<method>/curve_coefficients.parquet`

如果这些文件缺失，脚本会直接报错，提示先运行方法比较脚本。

## 默认方法与输出目录

默认方法：

- `M2_shape_normalized_weighted_curve`

默认输出目录：

- `data/analysis/date_type_curve_method_comparison/function_cluster_visualization/`

典型运行命令：

```bash
python analysis_scripts/visualize_fitted_function_clusters.py --method M2_shape_normalized_weighted_curve
```

## 核心参数

### 基础参数

- `--method`
  - 指定要可视化的方法目录名
- `--output-dir`
  - 指定输出目录
- `--random-state`
  - 控制抽样类图中的随机性

### 抽样与主图参数

- `--sample-per-cluster`
  - 默认 `200`
  - 用于抽样函数云图和 overlay 图
- `--overlay-center-type`
  - 可选 `mean`、`median`、`saved_center`
  - 默认 `saved_center`
  - 控制 `fitted_function_overlay.png` 中黑色中心线的来源

### y 轴参数

- `--main-y-max`
  - 默认 `2.2`
  - 用于主图、overlay、分位带图、代表节点图
- `--diagnostic-y-max`
  - 默认 `4.0`
  - 用于全量曲线诊断图
- `--plot-y-quantile`
  - 默认 `0.99`
  - 仅当固定 y 轴参数小于等于 `0` 时，回退为分位数自动计算

### 代表节点与诊断参数

- `--representative-top-n`
  - 默认 `6`
  - 每个 cluster 的代表节点曲线数量
- `--all-curves-alpha`
  - 默认 `0.035`
  - 全量诊断图中灰色曲线的透明度

### 观测曲线参数

- `--curve-type`
  - 可选 `fitted`、`observed`、`both`
  - 默认 `fitted`
- `--show-observed`
  - 兼容参数
  - 当 `curve-type` 仍为默认 `fitted` 时，会自动切换为 `both`

## 输出图表定位

### 1. `sampled_function_cloud_with_center.png`

定位：

- 推荐主图
- 更接近人工直观看类形态的效果

内容：

- 每类随机抽样若干条归一化拟合曲线
- 曲线使用 cluster 颜色
- 黑色粗线为 `saved cluster center`

适用场景：

- 论文正文
- 报告首页主图
- 快速解释三类函数形态差异

### 2. `fitted_function_overlay.png`

定位：

- 技术检查图

内容：

- 每类随机抽样若干条曲线
- 黑色中心线来源可切换为 `mean`、`median` 或 `saved_center`

适用场景：

- 检查样本曲线与中心线定义是否一致
- 对比不同中心线口径的影响

### 3. `cluster_function_quantile_bands.png`

定位：

- 辅助稳定性图

内容：

- `10%-90%` 区间带
- `25%-75%` 区间带
- `median` 曲线
- `mean` 曲线

适用场景：

- 说明类内波动范围
- 辅助判断类内是否稳定、是否存在宽分布

### 4. `representative_fitted_functions.png`

定位：

- 多样代表节点图

当前策略：

- 不再只选最接近中位曲线的节点
- 而是同时选取典型节点、中等差异节点和边界差异节点

适用场景：

- 展示“同一类内部仍然存在怎样的合理差异”
- 在正文或附录中展示代表样本

### 5. `all_fitted_functions_diagnostic.png`

定位：

- 全量诊断图

内容：

- 每类全部节点的归一化拟合曲线
- 统一灰色、极低透明度

适用场景：

- 检查极端异常曲线
- 观察类内离散度

注意：

- 不建议直接作为正文主图

### 6. `residual_distribution_by_cluster.png`

定位：

- 原始残差诊断图

内容：

- 按 cluster 比较原始 `residual` 的箱线分布

适用场景：

- 直接观察拟合误差绝对量级

### 7. `normalized_residual_distribution_by_cluster.png`

定位：

- 归一化残差诊断图

内容：

- 使用 `_normalized_residual = residual / observed_mean`
- 对不同平均流量水平的 cluster 做更公平的残差比较

适用场景：

- 横向比较不同 cluster 的相对拟合误差

## 输出表说明

### `function_cluster_summary.csv`

用于输出每个 cluster 的函数形态摘要，包括：

- 节点数
- 平均 `R2 / RMSE / MAE`
- 峰值时段与谷值时段
- 峰谷差
- 早晚峰强度与比值
- 人工解释标签

### `node_function_cluster_labels.csv`

用于输出每个节点的函数聚类标签，包括：

- `node_id`
- `method`
- `cluster_id`
- `cluster_name`
- `R2`
- `RMSE`
- `MAE`
- `mean_flow`

## 推荐阅读顺序

如果目标是快速解释实验结果，建议按以下顺序查看：

1. `sampled_function_cloud_with_center.png`
2. `representative_fitted_functions.png`
3. `cluster_function_quantile_bands.png`
4. `normalized_residual_distribution_by_cluster.png`
5. `all_fitted_functions_diagnostic.png`

如果目标是做技术核对，建议按以下顺序查看：

1. `fitted_function_overlay.png`
2. `cluster_mean_fitted_vs_center.png`
3. `residual_distribution_by_cluster.png`
4. `all_fitted_functions_diagnostic.png`

## 与主实验脚本的关系

该脚本与 `compare_date_type_curve_methods.py` 的关系可概括为：

```text
compare_date_type_curve_methods.py
-> 生成 fitted_curves / cluster_labels / cluster_summary / cluster_centers
-> visualize_fitted_function_clusters.py 读取这些结果
-> 输出论文图、技术检查图和解释性 CSV
```

因此，若上游重新运行并覆盖了某个方法目录，建议同步重新运行本脚本，以避免图表和 parquet 结果不一致。

## 相关文件

- `analysis_scripts/compare_date_type_curve_methods.py`
- `analysis_scripts/visualize_fitted_function_clusters.py`
- `docs/date_type_curve_method_comparison.md`
- `docs/project_pipeline.md`
