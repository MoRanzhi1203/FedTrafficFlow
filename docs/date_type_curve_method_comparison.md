# 日期类型曲线构造方法对比说明

## 概述

本文档说明 `analysis_scripts/compare_date_type_curve_methods.py` 的实验目的、输入数据、四种方法定义、聚类流程和输出目录结构。

该脚本的目标不是替代正式的节点日内曲线拟合流程，而是在统一数据基础上比较不同“日期类型处理方式”对日内曲线拟合质量与聚类可分性的影响。

基于该脚本输出的 parquet 结果，项目还提供后处理可视化脚本 `analysis_scripts/visualize_fitted_function_clusters.py`，用于从“函数曲线形态”而非 PCA 投影角度解释聚类结果。

当前脚本统一比较以下四种方法：

- `M0_original_fourier`
- `M1_three_date_type_curves`
- `M2_shape_normalized_weighted_curve`
- `M3_multiplicative_corrected_single_curve`

## 输入数据

输入目录：

- `data/analysis/node_intersection_flow_parquet/`

输入文件命名模式：

- `node_flow_chunk_000.parquet`
- `node_flow_chunk_001.parquet`
- ...
- `node_flow_chunk_060.parquet`

每个输入分片只使用以下字段：

- `节点ID`
- `时间段`
- `路口车流量`

脚本会先将 61 个日文件按顺序映射为 61 个自然日，并进一步标记为：

- `workday`
- `weekend`
- `holiday`

其中节假日和调休工作日由脚本内的日历映射显式指定。

## 日期类型定义

脚本内部使用以下日期类型规则：

- 法定节假日：标记为 `holiday`
- 调休工作日：标记为 `workday`
- 其余周六周日：标记为 `weekend`
- 其他日期：标记为 `workday`

当前内置日期范围对应 `2017-04-01` 至 `2017-05-31` 共 61 天。

## 四种方法定义

### M0：原始单曲线基线

方法名：

- `M0_original_fourier`

处理方式：

- 不区分日期类型
- 对同一节点的全部日期数据直接按 `日内时间段` 求平均
- 得到单条 96 点平均曲线

作用：

- 作为最基础的单曲线基线方法
- 便于与其他引入日期类型处理的方案比较

### M1：三类日期等权融合

方法名：

- `M1_three_date_type_curves`

处理方式：

- 先分别构造 `workday / weekend / holiday` 三条 96 点曲线
- 再在每个 `日内时间段` 上对三条曲线做等权平均
- 仅保留三类日期都完整存在的节点

作用：

- 检查“显式区分三类日期后再汇总”为单曲线，是否优于直接全量平均

### M2：日期类型形态归一化加权融合

方法名：

- `M2_shape_normalized_weighted_curve`

处理方式：

- 先分别构造三类日期曲线
- 对每类曲线按该节点该日期类型的日均流量做归一化，提取形态
- 按经验权重融合三类日期的归一化形态
- 再乘回节点级加权尺度，恢复到单条曲线

默认经验权重：

- `workday = 0.70`
- `weekend = 0.20`
- `holiday = 0.10`

作用：

- 让日期类型差异更多体现在“曲线形态”而不是“绝对流量水平”

### M3：全局日期类型乘性校正

方法名：

- `M3_multiplicative_corrected_single_curve`

处理方式：

- 先统计全局层面的“日期类型-时段”平均流量
- 再与全局时段平均流量比较，得到日期类型校正因子
- 对单条观测流量做乘性校正后，再聚合为单条节点曲线

作用：

- 通过全局日期类型校正减弱节假日/周末效应
- 保留单曲线输出结构，便于和 M0、M1、M2 横向比较

## 拟合与聚类流程

### 1. 曲线构造

所有方法最终都会落到“每个节点一条 96 点曲线”的表示形式。

例外说明：

- 在日期类型分解阶段，M1 和 M2 会先构造分日期类型曲线
- 但最终比较和聚类阶段依然统一为节点级表示

### 2. 傅里叶拟合

脚本会对每条节点曲线执行傅里叶最小二乘拟合：

```text
f(x) = a0 + sum_{h=1}^{H}[a_h cos(2*pi*h*x) + b_h sin(2*pi*h*x)]
```

当前固定谐波数：

```text
H = 6
```

拟合后会对负值做非负截断。

### 3. 节点级指标

每个节点都会计算：

- `RMSE`
- `MAE`
- `R2`
- `平均流量`
- `最大流量`
- `最小流量`
- 若干曲线形态摘要指标

### 4. 聚类特征

聚类阶段对节点曲线提取归一化形态特征，包括：

- 傅里叶系数形态特征
- 峰谷差
- 峰值时段的正余弦编码
- 早晚峰对数比

之后会执行：

- 极值裁剪 `winsorize`
- `RobustScaler` 标准化
- `KMeans` 聚类

### 5. 聚类数搜索

当前搜索范围：

```text
k = 3, 4, 5, 6
```

评估指标包括：

- `silhouette_score`
- `calinski_harabasz_score`
- `davies_bouldin_score`
- `negative_silhouette_ratio`
- 聚类规模平衡性相关指标

脚本会综合这些指标自动选择 `best_k`。

## 输出结构

输出根目录：

- `data/analysis/date_type_curve_method_comparison/`

该目录下会为每个方法创建一个子目录，例如：

- `M0_original_fourier/`
- `M1_three_date_type_curves/`
- `M2_shape_normalized_weighted_curve/`
- `M3_multiplicative_corrected_single_curve/`

每个方法目录包含：

- `daily_profiles.parquet`
- `fitted_curves.parquet`
- `curve_coefficients.parquet`
- `cluster_metrics.parquet`
- `cluster_labels.parquet`
- `cluster_summary.parquet`
- `cluster_centers.parquet`
- `cluster_label_mapping.parquet`

此外还会生成统一比较目录：

- `data/analysis/date_type_curve_method_comparison/comparison/`

其中包含：

- `method_fit_metrics_summary.parquet`
- `method_cluster_metrics_summary.parquet`
- `method_comparison_table.csv`
- `all_k_cluster_metrics.csv`
- `method_cluster_metric_comparison.png`
- `method_center_curve_comparison.png`
- `method_pca_comparison.png`
- `method_cluster_size_comparison.png`
- `all_k_metric_curves.png`

## 函数聚类可视化后处理

脚本：

- `analysis_scripts/visualize_fitted_function_clusters.py`

作用：

- 读取单个方法目录下既有的 parquet 输出
- 不重新运行 `KMeans`，不修改 `cluster_id`，只做可视化和结果解释
- 将聚类结果直接还原到“拟合函数曲线”层面，而不是只看 PCA 散点图

默认输入目录：

- `data/analysis/date_type_curve_method_comparison/<method>/`

默认输出目录：

- `data/analysis/date_type_curve_method_comparison/function_cluster_visualization/`

当前可输出的主要图表和表格包括：

- `*_sampled_function_cloud_with_center.png`
- `*_fitted_function_overlay.png`
- `*_cluster_function_quantile_bands.png`
- `*_cluster_mean_fitted_vs_center.png`
- `*_representative_fitted_functions.png`
- `*_all_fitted_functions_diagnostic.png`
- `*_residual_distribution_by_cluster.png`
- `*_normalized_residual_distribution_by_cluster.png`
- `*_function_cluster_summary.csv`
- `*_node_function_cluster_labels.csv`

推荐阅读顺序：

- 主解释图：`sampled_function_cloud_with_center.png`
- 技术检查图：`fitted_function_overlay.png`
- 稳定性辅助图：`cluster_function_quantile_bands.png`
- 代表样本图：`representative_fitted_functions.png`
- 诊断图：`all_fitted_functions_diagnostic.png`

更多细节见：

- [function_cluster_visualization.md](file:///e:/Jupter_Notebook/FedTrafficFlow/docs/function_cluster_visualization.md)

## 命令行参数

脚本支持以下主要参数：

### `--methods`

指定需要运行的方法列表，例如：

```bash
python analysis_scripts/compare_date_type_curve_methods.py --methods M0_original_fourier M2_shape_normalized_weighted_curve
```

### `--workday-weight`

指定 M2 中 `workday` 的经验权重。

### `--weekend-weight`

指定 M2 中 `weekend` 的经验权重。

### `--holiday-weight`

指定 M2 中 `holiday` 的经验权重。

### `--node-sample-size`

仅抽样部分节点进行流程验证，适合快速调试。

例如：

```bash
python analysis_scripts/compare_date_type_curve_methods.py --node-sample-size 500
```

## 适用说明

- 该脚本适合用于方法比较、论文实验和可解释性分析
- 若只需要正式的节点日内平均曲线拟合结果，仍以 `fit_node_flow_daily_curve.py` 为主流程
- 若输入分片数量不是 61 个，脚本会报错，因为当前内置日历映射固定为 61 天
- 该脚本会生成较多中间结果和图片，适合本地分析环境，不建议在轻量部署环境中默认执行
- 若需要进一步解释单个方法的类中心、类内离散度和代表节点，应在本脚本完成后继续运行 `visualize_fitted_function_clusters.py`

## 相关文件

- `analysis_scripts/compare_date_type_curve_methods.py`
- `analysis_scripts/fit_node_flow_daily_curve.py`
- `analysis_scripts/compare_node_flow_fourier_orders.py`
- `analysis_scripts/visualize_node_flow_daily_curve_fit.py`
- `analysis_scripts/visualize_fitted_function_clusters.py`
- `docs/node_flow_daily_curve_fit.md`
- `docs/function_cluster_visualization.md`
