# 真实数据训练入口审计

## 1. 当前问题

当前 `single_intersection_client` 与 `single_intersection_ablation` 之所以临时使用 `data/analysis/node_intersection_flow_parquet/`，原因不是论文主线改变，而是此前迁移时优先保证 Python 工程结构、标准 `FedAvg`、指标导出和可视化链路可运行；当时 notebook 直接依赖的 `6.池化网格张量.pt` 没有对应的正式 `.py` 生成脚本，也没有以规范命名存在于仓库的数据产物，因此只能采用“上游节点流量 parquet 直接读入”的 fallback 方案完成 smoke test。

现在重新审计后可以确认：

- 基础真实数据预处理主链已经 `.py` 化，并且磁盘上存在：
  - `data/processed/speed_data_chunks/`
  - `data/analysis/density_metrics_chunks/`
  - `data/analysis/node_intersection_flow_parquet/`
  - `data/analysis/node_flow_curve_fit/`
- 但面向 CCN 网格联邦训练的正式“网格化 -> 池化 -> `.pt` 张量”链路仍未 `.py` 化。
- `test/预处理5.ipynb` 与 `test/预处理6.ipynb` 中确实保留了 notebook 版网格化、池化和 `6.池化网格张量.pt` 保存逻辑。
- 仓库当前不存在：
  - `preprocessing_scripts/process_node_flow_grids.py`
  - `preprocessing_scripts/process_node_flow_tensor.py`
  - `data/processed/node_flow_grid/node_flow_grid_2ch.npy`
  - `data/processed/node_flow_grid/node_flow_grid_pooled.npy`
  - `data/processed/node_flow_grid/node_flow_grid_tensor.pt`
  - `data/processed/node_flow_grid/node_flow_grid_regions.csv`

因此，当前单路口实验使用 `node_intersection_flow_parquet` 只是用于 Python 工程 smoke test 的 fallback，不应当被视为最终 CCN 网格联邦训练入口。

## 2. 基础预处理 py 化状态

| 步骤名称 | 原 notebook 来源 | 现有 py 文件 | 是否存在 | 输入 | 输出 | 是否可复现 | 是否仍依赖 notebook | 备注 |
|---|---|---|---|---|---|---|---|---|
| 原始 link_gps 清洗 | `预处理1.ipynb` 历史来源 | `preprocessing_scripts/process_link_gps.py` | 是 | `data/raw/link_gps.v2` | `data/processed/link_gps_processed.csv` | 是 | 否 | 已形成正式脚本与输出。 |
| RNSD 路网清洗 | `预处理1.ipynb` 历史来源 | `preprocessing_scripts/process_rnsd.py` | 是 | `data/raw/road_network_sub-dataset.v2` | `data/processed/rnsd_processed.csv` | 是 | 否 | 已输出路段属性与起止节点信息。 |
| 速度数据合并分块 | `预处理2.ipynb` 历史来源 | `preprocessing_scripts/merge_speed_data.py` | 是 | `traffic_speed_sub-dataset.v2` + `link_gps_processed.csv` + `rnsd_processed.csv` | `data/processed/speed_data_chunks/speed_chunk_*.parquet` | 是 | 否 | 当前磁盘存在 61 个分片。 |
| 速度统计 | `预处理3.ipynb` 历史来源 | `analysis_scripts/analysis/summarize_speed_stats.py` | 是 | `data/processed/speed_data_chunks/` | `data/analysis/speed_class_overall_stats.csv`、`speed_class_daily_period_stats.csv` | 是 | 否 | 已有正式统计脚本。 |
| 速度直方图 | `预处理3.ipynb` 历史来源 | `analysis_scripts/real_data_analysis/visualize_speed_hist_by_period.py` | 是 | `data/processed/speed_data_chunks/` | `data/analysis/speed_histogram_counts_by_period_by_class.csv`、`speed_histograms_by_period_by_class/` | 是 | 否 | 当前磁盘存在图像输出。 |
| P99.5 统计 | `预处理3.ipynb` 历史来源 | `analysis_scripts/analysis/add_p995_to_speed_histogram.py` | 是 | `data/processed/speed_data_chunks/` | `data/analysis/speed_histograms_by_class_p995.csv` | 是 | 否 | 当前磁盘存在 CSV 与 PNG。 |
| Greenshields 密度和流量计算 | `预处理4.ipynb` 历史来源 | `analysis_scripts/preprocessing/compute_greenshields_density.py` | 是 | `speed_data_chunks/` + `speed_class_density_params.csv` | `data/analysis/density_metrics_chunks/density_chunk_*.parquet` | 是 | 否 | 当前磁盘存在 61 个分片。 |
| 路段流量聚合为节点流量 | `预处理4.ipynb` 历史来源 | `analysis_scripts/preprocessing/compute_node_intersection_flow_optimized.py` | 是 | `density_metrics_chunks/` + `rnsd_processed.csv` | `data/analysis/node_intersection_flow_parquet/node_flow_chunk_*.parquet` | 是 | 否 | 当前磁盘存在 61 个分片。 |
| 节点流量完整性检查 | `预处理4.ipynb` 历史来源 | `analysis_scripts/analysis/check_spatial_node_completeness.py` | 是 | `node_intersection_flow_parquet/` + `rnsd_processed.csv` | `data/analysis/node_intersection_flow_check_reports/*.csv` | 是 | 否 | 当前磁盘存在检查报告。 |
| 节点日内曲线拟合 | `预处理4.ipynb` 历史来源 | `analysis_scripts/preprocessing/fit_node_flow_daily_curve.py` | 是 | `node_intersection_flow_parquet/` | `data/analysis/node_flow_curve_fit/*.parquet` | 是 | 否 | 当前磁盘存在拟合结果。 |
| 傅里叶阶数比较 | `预处理4.ipynb` 历史来源 | `analysis_scripts/analysis/compare_node_flow_fourier_orders.py` | 是 | `node_intersection_flow_parquet/` | `data/analysis/node_flow_curve_fit/node_flow_fourier_order_comparison.json` | 是 | 否 | 当前磁盘存在 JSON。 |
| 日期类型曲线方法比较 | `预处理4.ipynb` 历史来源 | `analysis_scripts/analysis/compare_date_type_curve_methods.py` | 是 | `node_intersection_flow_parquet/` | `data/analysis/date_type_curve_method_comparison/` | 是 | 否 | 当前磁盘存在 M0-M3 结果。 |
| 函数聚类可视化 | `预处理4.ipynb` 历史来源 | `analysis_scripts/real_data_analysis/visualize_fitted_function_clusters.py` | 是 | `date_type_curve_method_comparison/` | `function_cluster_visualization/` | 是 | 否 | 当前磁盘存在函数聚类图。 |
| 节点流量网格化 | `预处理5.ipynb` | 无正式 `.py` | 否 | `node_intersection_flow_parquet` + 节点经纬度表 | 理应输出 `node_flow_grid_2ch.npy` | 否 | 是 | notebook 中使用 `1.路口节点经纬度.csv` 和 `4.路口节点车流量/*.csv`。 |
| 网格池化 | `预处理5.ipynb` | 无正式 `.py` | 否 | 网格化双通道 `.npy` | 理应输出 `node_flow_grid_pooled.npy` | 否 | 是 | notebook 通过 `torch.nn.functional.max_pool2d` 完成。 |
| 网格张量保存为 `.pt` | `预处理6.ipynb` | 无正式 `.py` | 否 | 池化后的网格化 `.npy` | notebook 输出 `6.池化网格张量.pt` | 否 | 是 | 仓库中未形成 `node_flow_grid_tensor.pt` 正式产物。 |

## 3. 网格化 / 池化 / 张量化状态

### 3.1 必查脚本与数据产物

| 路径 | 状态 | 结论 |
|---|---|---|
| `preprocessing_scripts/process_node_flow_grids.py` | 不存在 | 缺失正式网格化/池化脚本 |
| `preprocessing_scripts/process_node_flow_tensor.py` | 不存在 | 缺失正式 `.pt` 张量保存脚本 |
| `data/processed/node_flow_grid/` | 不存在 | 缺失正式网格化输出目录 |
| `data/processed/node_flow_grid/node_flow_grid_2ch.npy` | 不存在 | 缺失 |
| `data/processed/node_flow_grid/node_flow_grid_pooled.npy` | 不存在 | 缺失 |
| `data/processed/node_flow_grid/node_flow_grid_tensor.pt` | 不存在 | 缺失 |
| `data/processed/node_flow_grid/node_flow_grid_regions.csv` | 不存在 | 缺失 |

### 3.2 当前已存在的上游节点流量链路

当前磁盘上已经存在完整的上游节点流量中间数据：

```text
data/processed/speed_data_chunks/speed_chunk_*.parquet
-> data/analysis/density_metrics_chunks/density_chunk_*.parquet
-> data/analysis/node_intersection_flow_parquet/node_flow_chunk_*.parquet
-> data/analysis/node_flow_curve_fit/*.parquet
```

这说明“原始速度观测 -> 节点流量”链路是完整的、可复现的。

### 3.3 关键结论

网格化/池化/张量化 py 链路缺失，需要从 `预处理5.ipynb` 和 `预处理6.ipynb` 迁移。

## 4. notebook 与 py 的差异

### 4.1 `预处理5.ipynb` 当前仍保留的未迁移逻辑

`test/预处理5.ipynb` 中仍包含以下 notebook 专属逻辑：

- 读取 `1.路口节点经纬度.csv`；
- 读取 `4.路口节点车流量/section_chunk_00.csv` 等 CSV；
- 按 `grid_resolution = 0.009` 将节点经纬度映射到经纬网格；
- 对每个时间段构造双通道网格：
  - 通道 1：总车流量
  - 通道 2：平均车流量
- 将所有时间步保存为 `5.所有时间段网格化车流量_两通道.npy`；
- 使用 `torch.nn.functional.max_pool2d` 执行 2x2 max pooling；
- 保存 `5.所有时间段池化后的网格化车流量.npy`。

这些逻辑当前没有对应的正式 `.py` 文件，也没有落到 `data/processed/node_flow_grid/` 目录。

### 4.2 `预处理6.ipynb` 当前仍保留的未迁移逻辑

`test/预处理6.ipynb` 中仍包含以下 notebook 专属逻辑：

- 读取 `5.所有时间段池化后的网格化车流量.npy`；
- 提取每个字典中的 `pooled_grid_tensor`；
- 组装为 `np.ndarray`，形状打印为 `(5856, 2, 30, 21)`；
- 再 reshape 为 `torch.Size([2, 630, 5856])`；
- 最终保存为 `6.池化网格张量.pt`。

这一步也没有被迁移成正式 `.py` 文件，因此 `6.池化网格张量.pt` 目前仍属于 notebook 生成的历史产物命名，而不是工程化产物。

### 4.3 `6.池化网格张量.pt` 与 `node_flow_grid_tensor.pt` 的关系

- 二者不是当前仓库中“同一个已存在文件”，因为后者不存在；
- 但从角色和语义上看，它们应当属于同一类产物：
  - 都是“池化后的节点流量网格张量，供后续 CCN 类模型训练使用”；
  - 差别主要在于命名规范、保存目录与是否由正式 `.py` 脚本生成。

因此可以判断：

- `6.池化网格张量.pt` 是 notebook 阶段的历史命名；
- `node_flow_grid_tensor.pt` 应是工程化后的正式命名和正式训练入口文件。

## 5. 当前训练入口判断

### 5.1 对当前三个入口层级的判断

1. `data/analysis/node_intersection_flow_parquet/`

- 性质：上游节点流量中间数据；
- 作用：已足够支撑节点级统计、曲线拟合、日期类型分析、完整性检查；
- 不应直接作为最终 CCN 网格联邦训练入口。

2. `6.池化网格张量.pt`

- 性质：notebook 阶段的历史训练输入；
- 来源：`预处理5.ipynb` + `预处理6.ipynb`；
- 当前没有正式 `.py` 对应实现和规范输出目录。

3. `data/processed/node_flow_grid/node_flow_grid_tensor.pt`

- 性质：未来应补齐的正式工程化训练输入；
- 角色：应作为真实 CCN 网格联邦训练的最终 tensor-only 输入；
- 当前状态：文件不存在，但目标定位明确。

### 5.2 为什么当前单路口实验临时使用 parquet

当前单路口主实验和单路口消融实验直接读取 `node_intersection_flow_parquet`，是因为：

- 上游节点流量 parquet 已经真实存在、可审计、可复现；
- 正式网格化 / 池化 / `.pt` 张量脚本缺失；
- 正式 `node_flow_grid_tensor.pt` 也不存在；
- 为了先完成 Python 工程迁移、`FedAvg` 主线验证、指标和图表导出，只能用 parquet-direct 方式做 smoke test。

明确标注如下：

- 当前版本为 parquet-direct smoke test；
- 仅用于验证 py 工程结构、FedAvg、指标、可视化是否可运行；
- 不作为最终 CCN 网格联邦训练入口。

### 5.3 后续真实 CCN 联邦训练是否应改为 tensor-only

结论：是。

如果后续确认 `node_flow_grid_tensor.pt` 可重建，则真实 CCN 联邦训练应从：

```text
parquet-direct 输入
```

改为：

```text
tensor-only 输入
```

原因是：

- notebook 中的单路口 CCN 训练本来就是围绕池化网格张量组织；
- 网格化与池化体现了空间结构压缩，是 CCN 输入的重要前置；
- `node_intersection_flow_parquet` 只是节点级中间表，不是最终网格张量。

## 6. 后续建议

### 6.1 需要补写的正式脚本

#### 目标文件 1

`preprocessing_scripts/process_node_flow_grids.py`

功能：

- 从 `data/analysis/node_intersection_flow_parquet/` 读取节点流量；
- 结合节点经纬度；
- 映射到经纬度网格；
- 生成双通道网格特征；
- 保存 `node_flow_grid_2ch.npy`；
- 执行池化；
- 保存 `node_flow_grid_pooled.npy`；
- 可选保存 `node_flow_grid_regions.csv`。

#### 目标文件 2

`preprocessing_scripts/process_node_flow_tensor.py`

功能：

- 读取 `node_flow_grid_pooled.npy`；
- 整理为 `torch.Tensor`；
- 保存 `node_flow_grid_tensor.pt`；
- 记录 tensor `shape`、`dtype`、`NaN/Inf` 检查、时间步数、区域数。

### 6.2 对单路口实验代码的后续要求

本阶段不修改训练代码，但后续应明确把以下状态写入单路口实验说明：

- 当前版本为 parquet-direct smoke test；
- 仅用于验证 py 工程结构、FedAvg、指标、可视化是否可运行；
- 不作为最终 CCN 网格联邦训练入口。

一旦 `node_flow_grid_tensor.pt` 存在或可由正式脚本重建，则下一步应将：

- `single_intersection_client`
- `single_intersection_ablation`

从 parquet-direct 输入改为 tensor-only 输入。

## 7. 网格化与张量化代码补齐结果

| 脚本 | 是否生成 | 作用 | 输入 | 输出 |
|---|---|---|---|---|
| `preprocessing_scripts/process_node_flow_grids.py` | 是 | 节点流量网格化和池化 | `node_intersection_flow_parquet` | `node_flow_grid_2ch.npy` / `node_flow_grid_pooled.npy` |
| `preprocessing_scripts/process_node_flow_tensor.py` | 是 | 池化网格转 PyTorch tensor | `node_flow_grid_pooled.npy` | `node_flow_grid_tensor.pt` |

补齐情况说明：

- 本阶段已新增正式 Python 预处理脚本，承接 `test/预处理5.ipynb` 与 `test/预处理6.ipynb` 的工程化迁移；
- `process_node_flow_grids.py` 负责真实节点流量的网格化、平均池化、region sidecar 与 metadata 输出；
- `process_node_flow_tensor.py` 负责把池化网格从 `(T, C, H, W)` 转为正式训练输入 `(C, R, T)`；
- 本阶段仍未修改训练代码；
- `single_intersection_client` 和 `single_intersection_ablation` 仍然是 parquet-direct smoke test；
- 后续仍需要切换为 tensor-only 输入。

## 8. 关键问答结论

### Q1. 当前基础预处理步骤是否都已经从 ipynb 转为 py？

答：到“节点流量、曲线拟合、日期类型分析”这一层，基本已经转为 `.py`；但“网格化、池化、张量化”为止还没有完全转成 `.py`。

### Q2. 网格化与池化步骤是否已经有 py 代码？

答：没有正式 `.py` 代码。

### Q3. 是否存在 `process_node_flow_grids.py`？

答：不存在。

### Q4. 是否存在 `process_node_flow_tensor.py`？

答：不存在。

### Q5. 是否存在 `node_flow_grid_2ch.npy`？

答：不存在。

### Q6. 是否存在 `node_flow_grid_pooled.npy`？

答：不存在。

### Q7. 是否存在 `node_flow_grid_tensor.pt`？

答：不存在。

### Q8. notebook 中的 `6.池化网格张量.pt` 与 `node_flow_grid_tensor.pt` 是否是同一类产物？

答：是同一类训练输入产物，但不是当前仓库中的同一个文件。前者是 notebook 历史命名，后者应是工程化正式命名。

### Q9. 当前单路口实验直接读取 parquet 是否只是 fallback？

答：是，只是 parquet-direct smoke test fallback。

### Q10. 后续真实 CCN 联邦训练是否必须改为 tensor-only 输入？

答：从当前证据判断，应当改为 tensor-only 输入；`node_intersection_flow_parquet` 只应视为上游节点流量中间数据。
