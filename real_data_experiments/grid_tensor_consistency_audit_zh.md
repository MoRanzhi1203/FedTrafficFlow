# 网格张量一致性审计

## 1. 审计范围

本次一致性审计对比以下四个来源：

- `test/预处理5.ipynb`
- `test/预处理6.ipynb`
- `preprocessing_scripts/process_node_flow_grids.py`
- `preprocessing_scripts/process_node_flow_tensor.py`

目标是回答当前 smoke test 的空间区域数是否与历史 notebook 一致，并在必要时修正正式预处理脚本。

## 2. 历史 notebook 证据

### 2.1 `test/预处理5.ipynb`

notebook 中明确给出：

- `grid_resolution = 0.009`
- `lon_min = 116.100694`
- `lon_max = 116.648434`
- `lat_min = 39.748070`
- `lat_max = 40.138793`
- `lon_grid_count = int((lon_max - lon_min) / grid_resolution)`
- `lat_grid_count = int((lat_max - lat_min) / grid_resolution)`
- 池化方式：`F.max_pool2d`
- `kernel_size = 2`
- `stride = 2`

按上述 notebook 公式计算：

- `lon_grid_count = int((116.648434 - 116.100694) / 0.009) = 60`
- `lat_grid_count = int((40.138793 - 39.748070) / 0.009) = 43`

因此，历史 notebook 的原始网格张量空间尺寸应为：

- 原始单时刻网格：`(2, 60, 43)`，其中 notebook 实际先按 `(lon_grid_count, lat_grid_count)` 建网格
- 若写成批量形式：`(T, 2, 60, 43)`

同一 notebook 后续又打印出：

- `池化网格张量形状: (2, 30, 21)`

因此，历史 notebook 的池化后空间尺寸为：

- 单时刻池化网格：`(2, 30, 21)`
- 批量形式：`(T, 2, 30, 21)`

### 2.2 `test/预处理6.ipynb`

notebook 中打印：

- `池化网格张量的形状: (5856, 2, 30, 21)`
- `调整后的数据形状: (2, 630, 5856)`

因此，历史 notebook 的 tensor 结论是：

- 池化批量 shape：`(5856, 2, 30, 21)`
- 正式训练 tensor shape：`(2, 630, 5856)`

同时该 notebook 确实曾保存：

- `6.池化网格张量.pt`

但这是 notebook 阶段的临时命名，不应继续保留到工程化脚本中。

## 3. 当前脚本审计与修正

### 3.1 修正前问题

此前 `process_node_flow_grids.py` 的 smoke test 产物为：

- `raw_shape = (192, 2, 44, 61)`
- `pooled_shape = (192, 2, 22, 30)`
- `tensor_shape = (2, 660, 192)`

其主要原因有两点：

1. 网格计数使用了 `floor(range / resolution) + 1`
2. 正式脚本按 `row=lat, col=lon` 组织空间维度，而历史 notebook 按 `lon_grid_count, lat_grid_count` 的顺序直接建二维矩阵

第 1 点会把历史 notebook 的 `60 x 43` 扩展成 `61 x 44`，从而把 pooled region 数从 `630` 放大为 `660`。

### 3.2 已执行修正

本阶段已把 `process_node_flow_grids.py` 修正为：

- `grid_resolution` 仍为 `0.009`
- 默认 `pool_mode = max`
- 网格计数回到 notebook 公式：`int((max - min) / resolution)`
- 对上边界点继续保留 clip 策略，而不是额外 `+1`
- 新增三种池化模式：
  - `avg`
  - `max`
  - `sum_mean`
- 新增 `node_flow_grid_flow_audit.csv`
- 新增 `active_region_count / empty_region_count / active_region_ratio`
- `node_flow_grid_regions.csv` 新增 `is_active_region`

### 3.3 当前正式脚本的空间定义

修正后的 Python 脚本采用更标准的张量记法：

- `grid_row = lat`
- `grid_col = lon`

因此，修正后的批量 raw grid shape 为：

- `raw_shape = (T, 2, 43, 60)`

而不是历史 notebook 直接打印的 `(T, 2, 60, 43)`。

这两者的区别主要是空间轴交换，不影响：

- 总区域数
- 展平后的 `R`
- 后续 `(C, R, T)` tensor 规模

因此，修正后的 pooled shape 为：

- `pooled_shape = (T, 2, 21, 30)`

与历史 notebook 的 `(T, 2, 30, 21)` 相比，也是空间轴转置关系；两者的 region 数都等于 `630`。

## 4. 三种 pooling smoke test 结果

本次 smoke test 均采用：

- `--max-chunks 2`
- `T = 192`
- `grid_resolution = 0.009`
- `pool_kernel = 2`
- `pool_stride = 2`

输出目录分别为：

- `data/processed/node_flow_grid/smoke_avg/`
- `data/processed/node_flow_grid/smoke_max/`
- `data/processed/node_flow_grid/smoke_sum_mean/`

### 4.1 结果汇总

| pool_mode | raw_shape | pooled_shape | tensor_shape | active_region_count | empty_region_count | finite | 说明 |
|---|---|---|---|---:|---:|---|---|
| `avg` | `(192, 2, 43, 60)` | `(192, 2, 21, 30)` | `(2, 630, 192)` | 223 | 407 | `True` | 两通道都做 average pooling；channel 0 总量不守恒。 |
| `max` | `(192, 2, 43, 60)` | `(192, 2, 21, 30)` | `(2, 630, 192)` | 223 | 407 | `True` | 复刻历史 notebook 的 `F.max_pool2d`。 |
| `sum_mean` | `(192, 2, 43, 60)` | `(192, 2, 21, 30)` | `(2, 630, 192)` | 223 | 407 | `True` | `channel 0=sum pooling`，`channel 1=average pooling`，最接近流量守恒。 |

### 4.2 流量守恒审计

三种模式的 `node_flow_grid_flow_audit.csv` 统计结果如下：

| pool_mode | `max(relative_difference_raw_vs_grid)` | `max(relative_difference_grid_vs_pooled)` | `mean(relative_difference_grid_vs_pooled)` | 解释 |
|---|---:|---:|---:|---|
| `avg` | `0.0` | `0.75025030` | `0.75019526` | grid 阶段保持总量，但 average pooling 会把 channel 0 总量显著缩小。 |
| `max` | `0.0` | `0.55397261` | `0.55161506` | 与历史 notebook 一致，但 channel 0 总量不守恒。 |
| `sum_mean` | `0.0` | `0.00100120` | `0.00078105` | 只有极小差异，主要来自 odd dimension 下最后一行未进入 `2x2` 窗口。 |

结论：

- `raw_input_total_flow` 与 `grid_total_flow_channel0_sum` 完全一致，说明网格化前后总量没有损失；
- 差异主要发生在池化阶段；
- 从“复刻历史 notebook”看，`max` 最一致；
- 从“channel 0 流量守恒”看，`sum_mean` 最合理。

## 5. 关键问题逐项回答

### 5.1 历史 notebook 中 raw grid shape 是多少？

- 按 notebook 的建网格方式，历史 raw grid 应为 `(T, 2, 60, 43)`；
- 若按标准 `H=row(lat), W=col(lon)` 解释，同一份空间网格等价于 `(T, 2, 43, 60)`。

### 5.2 历史 notebook 中 pooled grid shape 是多少？

- notebook 直接打印结果为 `(5856, 2, 30, 21)`。

### 5.3 历史 notebook 中是否使用 `max_pool2d`？

- 是，`test/预处理5.ipynb` 明确调用 `F.max_pool2d(...)`。

### 5.4 当前 py 脚本为什么曾得到 `44 x 61`？

- 原因是使用了 `floor(range / resolution) + 1`；
- 同时采用 `row=lat, col=lon` 的标准空间顺序；
- 这会把历史 notebook 的 `43 x 60` 扩成 `44 x 61`。

### 5.5 当前 py 脚本为什么曾 pooled 后得到 `22 x 30`？

- 因为原始空间尺寸被放大为 `44 x 61`；
- 在 `kernel=2, stride=2` 下，输出会变成 `22 x 30`；
- 对应 region 数为 `660`，比 notebook 的 `630` 多出 `30` 个区域。

### 5.6 是否应当得到历史预期的 `29 x 32` 或其他尺寸？

- 从当前 notebook 证据看，不应得到 `29 x 32`；
- 历史 notebook 明确指向的是 `30 x 21`，即总 region 数 `630`；
- 当前工程化脚本采用标准 `row/col` 顺序后，对应 pooled shape 为 `21 x 30`，本质上与 notebook 一致，只是空间轴转置。

### 5.7 当前 `660` 个 region 是否符合论文实验原意？

- 不符合；
- notebook 证据显示历史预期 region 数是 `630`，不是 `660`；
- 本阶段已修正为 `630`。

### 5.8 是否存在经纬度边界、网格分辨率、池化方式或坐标来源差异？

- 存在，具体如下：
- 经纬度边界：当前正式脚本与 notebook 一致，均来自 `116.100694 ~ 116.648434`、`39.748070 ~ 40.138793`
- 网格分辨率：一致，均为 `0.009`
- 网格计数：已修正为 notebook 公式，不再额外 `+1`
- 池化方式：历史 notebook 为 `max_pool2d`；当前脚本已新增 `avg/max/sum_mean`
- 坐标来源：当前正式脚本使用 `rnsd_processed.csv + link_gps_processed.csv` 重建真实节点坐标，因为 `rnsd_processed.csv` 自带的 `start_lon/start_lat/end_lon/end_lat` 是相对偏移，不是可直接使用的真实经纬度

## 6. 张量化脚本一致性检查

`process_node_flow_tensor.py` 已补充并修正为：

- 支持 `--input-path`
- 支持 `--metadata-path`
- 支持 `--regions-path`
- 支持 `--output-dir`
- 支持 `--output-name`
- 默认输出仍为 `node_flow_grid_tensor.pt`
- 已删除历史兼容副本逻辑
- 已删除 `--write-legacy-copy` / `--save-legacy-copy` 类型参数
- 当前工程化代码不再生成 `6.池化网格张量.pt`

## 7. 历史命名处理

历史 notebook 中曾使用 `6.池化网格张量.pt` 作为临时命名；
当前工程化代码不再生成该文件；
正式输出统一为 `node_flow_grid_tensor.pt`。

## 8. 审计结论

1. 当前 `660` 个区域不合理，已修正为 `630`。
2. 修正后的 region 数与历史 notebook 一致；当前 `21 x 30` 与 notebook 的 `30 x 21` 属于空间轴转置，不属于区域规模不一致。
3. 正式默认方案改为 `pool_mode=sum_mean`，因为它在保留物理语义的同时最接近 `channel 0` 总量守恒。
4. `max` 仅保留为历史 notebook 的复刻对照，不作为正式默认方案。
5. 正式布局固定为 `standard`，即 `row=lat, col=lon`，pooled shape 为 `21 x 30`。
6. 正式全量数据输出目录为 `data/processed/node_flow_grid/final_sum_mean_standard/`。
7. 当前工程化代码不再生成 `6.池化网格张量.pt`，正式 tensor 输出统一为 `node_flow_grid_tensor.pt`。
8. 从空间区域数一致性和 pooling 方案审计看，可以进入全量网格生成；训练入口切换到 tensor-only 仍应放到后续阶段单独实施。

正式全量数据已生成并通过校验：

- `raw_shape = (5856, 2, 43, 60)`
- `pooled_shape = (5856, 2, 21, 30)`
- `tensor_shape = (2, 630, 5856)`
- `dtype = float32`
- `finite = True`

## 9. 待作者确认

- 正式全量导出后，是否立即在下一阶段把训练入口切换到 `final_sum_mean_standard/node_flow_grid_tensor.pt`
- 论文正文中是否需要单独保留一段“历史 notebook 复刻方案 = max”的补充说明
- 是否需要对 `sum_mean` 再补一张更强调“channel 0 守恒”的论文插图
