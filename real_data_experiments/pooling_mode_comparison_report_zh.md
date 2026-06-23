# 三种池化方案对比报告

## 1. 比较对象

本次比较三种池化方案：

- `avg`
- `max`
- `sum_mean`

比较输入全部来自已有 smoke test 结果目录：

- `data/processed/node_flow_grid/smoke_avg/`
- `data/processed/node_flow_grid/smoke_max/`
- `data/processed/node_flow_grid/smoke_sum_mean/`

可视化输出目录为：

- `results/pooling_mode_comparison/`

## 2. 网格与张量 shape 对比

三种方案的 shape 完全一致：

| pool_mode | raw_shape | pooled_shape | tensor_shape |
|---|---|---|---|
| `avg` | `(192, 2, 43, 60)` | `(192, 2, 21, 30)` | `(2, 630, 192)` |
| `max` | `(192, 2, 43, 60)` | `(192, 2, 21, 30)` | `(2, 630, 192)` |
| `sum_mean` | `(192, 2, 43, 60)` | `(192, 2, 21, 30)` | `(2, 630, 192)` |

说明：

- 正式布局采用 `standard`
- `row = lat`
- `col = lon`
- 因此 pooled shape 记为 `21 × 30`

## 3. 流量守恒对比

### 3.1 raw_vs_grid

三种方案均满足：

- `relative_difference_raw_vs_grid` 的均值与最大值均接近 `0`

这说明网格化阶段没有引入总流量损失。

### 3.2 grid_vs_pooled

对 smoke test 的 `grid_vs_pooled` 相对差异统计如下：

| pool_mode | `grid_vs_pooled` 均值 | `grid_vs_pooled` 最大值 | 解释 |
|---|---:|---:|---|
| `avg` | `0.75019526` | `0.75025030` | 平均池化会显著缩小 `channel 0 total_flow` |
| `max` | `0.55161506` | `0.55397261` | 与历史 notebook 一致，但 `channel 0` 不守恒 |
| `sum_mean` | `0.00078105` | `0.00100120` | 最接近流量守恒，仅有极小边界损失 |

## 4. 历史 notebook 一致性

历史 notebook `test/预处理5.ipynb` 明确使用：

- `F.max_pool2d`

因此：

- `max` 是最接近历史 notebook 的复刻方案

同时需要说明：

- 历史 notebook 的 pooled shape 显示为 `30 × 21`
- 当前正式工程化代码采用 `row=lat, col=lon`，因此显示为 `21 × 30`
- 二者只是空间轴转置关系，region 总数一致，均为 `630`

## 5. 交通物理意义

从交通物理意义看：

- `channel 0` 表示 `total_flow`
- `channel 1` 表示 `mean_flow`

若两者都使用 `avg`，则 `channel 0` 不再表达总流量；
若两者都使用 `max`，则 `channel 0` 变成局部最大流量，不再对应总量语义；
只有 `sum_mean` 同时保持：

- `channel 0` 仍表示 pooled 区域内的总流量
- `channel 1` 仍表示 pooled 区域内的平均流量

因此从物理意义和论文解释角度看，`sum_mean` 最合理。

## 6. 正式选择

正式结论如下：

- 正式 `pool_mode = sum_mean`
- 历史复刻对照 = `max`
- `avg` 不作为正式方案

原因排序如下：

1. 交通流量物理意义最清晰
2. `channel 0` 最接近守恒
3. 最便于论文解释
4. 同时保留 `max` 作为历史 notebook 对照复现

## 7. 空间布局选择

正式结论如下：

- 正式 `layout = standard`
- `row = lat`
- `col = lon`
- pooled shape = `21 × 30`

补充说明：

- 历史 notebook 的 `30 × 21` 仅作为转置对照
- 当前工程化代码不回退到 notebook 风格布局
- 正式输出统一采用标准布局

## 8. 下一步

将基于正式方案直接生成全量数据：

- 正式 `pool_mode = sum_mean`
- 正式 `layout = standard`
- 输出目录：`data/processed/node_flow_grid/final_sum_mean_standard/`

同时明确：

- 当前工程化代码不再生成 `6.池化网格张量.pt`
- 正式 tensor 输出统一为 `node_flow_grid_tensor.pt`
