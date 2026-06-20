# 图片资源缺失清单

## 1. 说明

本清单用于追踪真实数据缺失设置与插补恢复模块的图片资源缺口，覆盖 `g_mcar_pt`、`ntb_mix`、`nso_mix` 与 `snh_mix` 四个业务机制。清单按以下规则整理：

1. `缺失图片`：指当前业务分析有明确用途、且已有数据或脚本基础支持，但结果目录中尚未生成的图片资源。
2. `不适用`：指该机制本身不存在对应分析维度，例如 `g_mcar_pt` 不存在连续缺失长度组，因此不应计入缺口。
3. `有意不上传`：指图片文件实际存在，但与已上传图件构成重复信息或仅是泛化命名别名，不纳入正式上传集合；该类情况不计入缺失。

## 2. 缺失图片清单

| 业务模块 | 缺失图片资源 | 具体用途 | 缺失原因 | 证据与追踪依据 |
|---|---|---|---|---|
| `ntb_mix` | `ntb_mix_length_group_nrmse_by_method.png/pdf` | 用于补充 `ntb_mix` 在 short/mid/long 长度组下的归一化误差对比，支持跨长度组尺度归一分析 | 当前 `structured_missingness_imputation_pipeline.py` 仅生成长度组 `RMSE/MAE/sMAPE` 图，未实现长度组 `NRMSE` 图输出 | `structured_missingness_imputation_pipeline.py` 仅存在 `length_group_rmse/mae/smape` 输出逻辑 |
| `nso_mix` | `nso_mix_length_group_nrmse_by_method.png/pdf` | 用于补充 `nso_mix` 在 short/mid/long 长度组下的归一化误差对比，支持跨长度组尺度归一分析 | 当前 `structured_missingness_imputation_pipeline.py` 仅生成长度组 `RMSE/MAE/sMAPE` 图，未实现长度组 `NRMSE` 图输出 | `structured_missingness_imputation_pipeline.py` 仅存在 `length_group_rmse/mae/smape` 输出逻辑 |
| `snh_mix` | `snh_mix_spatial_vs_temporal_methods_rmse.png/pdf` | 用于直接比较空间方法族与时间方法族在 `snh_mix` 下的总体 RMSE 差异，属于高价值业务解释图 | `visualize_spatial_neighbor_holdout_results.py` 已定义 `save_spatial_vs_temporal_plot()` 与校验项，但 `main()` 未实际调用生成函数，导致文件缺失 | `snh_visualization_validation.csv/json` 中 `visualization_contains_spatial_methods = False`，并指向该文件名 |
| `snh_mix` | `snh_mix_flow_group_mae_by_method.png/pdf` | 用于分析不同流量层中绝对误差的分组稳健性 | 当前 `visualize_spatial_neighbor_holdout_results.py` 仅为 `flow_group` 输出 `RMSE` 图，未输出 `MAE` 图 | `visualize_spatial_neighbor_holdout_results.py` 在 `flow_group` 处只调用一次 `metric="rmse"` |
| `snh_mix` | `snh_mix_flow_group_smape_by_method.png/pdf` | 用于分析不同流量层中的相对误差稳健性 | 当前 `visualize_spatial_neighbor_holdout_results.py` 未为 `flow_group` 输出 `sMAPE` 图 | 同上 |
| `snh_mix` | `snh_mix_flow_group_nrmse_by_method.png/pdf` | 用于分析不同流量层中的归一化误差稳健性 | 当前 `visualize_spatial_neighbor_holdout_results.py` 未为 `flow_group` 输出 `NRMSE` 图 | 同上 |
| `snh_mix` | `snh_mix_length_group_mae_by_method.png/pdf` | 用于分析不同缺失跨度组中的绝对误差表现 | 当前 `visualize_spatial_neighbor_holdout_results.py` 仅为 `length_group` 输出 `RMSE` 图，未输出 `MAE` 图 | `visualize_spatial_neighbor_holdout_results.py` 在 `length_group` 处只调用一次 `metric="rmse"` |
| `snh_mix` | `snh_mix_length_group_smape_by_method.png/pdf` | 用于分析不同缺失跨度组中的相对误差表现 | 当前 `visualize_spatial_neighbor_holdout_results.py` 未为 `length_group` 输出 `sMAPE` 图 | 同上 |
| `snh_mix` | `snh_mix_length_group_nrmse_by_method.png/pdf` | 用于分析不同缺失跨度组中的归一化误差表现 | 当前 `visualize_spatial_neighbor_holdout_results.py` 未为 `length_group` 输出 `NRMSE` 图 | 同上 |

## 3. 不适用项说明

| 业务模块 | 图片类型 | 说明 |
|---|---|---|
| `g_mcar_pt` | `length_group_*` 系列图 | `g_mcar_pt` 为全局点级随机缺失，不存在连续缺失长度组，因此长度组图不适用，不计入缺口 |
| `g_mcar_pt` | `rmse_by_length_group_and_method` | 同上，不存在长度组二维结构 |

## 4. 有意不上传但不计入缺失的资源

### 4.1 `snh_mix` 冗余图件

以下 `snh_mix` 图件实际存在，但未纳入当前正式上传集合，原因是其信息已被表 8 与保留图件覆盖，不再额外增加新的业务解释维度：

| 图片资源 | 未上传原因 |
|---|---|
| `snh_mix_mae_by_method.png/pdf` | 与表 8 的 `MAE` 数值及 `Figure 5` 的总体方法排序共同构成重复信息 |
| `snh_mix_smape_by_method.png/pdf` | 与表 8 的 `sMAPE` 数值共同构成重复信息，未新增分组或结构层面的业务解释 |
| `snh_mix_nrmse_by_method.png/pdf` | 与表 8 的 `NRMSE` 数值共同构成重复信息，未新增分组或结构层面的业务解释 |

### 4.2 泛化命名别名图

以下资源属于与场景前缀图内容等价的泛化命名导出版本，为避免同一图片重复上传，不纳入正式上传集合，也不计入缺失：

| 业务模块 | 别名资源 | 正式上传版本 |
|---|---|---|
| `g_mcar_pt` | `multirate_rmse_by_method.png/pdf` 等 `multirate_*` 图 | `g_mcar_pt_*` 场景前缀图 |
| `ntb_mix` | `structured_multirate_*`、`structured_length_group_*` 图 | `ntb_mix_*` 场景前缀图与 `structured_rmse_by_length_group_and_method` |
| `nso_mix` | `outage_multirate_*`、`outage_length_group_*` 图 | `nso_mix_*` 场景前缀图与 `outage_rmse_by_length_group_and_method` |

## 5. 当前结论

1. `g_mcar_pt` 当前未发现真实缺失图片资源，场景专属的合格图片已可完整上传。
2. `ntb_mix` 与 `nso_mix` 的主要缺口集中在长度组 `NRMSE` 图，这属于结构化机制分析维度下的可补强资源。
3. `snh_mix` 的主要缺口集中在两类：一类是脚本已设计但未真正生成的“空间方法对时间方法”比较图；另一类是 `flow_group` 与 `length_group` 的非 `RMSE` 分组图。
4. `snh_mix` 已存在但未上传的 `MAE/sMAPE/NRMSE` 总体曲线不属于缺失，而是按“有明确业务用途、禁止冗余上传”的规则主动剔除。
