# 真实数据缺失实验短路径目录

1. 本目录统一管理四类真实数据缺失与补全实验。
2. 新根目录使用 `results\rdm_exp`，用于降低 Windows 长路径风险。
3. 四类机制分别使用 `g_mcar_pt`、`ntb_mix`、`nso_mix`、`snh_mix`。
4. 每个 scenario 下分为 `miss_set` 和 `imp`。
5. `miss_set` 下保留 `masks`、`miss_data`、`manifests`、`audits`。
6. `imp` 下保留 `imp_data`、`summaries`、`figures`、`audits`、`manifests`。
7. `comparison` 存放综合对比图表与审计。
8. parquet 大文件不进入 Git。
9. 当前结果是缺失值补全误差，不是交通流预测误差。

## 缩写说明

- `rdm_exp = real data missingness experiments`
- `g_mcar_pt = global MCAR point`
- `ntb_mix = node temporal block, mixed short-mid-long`
- `nso_mix = node subset temporal outage, mixed short-mid-long`
- `snh_mix = spatial neighbor holdout, mixed short-mid-long`
- `miss_set = missingness setting`
- `miss_data = missing datasets`
- `imp = imputation`
- `imp_data = imputed datasets`
- `mf = mean fill`
- `ff = forward fill`
- `hle = historical linear extrapolation`
- `rtn = road-topology neighbor`
- `fcf = function curve fit`
- `ctn = correlation-topology neighbor`
- `ast = adaptive spatio-temporal fill`
- `atfh = adaptive topology-function hybrid`

## Scenario IDs

- `g_mcar_pt`
- `ntb_mix`
- `nso_mix`
- `snh_mix`

## 路径索引

- `experiment_registry.json`: `results\rdm_exp\experiment_registry.json`
- `path_aliases.json`: `results\rdm_exp\path_aliases.json`
- `path_cleanup`: `results\rdm_exp\path_cleanup`
- `comparison`: `results\rdm_exp\comparison`
- `comparison_spatial_extension`: `results\rdm_exp\comparison_spatial_extension`
