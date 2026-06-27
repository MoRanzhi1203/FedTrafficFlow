# 实验 1：grid_cell main full v4 CUDA Sanity Check 报告

## 1. 配置

- selected_clients: `[290, 284, 318, 288, 289]`
- tensor_path: `E:\Jupter_Notebook\FedTrafficFlow\data\processed\node_flow_grid\final_sum_mean_standard\node_flow_grid_tensor.pt`
- regions_path: `E:\Jupter_Notebook\FedTrafficFlow\data\processed\node_flow_grid\final_sum_mean_standard\node_flow_grid_regions.csv`
- sequence_length: `12`
- prediction_horizon: `1`
- use_channels: `[0, 1]`
- target_channel: `0`
- split_bounds: `train=[0, 4099), val=[4099, 4977), test=[4977, 5856)`

## 2. 原始标签尺度检查

- 5 个 selected clients 的原始 `series / y_train / y_val / y_test` 都处于百万级，且不存在 `zero_ratio` 或 `near_zero_ratio` 异常。
- 这说明实验 1 当前使用的数据切分口径与既往 v3 审计一致，没有出现标签量纲错位或异常清零。

## 3. prediction_samples 审计

- `prediction_samples.csv` 行数：`200`
- methods：`['FedAvg', 'Independent']`
- `FedAvg y_true`：count=`100`，min=`1756842.125000`，max=`2035559.125000`，mean=`1929791.622500`，std=`75288.031963`
- `FedAvg y_pred`：count=`100`，min=`1753121.375000`，max=`2014655.500000`，mean=`1924139.128750`，std=`69738.670642`
- `FedAvg error`：count=`100`，min=`-71786.875000`，max=`42151.375000`，mean=`-5652.493750`，std=`17625.908971`
- `Independent y_true`：count=`100`，min=`1756842.125000`，max=`2035559.125000`，mean=`1929791.622500`，std=`75288.031963`
- `Independent y_pred`：count=`100`，min=`1737321.000000`，max=`2035786.875000`，mean=`1930237.460000`，std=`74349.591287`
- `Independent error`：count=`100`，min=`-66951.250000`，max=`48148.125000`，mean=`445.837500`，std=`15757.219898`
- unique `y_pred` count：
  - `FedAvg = 100`
  - `Independent = 100`

结论：

- `y_true / y_pred` 同尺度，均为原始百万级流量尺度。
- `FedAvg` 与 `Independent` 均不再是常数预测。
- 预测误差分布合理，未见 NaN / Inf 或明显广播错误痕迹。

## 4. 方法对比

| method | MSE | RMSE | MAE | MAPE | SMAPE | R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FedAvg | 445713657.805219 | 20815.803975 | 16604.522810 | 0.932170 | 0.929825 | 0.873045 |
| Independent | 224696815.731175 | 14693.368581 | 10501.618629 | 0.587359 | 0.586664 | 0.962666 |
| NaiveLastValue | 397907900.000000 | 19419.217079 | 13619.880887 | 0.758148 | 0.758963 | 0.938585 |

## 5. sanity 结论

- `FedAvg`：R2 为正，但仍落后于 `NaiveLastValue`，说明主线修复后训练有效，但联邦主方法尚未达到项目要求的基准线之上。
- `Independent`：显著优于 `NaiveLastValue`，说明当前数据、尺度和单客户端训练链路是健康的。
- 是否仍有常数预测：`否`
- 是否通过基础 sanity check：`是`
- 是否建议进入实验 2：`否`

原因：

- 当前 CUDA v4 已通过“设备正确、尺度正确、无 NaN/Inf、无常数预测、R2 为正、收敛正常”的基础 sanity。
- 但项目主线标准要求实验 1 的正式 CUDA 结果应先达到或优于 `NaiveLastValue`，而当前 `FedAvg` 仍未满足该条件。
