# 实验 1 Sanity Check 报告

## 1. 配置

- selected_clients: [290, 284, 318, 288, 289]
- tensor_path: `E:\Jupter_Notebook\FedTrafficFlow\data\processed\node_flow_grid\final_sum_mean_standard\node_flow_grid_tensor.pt`
- regions_path: `E:\Jupter_Notebook\FedTrafficFlow\data\processed\node_flow_grid\final_sum_mean_standard\node_flow_grid_regions.csv`
- sequence_length: 12
- prediction_horizon: 1
- use_channels: [0, 1]
- target_channel: 0
- split_bounds: train=[0, 4099), val=[4099, 4977), test=[4977, 5856)

## 2. 每个 selected client 的原始序列与标签统计

### region_id=290
- series: count=5856, min=1612853.375000, max=2059067.250000, mean=1953917.773630, std=86680.387676, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_train: count=4087, min=1612853.375000, max=2059067.250000, mean=1953937.349095, std=87783.061561, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_val: count=878, min=1727788.625000, max=2054093.250000, mean=1958253.120587, std=81168.315382, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_test: count=879, min=1666097.375000, max=2055649.375000, mean=1950864.085324, std=86472.107841, zero_ratio=0.000000, near_zero_ratio=0.000000

### region_id=284
- series: count=5856, min=1572611.125000, max=2056486.500000, mean=1914353.015390, std=107014.850501, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_train: count=4087, min=1572611.125000, max=2047440.500000, mean=1913453.369067, std=107283.149151, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_val: count=878, min=1610305.125000, max=2056486.500000, mean=1921556.545131, std=107358.687269, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_test: count=879, min=1601523.375000, max=2040950.000000, mean=1913543.223123, std=104093.382252, zero_ratio=0.000000, near_zero_ratio=0.000000

### region_id=318
- series: count=5856, min=1557965.875000, max=1981110.125000, mean=1857832.286608, std=94299.980310, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_train: count=4087, min=1564587.500000, max=1975647.125000, mean=1858012.956080, std=93488.025221, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_val: count=878, min=1557965.875000, max=1973020.125000, mean=1862805.912158, std=94675.212695, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_test: count=879, min=1605895.750000, max=1981110.125000, mean=1853425.456342, std=97212.544610, zero_ratio=0.000000, near_zero_ratio=0.000000

### region_id=288
- series: count=5856, min=1446559.375000, max=1819865.750000, mean=1702061.232923, std=78704.938531, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_train: count=4087, min=1446559.375000, max=1816716.500000, mean=1701367.259451, std=80709.932171, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_val: count=878, min=1487146.750000, max=1810842.750000, mean=1704929.719675, std=72097.133303, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_test: count=879, min=1480326.250000, max=1819865.750000, mean=1703952.361775, std=74741.359964, zero_ratio=0.000000, near_zero_ratio=0.000000

### region_id=289
- series: count=5856, min=1315587.625000, max=1716224.125000, mean=1659179.471333, std=38846.188319, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_train: count=4087, min=1437355.250000, max=1714012.375000, mean=1658798.012632, std=36915.564588, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_val: count=878, min=1315587.625000, max=1716224.125000, mean=1657715.081293, std=46325.096796, zero_ratio=0.000000, near_zero_ratio=0.000000
- y_test: count=879, min=1464554.750000, max=1709628.750000, mean=1662198.565415, std=39462.840147, zero_ratio=0.000000, near_zero_ratio=0.000000

## 3. 历史 prediction_samples 审计

- prediction_samples rows=200, methods=['FedAvg', 'Independent']
- FedAvg y_true: count=100, min=1756842.125000, max=2035559.125000, mean=1929791.622500, std=75288.031963, zero_ratio=0.000000, near_zero_ratio=0.000000
- FedAvg y_pred: count=100, min=1753243.875000, max=2012818.125000, mean=1923072.961250, std=68851.766385, zero_ratio=0.000000, near_zero_ratio=0.000000
- FedAvg error: count=100, min=-71725.875000, max=41513.375000, mean=-6718.661250, std=17807.773726, zero_ratio=0.000000, near_zero_ratio=0.000000
- Independent y_true: count=100, min=1756842.125000, max=2035559.125000, mean=1929791.622500, std=75288.031963, zero_ratio=0.000000, near_zero_ratio=0.000000
- Independent y_pred: count=100, min=1739478.875000, max=2035755.500000, mean=1929680.030000, std=74415.624779, zero_ratio=0.000000, near_zero_ratio=0.000000
- Independent error: count=100, min=-73565.375000, max=47968.250000, mean=-111.592500, std=15956.931233, zero_ratio=0.000000, near_zero_ratio=0.000000

## 4. 现有方法与 naive baseline 对比

| method         |         mse |    rmse |     mae |     mape |    smape |       r2 |
|:---------------|------------:|--------:|--------:|---------:|---------:|---------:|
| FedAvg         | 4.82029e+08 | 21669.2 | 17419.8 | 0.976708 | 0.97426  | 0.8641   |
| Independent    | 2.25198e+08 | 14709.8 | 10507.9 | 0.587672 | 0.58694  | 0.962633 |
| NaiveLastValue | 3.97908e+08 | 19419.2 | 13619.9 | 0.758148 | 0.758963 | 0.938585 |

## 5. 结论

- 如果 `y_true` 为百万级而模型 `y_pred` 接近小常数，则优先排查目标归一化与尺度一致性。
- 如果 naive baseline 明显优于现有方法，则说明当前训练或标签口径存在明显问题，不能直接进入正式重跑。
