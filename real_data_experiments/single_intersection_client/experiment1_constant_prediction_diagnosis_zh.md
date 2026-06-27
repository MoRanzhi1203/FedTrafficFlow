# 实验 1 常数预测诊断报告

## 1. 诊断配置

- device: cpu
- selected_clients: [290, 284, 318, 288, 289]
- sequence_length: 12
- batch_size: 32
- learning_rate: 0.001
- target_scaler.mean: 1817113.789265
- target_scaler.std: 144200.062559

## 2. 数据集与目标统计

### client_id=0, region_id=290
- X train shape: (2, 12)
- y train shape: (1,)
- train sample count: 4087
- val sample count: 878
- test sample count: 879
- X_train: min=2454.869141, max=2045735.500000, mean=967468.297189, std=967738.902620
- y_train: min=1612853.375000, max=2059067.250000, mean=1953937.349095, std=87783.061561
- y_val: min=1727788.625000, max=2054093.250000, mean=1958253.120587, std=81168.315382
- y_test: min=1666097.375000, max=2055649.375000, mean=1950864.085324, std=86472.107841
- y_train_norm: min=-1.416507, max=1.677901, mean=0.948845, std=0.608759
- y_val_norm: min=-0.619453, max=1.643407, mean=0.978774, std=0.562887
- y_test_norm: min=-1.047270, max=1.654199, mean=0.927533, std=0.599668

### client_id=1, region_id=284
- X train shape: (2, 12)
- y train shape: (1,)
- train sample count: 4087
- val sample count: 878
- test sample count: 879
- X_train: min=2314.530762, max=2037315.250000, mean=949246.828402, std=950288.286670
- y_train: min=1572611.125000, max=2047440.500000, mean=1913453.369067, std=107283.149151
- y_val: min=1610305.125000, max=2056486.500000, mean=1921556.545131, std=107358.687269
- y_test: min=1601523.375000, max=2040950.000000, mean=1913543.223123, std=104093.382252
- y_train_norm: min=-1.695579, max=1.597272, mean=0.668097, std=0.743988
- y_val_norm: min=-1.434179, max=1.660004, mean=0.724291, std=0.744512
- y_test_norm: min=-1.495079, max=1.552262, mean=0.668720, std=0.721868

### client_id=2, region_id=318
- X train shape: (2, 12)
- y train shape: (1,)
- train sample count: 4087
- val sample count: 878
- test sample count: 879
- X_train: min=2338.545166, max=1957391.375000, mean=918426.283536, std=918238.596702
- y_train: min=1564587.500000, max=1975647.125000, mean=1858012.956080, std=93488.025221
- y_val: min=1557965.875000, max=1973020.125000, mean=1862805.912158, std=94675.212695
- y_test: min=1605895.750000, max=1981110.125000, mean=1853425.456342, std=97212.544610
- y_train_norm: min=-1.751222, max=1.099399, mean=0.283628, std=0.648322
- y_val_norm: min=-1.797141, max=1.081181, mean=0.316866, std=0.656555
- y_test_norm: min=-1.464757, max=1.137283, mean=0.251815, std=0.674151

### client_id=3, region_id=288
- X train shape: (2, 12)
- y train shape: (1,)
- train sample count: 4087
- val sample count: 878
- test sample count: 879
- X_train: min=2117.024902, max=1803878.250000, mean=842247.936428, std=842168.770018
- y_train: min=1446559.375000, max=1816716.500000, mean=1701367.259451, std=80709.932171
- y_val: min=1487146.750000, max=1810842.750000, mean=1704929.719675, std=72097.133303
- y_test: min=1480326.250000, max=1819865.750000, mean=1703952.361775, std=74741.359964
- y_train_norm: min=-2.569724, max=-0.002755, mean=-0.802680, std=0.559708
- y_val_norm: min=-2.288259, max=-0.043488, mean=-0.777975, std=0.499980
- y_test_norm: min=-2.335558, max=0.019084, mean=-0.784753, std=0.518317

### client_id=4, region_id=289
- X train shape: (2, 12)
- y train shape: (1,)
- train sample count: 4087
- val sample count: 878
- test sample count: 879
- X_train: min=2284.690430, max=1704464.000000, mean=834789.703327, std=832555.358321
- y_train: min=1437355.250000, max=1714012.375000, mean=1658798.012632, std=36915.564588
- y_val: min=1315587.625000, max=1716224.125000, mean=1657715.081293, std=46325.096796
- y_test: min=1464554.750000, max=1709628.750000, mean=1662198.565415, std=39462.840147
- y_train_norm: min=-2.633553, max=-0.714989, mean=-1.097890, std=0.256002
- y_val_norm: min=-3.477989, max=-0.699651, mean=-1.105400, std=0.321256
- y_test_norm: min=-2.444930, max=-0.745388, mean=-1.074308, std=0.273667

## 3. 单 batch 前向检查

- batch x shape: (32, 2, 12)
- batch y shape: (32, 1)
- batch pred shape: (32, 1)
- batch_x: min=2580.486084, max=2038832.375000, mean=941291.897782, std=941281.948362
- batch_y_norm: min=-0.521861, max=1.537576, mean=0.693973, std=0.809989
- batch_pred_norm_before_train: min=-0.144884, max=-0.128980, mean=-0.141935, std=0.002987
- batch_pred_denorm_before_train: min=1796221.500000, max=1798514.750000, mean=1796646.757812, std=430.774164
- single batch loss before train: 1.354486

## 4. mini-batch 更新检查

| step | loss | grad_norm | update_norm | pred_mean_norm | pred_std_norm | pred_mean_denorm | pred_std_denorm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.354486 | 4.300751 | 0.005483 | -0.133332 | 0.003033 | 1797887.125000 | 437.331909 |
| 2 | 1.340182 | 4.255899 | 0.005477 | -0.124735 | 0.003030 | 1799127.000000 | 436.977997 |
| 3 | 1.326037 | 4.211094 | 0.005469 | -0.116146 | 0.003028 | 1800365.750000 | 436.613892 |
| 4 | 1.312052 | 4.166347 | 0.005459 | -0.107567 | 0.003025 | 1801602.750000 | 436.245667 |
| 5 | 1.298231 | 4.121671 | 0.005450 | -0.099000 | 0.003023 | 1802838.000000 | 435.880981 |

## 5. 初步诊断结论

- 若 `grad_norm` 和 `update_norm` 持续大于 0，则说明训练循环与参数更新并未完全失效。
- 若 `pred_std_denorm` 始终很小，而 `batch_x` 量级远大于 `y_norm`，则说明模型更可能在大尺度输入上退化为近常数输出。
- 本报告只提供运行时证据，不直接修改正式训练逻辑。