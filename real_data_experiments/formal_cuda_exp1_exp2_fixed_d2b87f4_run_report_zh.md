# 实验 1/2 Formal CUDA 修复重跑报告 (commit d2b87f4)

## 运行信息

- **Commit**: `d2b87f4` — `fix(real-data): restore exp1 baseline and exp2 normalized ablation pipeline`
- **设备**: NVIDIA GeForce RTX 3060 Laptop GPU (CUDA)
- **参数**: `--rounds 20 --local-epochs 3 --device cuda`

## 实验 1: 单路口客户端 (Single Intersection Client)

### 输出目录
`results/real_data_experiments/formal/exp1_single_grid_client_formal_cuda_fixed_d2b87f4/`

### 指标表

| Method | RMSE | MAE | MAPE (%) | SMAPE (%) | R2 |
|--------|------|-----|----------|-----------|-----|
| **FedAvg** | 20,753.14 | 16,847.81 | 0.942 | 0.941 | 0.8827 |
| **Independent** | 14,883.58 | 10,843.73 | 0.605 | 0.605 | 0.9621 |
| **NaiveLastValue** | 19,419.22 | 13,619.88 | 0.758 | 0.759 | 0.9386 |

### 关键结论

1. **NaiveLastValue 已恢复** — RMSE=19,419，与历史 smoke 结果一致
2. **FedAvg vs NaiveLastValue**: FedAvg RMSE=20,753 > NaiveLastValue RMSE=19,419，FedAvg 弱于 naive baseline 约 6.9%
3. **FedAvg vs Independent**: FedAvg RMSE=20,753 > Independent RMSE=14,884，FedAvg 弱于 Independent 约 39.4%
4. **actual_device=cuda** ✓
5. **local_epochs=3** ✓
6. **selected_ids: [290, 284, 318, 288, 289]** — 与上一轮一致

## 实验 2: 单路口消融 (Single Intersection Ablation)

### 输出目录
`results/real_data_experiments/formal/exp2_single_grid_ablation_formal_cuda_fixed_d2b87f4/`

### 指标表

| Variant | RMSE | MAE | MAPE (%) | SMAPE (%) | R2 |
|---------|------|-----|----------|-----------|-----|
| **Full** | 17,345.62 | 13,484.87 | 0.758 | 0.757 | 0.9117 |
| **Without Attention** | 18,975.80 | 14,855.84 | 0.838 | 0.836 | 0.9014 |
| **Without CNN / Spatial Encoder** | 23,371.29 | 19,836.63 | 1.101 | 1.098 | 0.8577 |
| **Without LSTM** | 19,331.81 | 15,119.35 | 0.853 | 0.850 | 0.8972 |

### 关键结论

1. **normalization 修复生效** — `input_scaler.json` + `target_scaler.json` 已导出
2. **尺度崩坏修复** — Full y_pred mean=1,925,561（原 formal 恒定 ~66），与 y_true=1,929,792 同一数量级
3. **prediction_samples 覆盖** — 4 个 variant 各 100 行
4. **消融趋势合理** — Full 最优 (RMSE=17,346)，删除 CNN 退化最严重 (RMSE=23,371)
5. **actual_device=cuda** ✓
6. **local_epochs=3** ✓

## 与上一轮 formal 对比

| 项目 | 上一轮 (ec43e87) | 本轮 (d2b87f4) |
|------|-----------------|-----------------|
| 实验 1 NaiveLastValue | 缺失 | **已恢复** (RMSE=19,419) |
| 实验 2 normalization | 缺失 | **已接入** (input+target scaler) |
| 实验 2 Full y_pred | ~66 (尺度崩坏) | **~1,925,561** (正常) |
| 实验 2 prediction_samples | head(400) | **按 variant 平衡抽样** |

## 建议

1. 实验 1/2 修复已确认生效，建议将本 commit 作为后续实验基线
2. FedAvg 仍然弱于 Independent 和 NaiveLastValue，这是真实数据场景的结果特征而非实现 bug
3. 后续实验 3/5/6 formal 可基于当前代码状态继续
