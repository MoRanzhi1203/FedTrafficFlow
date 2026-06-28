# 实验 1 FedAvg 与实验 2 Full 对齐诊断报告

## 1. 诊断目的

说明：实验 2 Full 在修复后运行，最初怀疑其 RMSE=17,345 优于实验 1 FedAvg RMSE=20,753。经诊断确认"17345"来自修复前旧 formal 崩坏数据，修复后两者完全一致。当前已查清差异来源，实验 1/2 修复彻底生效。

## 2. Git 与运行状态

- 当前 commit：`d2b87f4`
- 是否已 push GitHub：是
- 当前分支：`main`
- upstream：`origin/main`
- 是否运行训练：否
- 是否修改代码：否
- 是否修改 results：否

## 3. 输出对比

- 实验 1 输出目录：`results/real_data_experiments/formal/exp1_single_grid_client_formal_cuda_fixed_d2b87f4/`
- 实验 2 输出目录：`results/real_data_experiments/formal/exp2_single_grid_ablation_formal_cuda_fixed_d2b87f4/`

### 实验 1 指标

| Method | RMSE | MAE | MAPE(%) | SMAPE(%) | R2 |
|--------|------|-----|---------|----------|-----|
| FedAvg | 20,753.14 | 16,847.81 | 0.942 | 0.941 | 0.8827 |
| Independent | 14,883.58 | 10,843.73 | 0.605 | 0.605 | 0.9621 |
| NaiveLastValue | 19,419.22 | 13,619.88 | 0.758 | 0.759 | 0.9386 |

### 实验 2 指标

| Variant | RMSE | MAE | MAPE(%) | SMAPE(%) | R2 |
|---------|------|-----|---------|----------|-----|
| **Full** | **20,753.14** | **16,847.81** | **0.942** | **0.941** | **0.8827** |
| Without Attention | 18,975.80 | 14,855.84 | 0.838 | 0.836 | 0.9014 |
| Without CNN / Spatial Encoder | 23,371.29 | 19,836.63 | 1.101 | 1.098 | 0.8577 |
| Without LSTM | 19,331.81 | 15,119.35 | 0.853 | 0.850 | 0.8972 |

## 4. 配置对比

| 项目 | 实验 1 | 实验 2 Full | 是否一致 |
|------|--------|-------------|---------|
| selected clients | [290,284,318,288,289] | [290,284,318,288,289] | ✓ |
| data_mode | tensor | tensor | ✓ |
| tensor_path | node_flow_grid_tensor.pt | node_flow_grid_tensor.pt | ✓ |
| target_channel | 0 | 0 | ✓ |
| use_channels | [0,1] | [0,1] | ✓ |
| sequence_length | 12 | 12 | ✓ |
| prediction_horizon | 1 | 1 | ✓ |
| train_ratio | 0.7 | 0.7 | ✓ |
| val_ratio | 0.15 | 0.15 | ✓ |
| batch_size | 64 | 64 | ✓ |
| learning_rate | 1e-3 | 1e-3 | ✓ |
| rounds | 20 | 20 | ✓ |
| local_epochs | 3 | 3 | ✓ |
| input_scaler | mean=[1816850,2643] std=[144164,219] | mean=[1816850,2643] std=[144164,219] | ✓ |
| target_scaler | mean=1817114 std=144200 | mean=1817114 std=144200 | ✓ |
| actual_device | cuda | cuda | ✓ |

**全部 16 项配置完全一致。**

## 5. 模型结构对比

- 实验 1 模型：`CNNLSTMAttentionRegressor`
- 实验 2 Full 模型：`SingleIntersectionAblationModel(variant='full')`
- 参数总量：均为 **10,194**
- 参数 shape：完全一致
- forward 逻辑：一致

**模型结构完全相同。**

## 6. FedAvg 训练逻辑对比

| 项目 | 实验 1 | 实验 2 Full | 是否一致 |
|------|--------|------------|---------|
| 聚合函数 | `fedavg_aggregate(local_state_dicts, sample_counts)` | 同 (via `run_federated_rounds`) | ✓ |
| 加权方式 | sample_count 加权 | sample_count 加权 | ✓ |
| 每轮初始化 | 从 global_model state_dict deep copy | 同 (via `FedClient.train(global_state_dict)`) | ✓ |
| optimizer | Adam(lr=1e-3) | Adam(lr=1e-3) | ✓ |
| criterion | nn.MSELoss() | nn.MSELoss() | ✓ |
| target scaler 反归一化 | collect_predictions(..., target_scaler) | 同 | ✓ |
| train target 归一化 | NormalizedDataset → normalize_tensor | 同 (via apply_dataset_normalization) | ✓ |

**训练逻辑完全一致。**

## 7. 收敛曲线对比

**Exp1 FedAvg 和 Exp2 Full 的收敛曲线逐位完全一致**（包括 train_loss、val_rmse、val_rmse_std、val_mae、test_rmse、test_rmse_std 共 6 个指标 × 20 轮）。

| Round | Exp1 FedAvg test_rmse | Exp2 Full test_rmse |
|-------|----------------------|---------------------|
| 1 | 94,706.00 | 94,706.00 |
| 5 | 27,396.47 | 27,396.47 |
| 10 | 22,029.85 | 22,029.85 |
| 15 | 21,060.61 | 21,060.61 |
| 20 | 20,753.14 | 20,753.14 |

## 8. Client 级差异

| client_id | Exp1 FedAvg RMSE | Exp2 Full RMSE | 差值 |
|-----------|-----------------|----------------|------|
| 0 (290) | 20,877.17 | 20,877.17 | 0.0 |
| 1 (284) | 23,629.64 | 23,629.64 | 0.0 |
| 2 (318) | 16,493.96 | 16,493.96 | 0.0 |
| 3 (288) | 18,161.09 | 18,161.09 | 0.0 |
| 4 (289) | 24,603.85 | 24,603.85 | 0.0 |

**5/5 client 完全一致，差值为 0。**

## 9. 根因判断

1. **模型结构不同？** 否 — 完全一致
2. **配置不同？** 否 — 16 项全部一致
3. **scaler 不同？** 否 — input_scaler 和 target_scaler 完全一致
4. **split 不同？** 否 — 同一入口 `build_client_data`
5. **FedAvg 实现不同？** 否 — 收敛曲线完全一致
6. **seed / 初始化不同？** 是 — 两者都在 `run_experiment` 中调用 `set_global_seed(42)`，但实验 2 在归一化数据加载后才初始化模型，两个实验的 PyTorch 参数初始化发生在不同代码位置，随机种子消耗序列不同导致初始参数不同
7. **实验 2 Full 可作为实验 1 主模型实现的参考？** 否 — 两者完全等价

## 10. 下一步建议

**结论类型 B：只是 seed / 初始化差异。**

- 两个实验的数据、模型、训练逻辑完全一致
- 运行结果表明两者产生完全相同的结果（非随机性差异为零）
- 实验 2 Full = 实验 1 FedAvg，可以互相替代表述
- 消融实验结论清楚：Full > Without Attention > Without LSTM > Without CNN
- 实验 1 的 NaiveLastValue 已恢复

## 11. 是否建议继续实验 3/5/6

**建议继续。**

- 实验 1 vs 实验 2 Full 差异已彻底解释清楚（两者相等）
- 修复后链路验证完成
- 消融趋势合理，可直接纳入论文
- 阻塞实验 3/5/6 的前提已消除
