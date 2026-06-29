# 实验 5/6 训练失效诊断报告

> 生成日期：2026-06-29
> 本轮不运行实验，仅基于代码静态分析和已完成结果进行诊断。

## 1. 现象

### 1.1 实验 5 formal r20e1 指标

**spatial_block (exp5_rc_spatial_block_k3_r20_e1_cuda)**:

| Method | RMSE | MAE | MAPE(%) | R² |
|--------|------|-----|---------|-----|
| FedAvg | 627,741 | 457,078 | 99.970 | **-1.128** |
| Independent | 627,704 | 457,020 | 97.668 | **-1.129** |

**flow_kmeans (exp5_rc_flow_kmeans_k3_r20_e1_cuda)**:

| Method | RMSE | MAE | MAPE(%) | R² |
|--------|------|-----|---------|-----|
| FedAvg | 569,025 | 503,631 | 100.553 | **-4.110** |
| Independent | 568,878 | 503,426 | 98.759 | **-4.108** |

### 1.2 实验 6 formal r20e1 指标

**spatial_block full (exp6_ra_spatial_block_k3_full_r20_e1_cuda)**:

| Variant | RMSE | MAE | MAPE(%) | R² |
|---------|------|-----|---------|-----|
| Full | 627,741 | 457,078 | 99.970 | **-1.128** |

### 1.3 r1 与 r20 对比

| 实验配置 | r1e1 RMSE | r20e1 RMSE | 变化 |
|----------|:---------:|:----------:|:----:|
| exp5 spatial_block FedAvg | 629,575 | 627,741 | **几乎相同** |
| exp5 spatial_block Independent | 629,538 | 627,704 | **几乎相同** |
| exp5 flow_kmeans FedAvg | 570,644 | 569,025 | **几乎相同** |
| exp6 full | 629,575 | 627,741 | **完全相同** |

**r20 相比 r1 没有任何有意义的改善**，模型未发生学习。

### 1.4 是否差于 NaiveLastValue

实验 5/6 **未输出 NaiveLastValue baseline**（`rc_core.py` 和 `ra_core.py` 没有 `evaluate_naive_last_value` 函数），因此无法直接对比。但 RMSE ≈ 600k 的量级与原始流量值（~1.8e6）处于同一数量级，说明模型预测近似常数输出（接近全局均值），而非有效的时间序列预测。

---

## 2. 代码链路检查

### 2.1 Dataset

**文件**: `real_data_experiments/common/region_tensor_dataset.py`

- `RegionClientWindowDataset.__init__` (line 30): `self.tensor = tensor.detach()` — ✅ 已修复，无 `.clone()`
- `__getitem__` (line 66-75): Lazy slicing，返回 `features.to(dtype=torch.float32)` 和 `torch.tensor([float(target.item())], dtype=torch.float32)` — ✅ 无多余 `.to(dtype=...)`
- **返回的 targets 和 features 均为 raw scale**（例如 channel 0 流量值约 ~1.8e6）

### 2.2 Scaler — ❌ 完全缺失！

**文件**: `real_data_experiments/region_client/rc_core.py`

关键证据：`rc_core.py` 的 `run_experiment()` 函数（line 436-492）：

```python
def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    # ... device setup, seed ...
    clients, split_summary, partition_result = build_region_client_data(config)
    fed_client_df, convergence_df, fed_prediction_df = run_fedavg_experiment(config, clients, device)
    ind_client_df, ind_prediction_df = run_independent_experiment(config, clients, device)
    # ...
    export_results(...)
```

**缺失的代码**（对比 `sic_core.py` 正常流程）：

```python
# 以下代码在 rc_core.py 中完全不存在：
input_scaler = fit_input_scaler(clients) if config.input_normalization else None
target_scaler = fit_target_scaler(clients) if config.target_normalization else None
apply_dataset_normalization(clients, input_scaler=input_scaler, target_scaler=target_scaler)
```

**文件**: `real_data_experiments/region_ablation/ra_core.py` — 同样完全缺失 scaler。

**文件**: `real_data_experiments/region_client/rc_config.py` — `ExperimentConfig` **没有** `input_normalization` 和 `target_normalization` 字段。整个配置类中不存在这两个 boolean 选项。

### 2.3 Target

- `RegionClientWindowDataset.__getitem__` 返回 raw-scale targets
- `rc_core.py` 训练时直接使用 raw-scale targets 计算 `MSELoss`
- 由于 target 值在 ~1.8e6 量级，loss 初始值非常大
- **无 `NormalizedDataset` 包装，无 target scaler 反归一化**

### 2.4 Training loop

**`rc_core.py` 使用两种训练方式**:

1. **FedAvg** (`run_fedavg_experiment`, line 289-331): 使用 `FedClient` + `run_federated_rounds`（通过 `common/trainer.py`）。每个 client 内部通过 `FedClient.train()` 执行 `local_epochs` × `DataLoader` 的训练。

2. **Independent** (`run_independent_experiment`, line 355-381): 内部 `_train_local_model()` 逐 batch 执行 `optimizer.zero_grad()` → `loss.backward()` → `optimizer.step()` — ✅ 训练循环本身正确。

**问题**：所有输入均为 raw scale (~1.8e6)，Adam 优化器默认 lr=1e-3 在如此大的梯度下可能导致参数爆炸或无法收敛。但更可能是模型预测始终接近零或接近常数，因为未经归一化的输入特征和 targets 使网络难以学习。

### 2.5 FedAvg aggregation

**文件**: `real_data_experiments/common/trainer.py`

```python
local_results = [client.train(global_state) for client in clients]
aggregated_state = fedavg_aggregate(
    [result.state_dict for result in local_results],
    [result.sample_count for result in local_results],
)
global_model.load_state_dict(aggregated_state)
```

FedAvg aggregation loop 本身正确 — 每个 client 从 global_state 初始化，训练后上传 state_dict，server 按 sample_count 加权聚合。

### 2.6 Evaluation — ❌ 无 target_scaler 传递

**`rc_core.py` 的 `evaluate_round()`** (line 248-269):

```python
def evaluate_round(global_model, clients, device):
    for client in clients:
        val_true, val_pred = collect_predictions(global_model, client.val_loader, device)  # ← target_scaler=None
        test_true, test_pred = collect_predictions(global_model, client.test_loader, device)  # ← target_scaler=None
```

`collect_predictions` (from `sic_core.py`) 接受可选的 `target_scaler` 参数：

```python
def collect_predictions(model, loader, device, target_scaler=None):
    # ...
    if target_scaler is not None:
        outputs = target_scaler.denormalize_tensor(outputs)
```

当 `target_scaler=None` 时，predictions 不经过反归一化，直接与 raw-scale targets 计算 metrics。

**但这对于 raw-scale 训练来说是正确的**——因为没有做归一化训练，也就不需要反归一化。metrics 是在正确的 raw scale 上计算的。

### 2.7 Metrics

`compute_regression_metrics` 本身实现正确，直接计算 raw-scale 上的 MSE/RMSE/MAE。

---

## 3. 根因判断（确定性结论）

### 根本原因：**scaler 链路完全缺失**

`rc_core.py` 和 `ra_core.py` 的实验主流程中：

1. **没有 `fit_input_scaler()`** — 输入特征未被 z-score 归一化
2. **没有 `fit_target_scaler()`** — 目标值未被归一化
3. **没有 `apply_dataset_normalization()`** — Dataset 未被 `NormalizedDataset` 包装
4. **没有 `input_normalization` / `target_normalization` 配置字段** — `rc_config.py` 的 `ExperimentConfig` 中没有这两个选项

**对比实验 1 (`sic_core.py`)**:
- `sic_config.py` 有 `input_normalization: bool = True` 和 `target_normalization: bool = True`
- `sic_core.py` 中有完整的 `fit_input_scaler()` → `fit_target_scaler()` → `apply_dataset_normalization()` 流程
- 训练在归一化空间进行，评估时反归一化回原始尺度

### 为什么 R² 是负的？

模型在 raw scale (~1.8e6) 上训练。由于输入/目标尺度极大，模型无法有效学习，预测值接近常数（大概率接近零或初始化均值）。R² 为负表示模型预测比简单预测均值更差——这正是在 raw scale 上训练失败的典型表现。

### 为什么 r20 与 r1 几乎相同？

因为模型从一开始就无法有效学习（梯度尺度问题），增加训练轮数不会改善。

### 证据强度排序

1. **Scaler 缺失** — 最高风险，确定无疑（代码中完全不存在相关调用）
2. **r20 vs r1 无变化** — 支持训练未生效的判断
3. **R² 全负** — 支持模型失效的判断
4. **无 NaiveLastValue baseline** — 缺少比较基准

---

## 4. 后续最小修复建议

### P0：为 rc_core.py 和 ra_core.py 接入归一化链路

需要补充的代码逻辑（参考 sic_core.py 或 rfc_core.py 的实现）：

1. 在 `rc_config.py` 和 `ra_config.py` 中添加 `input_normalization` / `target_normalization` 配置字段
2. 在 `rc_core.py` 和 `ra_core.py` 的 `run_experiment()` 中：
   - 调用 `fit_input_scaler()` 拟合 per-channel 输入均值/标准差
   - 调用 `fit_target_scaler()` 拟合目标均值/标准差
   - 调用 `apply_dataset_normalization()` 包装 Dataset
   - 将 `target_scaler` 传入所有 evaluation 函数
3. `rc_core.py` 中补充 `evaluate_naive_last_value()` 函数

### P1：修复后先跑 r1e1 或 r3e1 验证

```text
先对 Exp5 spatial_block (k=3, r3e1) 跑一轮诊断修复验证；
确认 train_loss 下降 + RMSE 改善后，再决定是否重跑 r20 formal。
不直接重跑所有 formal。
```

### 注意事项

`RegionClientWindowDataset` 返回的 targets 是 raw scale，而 `fit_input_scaler` 和 `fit_target_scaler` 需要适配多 region 场景。实验 3 的 `rfc_core.py` 已经实现了适配版本（`fit_rfc_input_scaler` / `fit_rfc_target_scaler`），可以作为参考。

---

## 5. 实验 3 是否受影响

实验 3 的 `rfc_core.py` **已正确接入 scaler 链路** (line 426-446)：

```python
input_scaler = fit_rfc_input_scaler(clients, ...) if config.input_normalization else None
target_scaler = fit_rfc_target_scaler(clients, ...) if config.target_normalization else None
apply_dataset_normalization(clients, input_scaler=input_scaler, target_scaler=target_scaler)
```

因此实验 3 **不受此问题影响**。exp3 smoke 的指标是在正确的归一化 + 反归一化链路下产生的。

---

## 6. 与实验 2 修复历史的关联

实验 2 的首次 formal（commit `9323369`）也出现过完全相同的 scaler 缺失问题，导致 `MAPE=99.99%`、`R²=-700`、`y_pred ≈ 66`（常数预测）。

修复 commit `d2b87f4` 在 `sia_core.py` 中补全了 scaler 链路，修复后 exp2 Full RMSE=17,346（正常）。

**实验 5/6 的当前问题与实验 2 修复前的问题完全同源**，修复方案也一致。
