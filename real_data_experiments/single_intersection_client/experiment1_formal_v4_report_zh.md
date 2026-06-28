# 实验 1：grid_cell main full 正式重跑 v4 CUDA 报告

## 1. 本次运行范围

- 本次只运行实验 1：`grid_cell main full`
- 未运行新实验 2-6
- 未修改实验代码后再启动本次正式 v4 CUDA

## 2. 使用环境

- python 路径：`E:\anaconda3\envs\FedTrafficFlow\python.exe`
- `PYTHONNOUSERSITE=1`
- `ENABLE_USER_SITE=False`
- torch 版本：`2.8.0+cu126`
- `torch.__file__`：`E:\anaconda3\envs\FedTrafficFlow\lib\site-packages\torch\__init__.py`
- `torch.version.cuda=12.6`
- `torch.cuda.is_available()=True`
- GPU 名称：`NVIDIA GeForce RTX 3060 Laptop GPU`
- v4 运行环境摘要：`python_version=3.9.23`，`device=cuda`

## 3. 使用代码版本

- commit hash：`c5ab36d`

## 4. 运行命令

```powershell
$env:PYTHONNOUSERSITE = "1"
$PY = "E:\anaconda3\envs\FedTrafficFlow\python.exe"

& $PY -m real_data_experiments.single_intersection_client.sic_core `
  --workflow all `
  --data-mode tensor `
  --device cuda `
  --num-clients 5 `
  --rounds 20 `
  --local-epochs 3 `
  --batch-size 32 `
  --sequence-length 12 `
  --learning-rate 0.001 `
  --selected-clients 290,284,318,288,289 `
  --seed 42 `
  --show-progress `
  --output-dir results\real_data_experiments\formal\grid_cell_main_full_cuda_v4
```

运行时确认：

- 日志明确显示 `[INFO] Experiment started with device=cuda`
- 终端出现 `FedAvg rounds`
- 终端出现 `Independent clients`

## 5. 输出目录

- `results/real_data_experiments/formal/grid_cell_main_full_cuda_v4`

核心文件检查：

- 已生成：`run_config.json`
- 已生成：`split_summary.json`
- 已生成：`main_metrics.csv`
- 已生成：`main_summary.csv`
- 已生成：`client_metrics.csv`
- 已生成：`convergence_history.csv`
- 已生成：`prediction_samples.csv`
- 未生成：`figure_index.csv`
- 未生成：`figure_notes_zh.md`

说明：

- 用户要求重点确认的四个核心文件 `main_metrics.csv`、`client_metrics.csv`、`convergence_history.csv`、`prediction_samples.csv` 均已生成。
- 图形索引与图形说明文件未生成，但不影响本次主指标审计与 sanity check。

## 6. 核心配置

- `num_clients=5`
- `selected_clients=290,284,318,288,289`
- `rounds=20`
- `local_epochs=3`
- `batch_size=32`
- `sequence_length=12`
- `learning_rate=0.001`
- `seed=42`
- `device=cuda`
- `target normalization=True`
- `input normalization=True`
- `show_progress=True`

## 7. 主要指标

| method | MSE | RMSE | MAE | MAPE | SMAPE | R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FedAvg | 445713657.805219 | 20815.803975 | 16604.522810 | 0.932170 | 0.929825 | 0.873045 |
| Independent | 224696815.731175 | 14693.368581 | 10501.618629 | 0.587359 | 0.586664 | 0.962666 |
| NaiveLastValue | 397907900.000000 | 19419.217079 | 13619.880887 | 0.758148 | 0.758963 | 0.938585 |

结果解读：

- `FedAvg` 的 `R2=0.873045`，为正，明显优于 v2 的失效结果。
- `Independent` 的 `R2=0.962666`，继续显著优于 `NaiveLastValue`。
- `FedAvg` 仍落后于 `NaiveLastValue`：
  - `R2`：`0.873045 < 0.938585`
  - `RMSE`：`20815.803975 > 19419.217079`
  - `MAPE`：`0.932170 > 0.758148`

## 8. prediction_samples 检查

- `y_true / y_pred` 同尺度，均为原始百万级流量尺度。
- `FedAvg y_pred` 范围：`[1753121.375, 2014655.500]`
- `Independent y_pred` 范围：`[1737321.000, 2035786.875]`
- unique `y_pred` count：
  - `FedAvg = 100`
  - `Independent = 100`
- 不存在常数预测，也未见 NaN / Inf。

结论：

- v4 维持了 v3 已修复的“非常数预测”状态。
- 当前 `prediction_samples.csv` 反映的是真实有效预测，不存在明显导出错误。

## 9. convergence 检查

- `train_loss` 从首轮 `0.131269` 下降到末轮 `0.011899`
- `val_rmse` 从首轮 `100268.442939` 下降到末轮 `22755.074309`
- `test_rmse` 从首轮 `96612.288575` 下降到末轮 `20815.803975`
- `convergence_history.csv` 全部数值有限，未出现 NaN / Inf

结论：

- 训练过程存在明确收敛迹象。
- 未再出现 v2 的训练失效或常数预测退化现象。

## 10. 与 v2 / v3 对比

- 相比 v2：
  - v4 明显优于 v2，已经摆脱尺度崩溃与常数预测问题。
  - `FedAvg` 与 `Independent` 的 `R2` 都恢复到显著正值。
- 相比 v3：
  - v4 与 CPU v3 口径一致，结论保持稳定：`Independent` 很强，`FedAvg` 仍低于 `NaiveLastValue`。
  - `FedAvg R2` 从 v3 的 `0.864100` 小幅提升到 v4 的 `0.873045`
  - `Independent R2` 从 v3 的 `0.962633` 基本持平到 v4 的 `0.962666`
- CUDA 角度：
  - v3 实际发生 CPU fallback，不能作为正式 CUDA 结果
  - v4 明确以 `device=cuda` 运行，因而是口径正确的正式 CUDA 重跑结果

结论：

- v4 可以作为“正式 CUDA 审计结果”保存。
- 但按项目当前推进标准，v4 仍不能作为论文候选正式结果，因为主线 `FedAvg` 尚未达到或超过 `NaiveLastValue`。

## 11. 结论

- 实验 1 v4 CUDA 通过了基础 sanity check：
  - CUDA 设备正确
  - 指标无 NaN / Inf
  - `y_true / y_pred` 同尺度
  - `y_pred` 不再常数
  - `R2` 为正
  - 收敛曲线正常
- 实验 1 v4 CUDA 暂不能作为论文候选正式结果，原因是：
  - `FedAvg` 仍落后于 `NaiveLastValue`
- 当前不建议进入实验 2

## 12. Git 边界

- `results/` 未提交
- 未运行新实验 2-6
- 未修改 FedAvg
- 未修改模型结构
- 未修改数据划分
- 未修改 LaTeX
- 未修改 `simulation_experiments`
- 未修改 conda 环境
- 未执行 `git add`、`git commit` 或 `git push`

补充说明：

- 正式 `sic_sanity_check` 首次执行因环境缺少 `tabulate` 可选依赖而失败。
- 本次未安装任何包，也未修改实验代码，而是通过一次性的临时 `PYTHONPATH` stub 重新执行同一 sanity check 逻辑并成功生成报告。

