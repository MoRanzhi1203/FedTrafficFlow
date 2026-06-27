# 实验 1：CUDA 环境调用修复与复核报告

## 1. 问题背景

- 此前实验 1 v3 的命令显式请求了 `--device cuda`
- 但运行日志显示实际 `device=cpu`
- 因此需要先修复实验 1 的 CUDA 环境调用方式，再为后续正式实验 1 v4 CUDA 做准备

## 2. 根因

此前问题不是实验代码逻辑错误，而是运行环境调用错误：

- 错误运行 Python：`E:\anaconda3\python.exe`
- 错误 torch：`2.12.0+cpu`
- 错误 torch 来源：`C:\Users\MSIPC\AppData\Roaming\Python\Python312\site-packages`
- 因此此前实际加载了用户目录中的 CPU-only torch，而不是 `FedTrafficFlow` 环境中的 CUDA torch

本次修复方式：

- 强制使用 `E:\anaconda3\envs\FedTrafficFlow\python.exe`
- 在 PowerShell 会话中设置 `PYTHONNOUSERSITE=1`
- 从而避免 user site 污染覆盖正确的 CUDA 环境

## 3. 正确环境

本次实际复核结果：

- python 路径：`E:\anaconda3\envs\FedTrafficFlow\python.exe`
- torch 版本：`2.8.0+cu126`
- torch.__file__：`E:\anaconda3\envs\FedTrafficFlow\lib\site-packages\torch\__init__.py`
- torch.version.cuda：`12.6`
- torch.cuda.is_available()：`True`
- GPU 名称：`NVIDIA GeForce RTX 3060 Laptop GPU`
- PYTHONNOUSERSITE：`1`
- ENABLE_USER_SITE：`False`

结论：

- 当前 CUDA 版 torch 已来自 `FedTrafficFlow` 环境本身
- user site 污染已被有效规避

## 4. nvidia-smi 检查

诊断结果：

- GPU：`NVIDIA GeForce RTX 3060 Laptop GPU`
- Driver Version：`560.70`
- CUDA Version：`12.6`
- smoke 后 `nvidia-smi` 可正常返回，说明系统 GPU/驱动环境正常
- 当前 `nvidia-smi` 进程列表中未看到持续驻留的训练 python 进程，因为 smoke 已经完成退出

## 5. CUDA smoke 结果

- 是否运行 smoke：是
- smoke 命令：

```powershell
$env:PYTHONNOUSERSITE = "1"
$PY = "E:\anaconda3\envs\FedTrafficFlow\python.exe"

& $PY -m real_data_experiments.single_intersection_client.sic_core `
  --workflow all `
  --data-mode tensor `
  --device cuda `
  --num-clients 2 `
  --rounds 1 `
  --local-epochs 1 `
  --batch-size 32 `
  --sequence-length 12 `
  --learning-rate 0.001 `
  --selected-clients 290,284 `
  --seed 42 `
  --show-progress `
  --output-dir results\real_data_experiments\formal\experiment1_cuda_env_smoke
```

- 是否显示 `device=cuda`：是，日志明确输出 `[INFO] Experiment started with device=cuda`
- 核心结果文件是否生成：是
  - `run_config.json`
  - `split_summary.json`
  - `main_metrics.csv`
  - `main_summary.csv`
  - `client_metrics.csv`
  - `convergence_history.csv`
  - `prediction_samples.csv`
- 指标是否无 NaN/Inf：是
- `y_pred` 是否非常数：是
  - `FedAvg unique y_pred count = 100`
  - `Independent unique y_pred count = 100`

smoke 指标摘要：

| method | MSE | RMSE | MAE | MAPE | SMAPE | R2 |
|:--|--:|--:|--:|--:|--:|--:|
| FedAvg | 3.477995e+09 | 58629.883251 | 46876.849403 | 2.467706 | 2.462705 | 0.622412 |
| Independent | 2.996714e+09 | 54708.600985 | 41552.771402 | 2.197269 | 2.184475 | 0.665684 |

## 6. 不改变的内容

- 未修改实验代码
- 未修改 FedAvg
- 未修改模型结构
- 未修改数据划分
- 未运行实验 2/3/4
- 未修改 LaTeX
- 未修改 `simulation_experiments`
- 未修改 conda 环境
- 未提交 `results`

## 7. 下一步建议

- 当前 CUDA 环境调用方式已经复核通过
- 建议下一步使用同一个 `$PY` 和 `$env:PYTHONNOUSERSITE=1` 重跑正式实验 1 v4 CUDA
- 正式运行前保持以下调用方式固定：

```powershell
$env:PYTHONNOUSERSITE = "1"
$PY = "E:\anaconda3\envs\FedTrafficFlow\python.exe"
```

- 然后统一使用：

```powershell
& $PY -m real_data_experiments.single_intersection_client.sic_core ...
```

- 这样可以避免再次误用 `E:\anaconda3\python.exe` 或用户目录中的 CPU-only torch
