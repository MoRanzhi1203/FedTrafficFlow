# 实验 1：CUDA fallback 排查与修复报告

## 1. 问题背景

- 实验 1 v3 的运行命令显式传入了 `--device cuda`
- 但运行日志显示 `[INFO] Experiment started with device=cpu`
- 因此正式 v3 实际发生了 CPU fallback，不能直接作为目标中的 `full_cuda` 正式结果

## 2. 环境检查

- python 路径：`E:\anaconda3\python.exe`
- torch 版本：`2.12.0+cpu`
- `torch.version.cuda`：`None`
- `torch.cuda.is_available()`：`False`
- `torch.cuda.device_count()`：`0`
- GPU 名称：`None`

`pip show torch` 结果：

- torch 安装位置：`C:\Users\MSIPC\AppData\Roaming\Python\Python312\site-packages`
- 该位置是用户站点包目录，不是项目期望的 CUDA conda 环境隔离路径

`nvidia-smi` 摘要：

- Driver Version：`560.70`
- CUDA Version：`12.6`
- GPU：`NVIDIA GeForce RTX 3060 Laptop GPU`
- 系统 GPU 正常可见，但当前 Python 进程没有加载 CUDA 版 PyTorch

## 3. 代码问题定位

本次 CPU fallback 的直接原因不是实验 1 代码把可用 CUDA 错误回退到 CPU，而是当前 Python 环境本身不是 CUDA 环境：

- 当前导入到的 `torch` 是 `2.12.0+cpu`
- `torch.version.cuda=None`
- `torch.cuda.is_available()=False`
- 因此无论实验 1 代码是否请求 `--device cuda`，当前环境都不具备实际走 CUDA 的条件

额外代码审计结论：

- `sic_config.py` 中 `args.device` 会正确写入 `config.device`
- `sic_core.py` 中当前设备解析通过 `resolve_default_device(config.device)` 执行
- 当前链路不会把“可用 CUDA”静默覆盖为 CPU；本次日志中的 `device=cpu` 与环境检测结果一致

## 4. 实际修改

- 本阶段未修改 `sic_core.py`
- 本阶段未修改 `sic_config.py`
- 原因：按照约束，当 `torch.cuda.is_available()=False` 时，不应继续修改实验代码来掩盖环境问题

## 5. CUDA smoke 结果

- 本阶段未运行 CUDA smoke
- 原因：当前环境中 `torch.cuda.is_available()=False`
- 在这种前提下继续跑 `--device cuda` smoke 没有意义，也不满足“resolved_device=cuda”的验证目标

## 6. 不改变的内容

- 未修改 FedAvg
- 未修改模型结构
- 未修改数据划分
- 未修改实验 2/3/4
- 未修改 LaTeX
- 未修改 `simulation_experiments`
- 未提交 `results/`

## 7. 下一步建议

- 暂不重跑正式实验 1 v4 CUDA
- 先修复运行环境，确保当前 Python 实际加载 CUDA 版 PyTorch，而不是用户站点中的 CPU-only torch
- 优先检查：
  - 是否启用了错误的用户站点包覆盖
  - 是否需要设置 `PYTHONNOUSERSITE=1`
  - 是否需要切换到项目期望的 conda 环境并重新验证 `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"`
- 当且仅当：
  - `torch.cuda.is_available()=True`
  - `torch.version.cuda` 非空
  - `torch.cuda.get_device_name(0)` 可正常返回

  再继续做实验 1 的 CUDA smoke，并准备重跑正式实验 1 v4 CUDA

