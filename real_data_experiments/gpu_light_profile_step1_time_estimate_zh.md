# 轻量 GPU Profiling Step 1 耗时估算报告

## 1. 本阶段范围

本阶段只做耗时估算，不运行 profiling、不运行训练、不修改代码、不修改训练代码、不生成 results。

本次估算仅用于工程排程与执行前预判，不作为正式 profiling 结果，也不作为论文正式实验结果。

## 2. 参考基线

参考基线来自既有 `gpu_cuda_smoke` 的 `results/real_data_experiments/gpu_cuda_smoke/profiling_summary.json`：

- setting：`grid_cell`
- task：`main`
- device：`cuda`
- num_clients：`3`
- rounds：`1`
- local_epochs：`1`
- batch_size：`16`
- sequence_length：`12`
- wall_time_sec：`18.55321`
- status：`ok`
- exit_status：`success`

说明：

- 该基线已经在当前 CUDA 环境下真实跑通。
- 当前参考基线与 Step 1 计划参数相比，仅 `rounds` 从 `1` 增加到 `3`。

## 3. Step 1 计划参数

- setting = `grid_cell`
- task = `main`
- device = `cuda`
- num_clients = `3`
- rounds = `3`
- local_epochs = `1`
- batch_size = `16`
- sequence_length = `12`

## 4. 估算方法

采用基于 rounds 的线性估算：

`estimated_time = smoke_wall_time_sec * 3`

其中：

- `smoke_wall_time_sec = 18.55321`
- `3` 来自 Step 1 的 `rounds = 3`

保守范围按如下方式估算：

`estimated_safe_range = estimated_time ~ estimated_time * 1.5`

计算结果：

- `estimated_linear_time = 18.55321 * 3 = 55.65963 sec`
- `estimated_safe_range_low = 55.65963 sec`
- `estimated_safe_range_high = 83.489445 sec`

## 5. 估算结论

- 线性估算约 `55.66` 秒。
- 保守估算范围约 `55.66 - 83.49` 秒。
- 便于人工排程时，可按 `56 - 84` 秒理解。
- 建议人工预留 `1 - 2` 分钟。
- 该估算不是正式 profiling 结果。

## 6. 风险与波动来源

可能影响实际耗时的因素包括：

- GPU 后台占用。
- 数据加载与文件 I/O。
- CUDA 初始化。
- Windows 进程调度。
- 首次运行缓存状态。
- 显存碎片或残留进程。

补充说明：

- 当前 `nvidia-smi` 显示系统仍存在浏览器、QQ、微信、Trae 等图形界面相关进程，这会导致轻微波动。
- 即使 Step 1 与 smoke 的模型配置主参数接近，首次进入具体任务目录、日志写入和系统调度也可能带来额外秒级开销。

## 7. 是否建议执行 Step 1

- 可以执行 Step 1，建议预留 `1 - 2` 分钟。

当前前提：

- 当前仓库工作区干净。
- 当前 `main` 与 `origin/main` 同步。
- 当前 CUDA 仍可用，`torch=2.8.0+cu126`、`torch.version.cuda=12.6`、`cuda_available=True`。
- 当前 GPU 设备名称仍为 `NVIDIA GeForce RTX 3060 Laptop GPU`。

结论边界：

- 该估算仅用于工程排程，不是正式 profiling 结果。
- 实际耗时可能受 GPU 状态、I/O、数据加载、首次 CUDA kernel 初始化、后台进程影响。
