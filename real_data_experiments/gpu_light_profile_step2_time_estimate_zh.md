# 轻量 GPU Profiling Step 2 耗时估算报告：grid_cell ablation r2e1

## 1. 本阶段范围

本阶段只做 Step 2 耗时估算，不执行 Step 2 profiling。

- 未运行 Step 2 profiling。
- 未运行 ablation 实际任务。
- 未运行训练。
- 未运行 cluster。
- 未运行正式论文实验。
- 未修改训练代码。
- 未修改 FedAvg 聚合逻辑。
- 未修改 LaTeX。
- 未修改 `simulation_experiments`。
- 未修改 conda 环境。
- 未生成新的 results。

## 2. 参考基线

参考基线来自已经完成的 Step 1 `grid_cell main r3e1`：

- setting：`grid_cell`
- task：`main`
- device：`cuda`
- num_clients：`3`
- rounds：`3`
- local_epochs：`1`
- batch_size：`16`
- sequence_length：`12`
- wall_time_sec：`19.939549`
- status：`ok`
- exit_status：`success`

说明：

- 当前 Step 1 基线来自既有 `profiling_summary.json`，未重新运行。
- 当前仓库工作区干净，`main` 与 `origin/main` 同步。
- 当前 CUDA 仍可用，`torch=2.8.0+cu126`、`torch.version.cuda=12.6`、`cuda_available=True`。

## 3. Step 2 计划参数

- setting = `grid_cell`
- task = `ablation`
- device = `cuda`
- num_clients = `3`
- rounds = `2`
- local_epochs = `1`
- batch_size = `16`
- sequence_length = `12`
- output_dir = `results/real_data_experiments/gpu_light_profile/grid_cell_ablation_r2e1`

## 4. ablation 结构确认

根据静态阅读 `profile_tensor_experiments.py` 与 `single_intersection_ablation/sia_config.py`：

- `--task ablation` 已确认受支持。
- `--setting grid_cell` 已确认受支持。
- Step 2 命令中的 `--device`、`--num-clients`、`--rounds`、`--local-epochs`、`--batch-size`、`--sequence-length`、`--output-dir` 已确认存在于脚本 `--help`。
- `grid_cell + ablation` 会走 `real_data_experiments.single_intersection_ablation.sia_core`。
- ablation 变体来自 `SIA_DEFAULT_VARIANTS`。
- 已静态确认 `variant_count = 4`。

当前默认 variants 为：

- `full`
- `without_attention`
- `without_cnn`
- `without_lstm`

## 5. 估算方法

估算公式如下：

`main_time_per_round = step1_wall_time_sec / 3`

`per_variant_time = main_time_per_round * 2`

`estimated_linear_time = per_variant_time * variant_count`

`estimated_safe_high = estimated_linear_time * 1.5`

代入本次已知值：

- `step1_wall_time_sec = 19.939549`
- `variant_count = 4`

计算结果：

- `main_time_per_round = 19.939549 / 3 = 6.646516 sec`
- `per_variant_time = 6.646516 * 2 = 13.293032 sec`
- `estimated_linear_time = 13.293032 * 4 = 53.172128 sec`
- `estimated_safe_high = 53.172128 * 1.5 = 79.758192 sec`

## 6. 估算结论

- 线性估算约 `53.17` 秒。
- 保守估算范围约 `53 - 80` 秒。
- 建议人工预留 `2 - 3` 分钟。
- 该估算只用于工程排程。
- 该估算不是正式 profiling 结果。
- 该估算不是论文正式实验结果。

## 7. 风险与波动来源

可能影响实际耗时的因素包括：

- ablation variant 数量与每个 variant 的内部执行路径。
- GPU 后台占用。
- 数据加载与文件 I/O。
- CUDA 初始化。
- Windows 进程调度。
- 首次运行缓存状态。
- 显存碎片或残留进程。

补充说明：

- 当前 `nvidia-smi` 显示仍有桌面图形相关后台进程，这会带来一定秒级波动。
- Step 2 属于 ablation，多 variant 串行执行时，实际耗时可能比单一 `main` 任务更受日志写入与任务切换影响。

## 8. 是否建议执行 Step 2

- 可以执行 Step 2，建议预留 `2 - 3` 分钟。

执行边界：

- 当前估算仅用于工程排程。
- 不应将 Step 2 结果视为论文正式实验结果。
- 执行时仍应保持轻量 profiling 边界，不自动扩大规模，不切换到 cluster，不进入正式论文实验。
