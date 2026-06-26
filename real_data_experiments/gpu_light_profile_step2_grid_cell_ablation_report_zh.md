# 轻量 GPU Profiling Step 2 报告：grid_cell ablation r2e1

## 1. 本阶段范围

本阶段只运行 Step 2：`grid_cell ablation r2e1`。

- 未运行 Step 3。
- 未运行 Step 4。
- 未运行 cluster。
- 未运行正式论文实验。
- 未修改训练代码。
- 未修改 FedAvg 聚合逻辑。
- 未修改 LaTeX。
- 未修改 `simulation_experiments`。
- 未修改 conda 环境。

## 2. 环境状态

- Conda 环境路径：`E:\anaconda3\envs\FedTrafficFlow`
- Python 解释器：`E:\anaconda3\envs\FedTrafficFlow\python.exe`
- torch 版本：`2.8.0+cu126`
- `torch.version.cuda`：`12.6`
- `cuda_available`：`True`
- GPU device name：`NVIDIA GeForce RTX 3060 Laptop GPU`
- `PYTHONNOUSERSITE`：`1`
- `ENABLE_USER_SITE`：`False`

## 3. 执行命令

实际执行命令如下：

```powershell
python real_data_experiments/profile_tensor_experiments.py --setting grid_cell --task ablation --device cuda --num-clients 3 --rounds 2 --local-epochs 1 --batch-size 16 --sequence-length 12 --output-dir results/real_data_experiments/gpu_light_profile/grid_cell_ablation_r2e1
```

## 4. 输出目录

输出目录为：

`results/real_data_experiments/gpu_light_profile/grid_cell_ablation_r2e1`

该目录受 `.gitignore` 忽略，不提交到仓库。

主要生成文件包括：

- `profiling_summary.json`
- `compute_time_estimation_summary.csv`
- `dataset_scale_summary.json`
- `hardware_summary.json`
- `profile_grid_cell_ablation_cuda.csv`
- `profile_gpu_grid_cell_ablation_r2e1/environment_summary.json`
- `profile_gpu_grid_cell_ablation_r2e1/experiment_notes_zh.md`
- `profile_gpu_grid_cell_ablation_r2e1/ablation_summary.csv`
- `profile_gpu_grid_cell_ablation_r2e1/ablation_metrics.csv`
- `profile_gpu_grid_cell_ablation_r2e1/ablation_client_metrics.csv`
- `profile_gpu_grid_cell_ablation_r2e1/run_config.json`

## 5. Profiling 结果摘要

以下结果来自 `profiling_summary.json`：

- `status = ok`
- `exit_status = success`
- `wall_time_sec = 49.222596`
- `setting = grid_cell`
- `task = ablation`
- `device = cuda`
- `num_clients = 3`
- `rounds = 2`
- `local_epochs = 1`
- `batch_size = 16`
- `sequence_length = 12`
- `output_dir = results\real_data_experiments\gpu_light_profile\grid_cell_ablation_r2e1\profile_gpu_grid_cell_ablation_r2e1`

补充工程信息：

- `gpu_total_memory_MB = 6144.0`
- `gpu_max_allocated_MB = 25.565`
- `gpu_max_reserved_MB = 42.0`
- `total_train_samples = 12261`
- `round_count = 8`
- `avg_time_per_round_sec = 6.152825`

## 6. Ablation 结构摘要

- `variant_count = 4`
- `variants = full, without_attention, without_cnn, without_lstm`

当前输出目录中已生成以下消融摘要文件：

- `ablation_summary.csv`
- `ablation_metrics.csv`
- `ablation_client_metrics.csv`

如果这些文件中包含各 variant 指标，本报告只将其视为工程记录，不作为论文正式结果。

## 7. GPU 状态

任务前后 `nvidia-smi` 观察如下：

- 运行前显存占用约为 `1417 MiB / 6144 MiB`
- `profiling_summary.json` 记录的任务启动前显存约为 `1474 MiB / 6144 MiB`
- 运行后外部检查显存约为 `1409 MiB / 6144 MiB`
- `profiling_summary.json` 记录的任务结束后显存约为 `1618 MiB / 6144 MiB`
- 未发生 CUDA OOM
- 未发现异常残留计算进程，主要仍为桌面图形界面相关进程

## 8. 与估算对比

参考 Step 2 耗时估算报告：

- 估算 wall time 约 `53.17` 秒
- 保守范围约 `53 - 80` 秒
- 建议人工预留 `2 - 3` 分钟

本次实际结果：

- 实际 `wall_time_sec = 49.222596` 秒
- 实际耗时略低于线性估算值
- 实际耗时也低于保守范围下界，但仍与估算处于同一数量级
- 说明当前 Step 2 在现有 GPU 环境下运行稳定，且比保守排程略快

## 9. 结果使用边界

- 本次结果仅用于工程耗时、显存和运行链路评估
- 不作为论文正式结果
- 不用于论文正文表格
- 不用于审稿回复正式指标
- 不改变标准 FedAvg 主线

## 10. 是否可以进入 Step 3

- 可以进入 Step 3：`cluster main r2e1 cap1024`

前提说明：

- 继续保持当前 `.gitignore` 规则，确保 `results/real_data_experiments/gpu_light_profile/` 不进入 Git 状态
- 继续维持轻量 profiling 边界，不自动扩大规模
- Step 3 开始前仍应重新做 Git 与 CUDA 前置核验
