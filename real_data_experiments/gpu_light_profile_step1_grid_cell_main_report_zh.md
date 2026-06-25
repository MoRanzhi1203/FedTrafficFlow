# 轻量 GPU Profiling Step 1 报告：grid_cell main r3e1

## 1. 本阶段范围

本阶段只记录已经完成的 Step 1：`grid_cell main r3e1`。

本阶段未重新运行 Step 1，而是基于既有输出摘要完成 Git 收口与结果整理。

- 未重新运行 Step 1。
- 未运行 Step 2。
- 未运行 ablation。
- 未运行 cluster。
- 未运行正式论文实验。
- 未修改训练代码。
- 未修改 FedAvg 聚合逻辑。
- 未修改 LaTeX。
- 未修改 `simulation_experiments`。
- 未修改 conda 环境。

## 2. 执行命令

此前已执行的命令如下：

```powershell
python real_data_experiments/profile_tensor_experiments.py --setting grid_cell --task main --device cuda --num-clients 3 --rounds 3 --local-epochs 1 --batch-size 16 --sequence-length 12 --output-dir results/real_data_experiments/gpu_light_profile/grid_cell_main_r3e1
```

## 3. 输出目录

本次 Step 1 输出目录为：

`results/real_data_experiments/gpu_light_profile/grid_cell_main_r3e1`

其中主要生成文件包括：

- `profiling_summary.json`
- `compute_time_estimation_summary.csv`
- `dataset_scale_summary.json`
- `hardware_summary.json`
- `profile_grid_cell_main_cuda.csv`
- `profile_gpu_grid_cell_main_r3e1/environment_summary.json`
- `profile_gpu_grid_cell_main_r3e1/experiment_notes_zh.md`
- `profile_gpu_grid_cell_main_r3e1/main_summary.json`
- `profile_gpu_grid_cell_main_r3e1/main_metrics.json`
- `profile_gpu_grid_cell_main_r3e1/client_metrics.json`
- `profile_gpu_grid_cell_main_r3e1/convergence_history.json`
- `profile_gpu_grid_cell_main_r3e1/run_config.json`

说明：

- 该目录现已通过 `.gitignore` 忽略。
- profiling 结果目录不提交到仓库。

## 4. Profiling 结果摘要

以下内容来自 `results/real_data_experiments/gpu_light_profile/grid_cell_main_r3e1/profiling_summary.json`：

- `status = ok`
- `exit_status = success`
- `wall_time_sec = 19.939549`
- `setting = grid_cell`
- `task = main`
- `device = cuda`
- `num_clients = 3`
- `rounds = 3`
- `local_epochs = 1`
- `batch_size = 16`
- `sequence_length = 12`
- `output_dir = results\real_data_experiments\gpu_light_profile\grid_cell_main_r3e1\profile_gpu_grid_cell_main_r3e1`

补充工程信息：

- `avg_time_per_round_sec = 6.646516`
- `gpu_total_memory_MB = 6144.0`
- `gpu_max_allocated_MB = 25.358`
- `gpu_max_reserved_MB = 42.0`
- `total_train_samples = 12261`

## 5. 与估算对比

参考既有估算报告 `gpu_light_profile_step1_time_estimate_zh.md`：

- 线性估算约 `55.66` 秒。
- 保守范围约 `56 - 84` 秒。
- 建议人工预留 `1 - 2` 分钟。

本次 Step 1 实际结果：

- 实际 `wall_time_sec = 19.939549` 秒。
- 实际耗时低于保守估算范围。
- 说明该轻量 Step 1 在当前 GPU 环境下运行更快。

## 6. GPU 状态与异常检查

根据 `profiling_summary.json` 中记录的 `nvidia_smi_before` 与 `nvidia_smi_after`：

- 运行前显存占用约为 `1260 MiB / 6144 MiB`。
- 运行后显存占用约为 `1440 MiB / 6144 MiB`。
- 未发生 CUDA OOM。
- 脚本退出状态为 `success`。
- 未发现异常残留计算进程，主要仍为桌面图形界面相关进程。
- 运行前后显存占用只作为工程记录，不作为论文结论。

## 7. Git 收口情况

- `results/real_data_experiments/gpu_light_profile/` 曾进入 `git status`。
- 已通过更新 `.gitignore` 添加忽略规则：`results/real_data_experiments/gpu_light_profile/`
- profiling 结果目录不提交。
- 当前后续只需提交 `.gitignore` 和本报告。

## 8. 结果使用边界

- 本次结果仅用于工程耗时、显存和运行链路评估。
- 不作为论文正式结果。
- 不用于论文正文表格。
- 不用于审稿回复正式指标。
- 不改变标准 FedAvg 主线。

## 9. 是否可以进入 Step 2

- 如果 Git 收口完成且本报告提交后，可以进入 Step 2：`grid_cell ablation r2e1`。

前提说明：

- 继续保持当前 `.gitignore` 规则。
- 不提交 `results/`。
- 仍按既定边界执行轻量 profiling，而不是进入正式论文实验。
