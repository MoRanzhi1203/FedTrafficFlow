# CUDA GPU Smoke 测试报告

## 1. 本阶段范围

本阶段只运行最小 GPU smoke，不运行正式训练、不运行正式 profiling、不修改训练代码、不修改 LaTeX、不修改 `simulation_experiments`、不修改 FedAvg 主线。

本阶段仅用于验证 CUDA 环境、GPU 设备可用性以及最小运行链路，不作为论文正式实验。

## 2. 环境状态

- Conda 环境路径：`E:\anaconda3\envs\FedTrafficFlow`
- Python 解释器：`E:\anaconda3\envs\FedTrafficFlow\python.exe`
- torch 版本：`2.8.0+cu126`
- `torch.version.cuda`：`12.6`
- `cuda_available`：`True`
- GPU device name：`NVIDIA GeForce RTX 3060 Laptop GPU`
- `PYTHONNOUSERSITE`：`1`
- `ENABLE_USER_SITE`：`False`

## 3. 前置 CUDA 张量测试

测试命令：

`python -c "import torch; x=torch.randn(256,256,device='cuda'); y=x @ x; torch.cuda.synchronize(); print('cuda_tensor_test_ok=', y.shape, 'mean=', float(y.mean().cpu()))"`

测试结果：`通过`

输出摘要：

- `cuda_tensor_test_ok= torch.Size([256, 256])`
- `mean= 0.10913728177547455`

## 4. GPU smoke 命令

实际执行命令：

`python real_data_experiments/profile_tensor_experiments.py --setting grid_cell --task main --device cuda --num-clients 3 --rounds 1 --local-epochs 1 --batch-size 16 --sequence-length 12 --output-dir results/real_data_experiments/gpu_cuda_smoke`

执行结果：

- 命令退出码：`0`
- 脚本输出：`[profiling] completed -> results\real_data_experiments\gpu_cuda_smoke`

## 5. 输出目录

输出目录：

`results/real_data_experiments/gpu_cuda_smoke/`

主要生成文件：

- `results/real_data_experiments/gpu_cuda_smoke/profiling_summary.json`
- `results/real_data_experiments/gpu_cuda_smoke/profile_grid_cell_main_cuda.csv`
- `results/real_data_experiments/gpu_cuda_smoke/hardware_summary.json`
- `results/real_data_experiments/gpu_cuda_smoke/dataset_scale_summary.json`
- `results/real_data_experiments/gpu_cuda_smoke/compute_time_estimation_summary.csv`
- `results/real_data_experiments/gpu_cuda_smoke/profile_gpu_grid_cell_main_r1e1/main_summary.json`
- `results/real_data_experiments/gpu_cuda_smoke/profile_gpu_grid_cell_main_r1e1/main_metrics.json`
- `results/real_data_experiments/gpu_cuda_smoke/profile_gpu_grid_cell_main_r1e1/client_metrics.json`
- `results/real_data_experiments/gpu_cuda_smoke/profile_gpu_grid_cell_main_r1e1/environment_summary.json`
- `results/real_data_experiments/gpu_cuda_smoke/profile_gpu_grid_cell_main_r1e1/experiment_notes_zh.md`

关键 smoke 摘要来自 `profiling_summary.json`：

- `status = ok`
- `exit_status = success`
- `device = cuda`
- `num_clients = 3`
- `rounds = 1`
- `local_epochs = 1`
- `batch_size = 16`
- `sequence_length = 12`
- `wall_time_sec = 18.55321`
- `gpu_max_allocated_MB = 25.358`
- `gpu_max_reserved_MB = 42.0`

smoke 后再次执行 `nvidia-smi`，未见由本次 smoke 残留的独占计算进程；显存中仍主要为系统图形界面进程占用。

## 6. 结果解释边界

- 该结果只用于验证 CUDA 环境和 GPU 运行链路。
- 不是论文正式实验结果。
- 不能作为性能比较结论。
- 不能用于正文表格或审稿回复中的正式指标。

## 7. 是否可以进入正式 GPU profiling

- 本次最小 GPU smoke 已通过，可以进入下一阶段轻量 GPU profiling 方案设计。
- 当前建议仍是先设计 profiling 范围、输出边界与资源上限，再决定是否执行轻量 profiling。
- 当前不建议基于本次 smoke 结果直接进入正式训练。
