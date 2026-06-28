# 实验 1/2 formal CUDA 运行报告

## 1. 运行目标

本轮不执行 smoke，只运行真实 formal 实验 1 和实验 2。

## 2. 环境

- Python executable: `E:\anaconda3\envs\FedTrafficFlow\python.exe`
- torch version: `2.8.0+cu126`
- cuda_available: `True`
- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU`
- commit: `9323369`

## 3. 实验运行总览

| 实验 | 输出目录 | 日志文件 | 是否完成 | requested_device | actual_device | cuda_available | rounds | local_epochs | 主要结果文件 | 备注 |
|---|---|---|---|---|---|---|---:|---:|---|---|
| 实验 1 | `results/real_data_experiments/formal/exp1_single_grid_client_formal_cuda` | `logs/real_data_experiments/formal_cuda/exp1_single_grid_client_formal_cuda.log` | 是 | `cuda` | `cuda` | `true` | 20 | 3 | `main_metrics.csv`, `main_summary.csv`, `client_metrics.csv`, `convergence_history.json`, `prediction_samples.json` | 日志包含完整训练进度条与独立训练收尾信息 |
| 实验 2 | `results/real_data_experiments/formal/exp2_single_grid_ablation_formal_cuda` | `logs/real_data_experiments/formal_cuda/exp2_single_grid_ablation_formal_cuda.log` | 是 | `cuda` | `cuda` | `true` | 20 | 3 | `ablation_metrics.csv`, `ablation_summary.csv`, `ablation_client_metrics.csv` | 日志为精简输出，记录设备信息、完成标志与 variant 列表 |

## 4. 实验 1：单个网格作为单个客户端的对比实验

- 命令：`E:\anaconda3\envs\FedTrafficFlow\python.exe -m real_data_experiments.single_intersection_client.sic_core --workflow train --rounds 20 --local-epochs 3 --device cuda --output-dir results/real_data_experiments/formal/exp1_single_grid_client_formal_cuda`
- 输出目录：`results/real_data_experiments/formal/exp1_single_grid_client_formal_cuda`
- 日志路径：`logs/real_data_experiments/formal_cuda/exp1_single_grid_client_formal_cuda.log`
- run_config 设备字段：`requested_device=cuda`, `actual_device=cuda`, `cuda_available=true`, `cuda_device_name=NVIDIA GeForce RTX 3060 Laptop GPU`, `device_fallback_reason=null`
- 结果文件：`main_metrics.csv`, `main_summary.csv`, `client_metrics.csv`, `convergence_history.json`, `prediction_samples.json`, `run_config.json`
- 是否成功：是
- 日志末尾摘要：FedAvg 20 轮与独立训练均完成；尾部显示 `Independent training finished`、`Results written to ...exp1_single_grid_client_formal_cuda`、`[single_intersection_client] completed`，selected ids 为 `[290, 284, 318, 288, 289]`

## 5. 实验 2：单个网格作为单个客户端的消融实验

- 命令：`E:\anaconda3\envs\FedTrafficFlow\python.exe -m real_data_experiments.single_intersection_ablation.sia_core --workflow train --rounds 20 --local-epochs 3 --device cuda --output-dir results/real_data_experiments/formal/exp2_single_grid_ablation_formal_cuda`
- 输出目录：`results/real_data_experiments/formal/exp2_single_grid_ablation_formal_cuda`
- 日志路径：`logs/real_data_experiments/formal_cuda/exp2_single_grid_ablation_formal_cuda.log`
- run_config 设备字段：`requested_device=cuda`, `actual_device=cuda`, `cuda_available=true`, `cuda_device_name=NVIDIA GeForce RTX 3060 Laptop GPU`, `device_fallback_reason=null`
- 结果文件：`ablation_metrics.csv`, `ablation_summary.csv`, `ablation_client_metrics.csv`, `run_config.json`
- 是否成功：是
- 日志末尾摘要：日志显示 `[device] requested=cuda, actual=cuda, cuda_available=True`，随后输出 `[single_intersection_ablation] completed`，variants 为 `['Full', 'Without Attention', 'Without CNN / Spatial Encoder', 'Without LSTM']`，selected ids 为 `[290, 284, 318, 288, 289]`

## 6. 结论

- 实验 1 是否完成：是
- 实验 2 是否完成：是
- 是否均为 actual_device=cuda：是
- 是否运行 smoke：否
- 是否修改 FedAvg：否
- 是否修改模型结构：否
- 是否修改数据划分：否
