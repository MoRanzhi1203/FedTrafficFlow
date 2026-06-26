# Batch 1：grid_cell main full 正式实验结果报告（CUDA）

## 1. 本阶段范围

本阶段只运行 Batch 1 `grid_cell main full`，`device=cuda`；未运行 Batch 2、未运行 ablation、未运行 cluster、未运行 profiling；未修改训练代码、LaTeX、`simulation_experiments`、conda 环境。

本阶段不提交 `results/`，只基于本次正式输出生成中文结果报告。

## 2. 前置环境核验

- Python executable：`E:\anaconda3\envs\FedTrafficFlow\python.exe`
- conda env：`E:\anaconda3\envs\FedTrafficFlow`
- `PYTHONNOUSERSITE = 1`
- `ENABLE_USER_SITE = False`
- `torch version = 2.8.0+cu126`
- `torch.version.cuda = 12.6`
- `cuda_available = True`
- GPU device name：`NVIDIA GeForce RTX 3060 Laptop GPU`
- `nvidia-smi` 简要状态：Driver `560.70`，CUDA `12.6`，运行前显存占用约 `1356 MiB / 6144 MiB`
- git commit：`ac933ef chore: ignore real data formal outputs`
- 运行前 `git status`：干净，`main` 与 `origin/main` 同步，`origin/main...HEAD = 0 0`

## 3. 执行命令

本次实际执行的正式训练命令为：

```powershell
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --device cuda --num-clients 5 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --selected-clients 290,284,318,288,289 --seed 42 --output-dir results\real_data_experiments\formal\grid_cell_main_full_cuda
```

按 `RUN_TENSOR_ONLY_EXPERIMENTS_zh.md` 的正式流程，本次随后补跑了同目录可视化命令：

```powershell
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all --input-dir results\real_data_experiments\formal\grid_cell_main_full_cuda --dpi 300
```

说明：

- 虽然当前执行请求中的优先配置草案给出 `batch_size = 16`，但正式 RUN 文档与 Batch 1 方案文档冻结值为 `batch_size = 32`，本次按文档冻结值执行。
- `profile_tensor_experiments.py` 只是 profiling wrapper，不作为本次正式入口。

## 4. 冻结参数

- device：`cuda`
- num_clients：`5`
- selected_clients：`290,284,318,288,289`
- rounds：`20`
- local_epochs：`3`
- batch_size：`32`
- sequence_length：`12`
- seed：`42`
- input tensor path：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- regions path：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`
- output_dir：`results/real_data_experiments/formal/grid_cell_main_full_cuda`
- workflow：`all`
- data_mode：`tensor`
- learning_rate：`0.001`

## 5. 运行耗时

- start time：`2026-06-26 22:35:39`
- 训练结果文件完成时间：`2026-06-26 22:43:49`
- 可视化结果文件完成时间：`2026-06-26 22:52:13`
- 训练阶段耗时：约 `00:08:10`
- 含可视化的完整输出链路完成时间：约 `00:16:34`
- 与 GPU 预估对比：训练阶段未超过此前 `7.75 - 16.62 min` 的 GPU 保守区间
- 是否发生 OOM：否
- 是否发生报错：否

说明：

- 训练阶段耗时依据命令启动时间与 `run_config.json` / `main_summary.csv` 的落盘时间估算；
- 可视化阶段完成时间依据 `figure_index.csv` 的落盘时间记录；
- 本次没有自动重跑，也没有自动切换 CPU。

## 6. 输出文件核验

输出目录：`results/real_data_experiments/formal/grid_cell_main_full_cuda`

关键训练输出文件：

- `run_config.json`
- `split_summary.json`
- `main_metrics.csv`
- `main_summary.csv`
- `client_metrics.csv`
- `convergence_history.csv`
- `prediction_samples.csv`

关键可视化输出文件：

- `figure_index.csv`
- `figure_notes_zh.md`

其余已生成文件还包括：

- `environment_summary.json`
- `experiment_notes_zh.md`
- `run_commands.txt`
- `selected_regions.csv`
- `main_metrics.json`
- `main_summary.json`
- `client_metrics.json`
- `convergence_history.json`
- `prediction_samples.json`
- `main_metrics_comparison.png`
- `client_metrics_comparison.png`
- `convergence_curve.png`
- `prediction_vs_ground_truth.png`

核验结论：

- 输出目录存在；
- `main_metrics.csv` 存在；
- `main_summary.csv` 存在；
- `run_config.json` 存在；
- 指标可被 `pandas` 正常读取；
- 数值列未发现 `NaN / Inf`。

## 7. 主要指标摘要

`main_metrics.csv` 实际提供了两行方法对比：`FedAvg` 与 `Independent`。本次正式实验主线指标以 `FedAvg` 为主，具体如下：

### 7.1 FedAvg

- MSE：`3320353503963.0122`
- RMSE：`1818555.5234`
- MAE：`1816675.5764`
- MAPE：`99.9932901346`
- SMAPE：`199.9731623504`
- R2：`-700.76445973`

### 7.2 Independent

- MSE：`3320450185889.8125`
- RMSE：`1818581.7807`
- MAE：`1816701.8598`
- MAPE：`99.9947286453`
- SMAPE：`199.9789157328`
- R2：`-700.7853923282`

### 7.3 main_summary.csv 摘要

`main_summary.csv` 额外给出了按 5 个客户端汇总后的 `mean / std / cv / min / max / count`。其中 `FedAvg` 统计摘要为：

- `mse`：mean `3320353503963.0122`，std `414864837216.4174`
- `rmse`：mean `1818555.5234`，std `114931.7717`
- `mae`：mean `1816675.5764`，std `114289.1484`
- `mape`：mean `99.9932901346`，std `0.0004246013`
- `smape`：mean `199.9731623504`，std `0.0016981763`
- `r2`：mean `-700.76445973`，std `541.5987508734`

说明：

- `main_metrics.csv` / `main_summary.csv` 中未提供单独命名为“MAE 以外的其他新指标”；因此本报告不编造额外指标。
- 当前结果只记录文件中实际存在的指标。

## 8. 客户端与数据划分摘要

依据 `split_summary.json` 与 `run_config.json`：

- tensor shape：`(2, 630, 5856)`
- total_region_count：`630`
- active_region_count：`223`
- used_region_count：`5`
- selected clients：`[290, 284, 318, 288, 289]`
- split strategy：`temporal_contiguous_by_target_time`
- train / val / test ratio：`0.70 / 0.15 / 0.15`
- train range：`0 - 4099`
- val range：`4099 - 4977`
- test range：`4977 - 5856`
- 每个 client 的样本量：
  - train：`4087`
  - val：`878`
  - test：`879`

各 client 对应 region 元数据如下：

- `290`：`pooled_row=9`，`pooled_col=20`，`source_node_count=667`
- `284`：`pooled_row=9`，`pooled_col=14`，`source_node_count=698`
- `318`：`pooled_row=10`，`pooled_col=18`，`source_node_count=711`
- `288`：`pooled_row=9`，`pooled_col=18`，`source_node_count=698`
- `289`：`pooled_row=9`，`pooled_col=19`，`source_node_count=663`

## 9. 结果性质说明

- 本次结果是 Batch 1 `grid_cell main full` 的正式实验候选结果；
- 本次结果仍需与后续 `ablation / cluster` 结果一起统一审查；
- 本次没有把 smoke / profiling / 耗时估算结果混入正式指标；
- 本次没有改变标准样本量加权 FedAvg 主线；
- 本次使用的是 tensor-only 正式数据入口，没有重新生成 tensor 数据。

## 10. Git 与提交边界

- `results/` 不提交；
- 本阶段只生成本报告文档；
- 当前未执行 `git add`、`git commit`、`git push`；
- 后续如需提交，只提交报告文档，不提交 `results/`；
- `.pt / .npy / parquet / png / pdf` 大文件与中间结果文件不纳入本次提交范围。

## 11. 结论

- Batch 1 `grid_cell main full` 已成功完成；
- 本次按 `device=cuda` 运行，未自动切换 CPU；
- 正式训练输出与可视化索引都已生成，关键文件齐全，指标可读且未发现 `NaN / Inf`；
- 从流程完整性看，已经具备进入 Batch 2 规划或执行准备阶段的条件；
- 建议下一步先单独提交本报告文档，再生成 Batch 2：`grid_cell ablation full` 的运行方案或精确执行指令。
