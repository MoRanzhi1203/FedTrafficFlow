# profile_tensor_experiments.py 审查报告

## 1. 本阶段范围

本阶段只对 profiling 脚本做静态审查与语法检查，不运行训练、不提交、不 push。

## 2. 文件定位

`profile_tensor_experiments.py` 的用途可定位为：

- 独立 profiling 外层包装脚本；
- 用于记录 wall-clock time、硬件环境、数据规模与估算摘要；
- 不作为训练核心逻辑；
- 不作为论文正式实验结果生成脚本。

从脚本结构看，它主要完成以下工作：

- 解析 profiling 参数；
- 构造现有实验入口所需的 `ExperimentConfig`；
- 调用既有 `run_experiment()`；
- 读取输出目录中的 `split_summary.json`、`convergence_history.csv` 等结果做统计；
- 将 profiling 行记录、硬件摘要和估算表写入 `results/real_data_experiments/compute_time_profile/`。

## 3. 安全性检查

| 检查项 | 结果 | 说明 |
|---|---|---|
| 是否修改 FedAvg | 否 | 脚本本身未实现或重写聚合逻辑，只调用既有 `sic_core` / `sia_core` / `rc_core` / `ra_core` 的 `run_experiment()` |
| 是否修改模型结构 | 否 | 未定义或修改模型类，只向既有实验配置传参 |
| 是否修改数据预处理 | 否 | 只读取现有 tensor 与 region CSV，并做数据规模摘要，不写回 `data/processed/` |
| 是否硬编码个人路径 | 否 | 使用 `Path(__file__).resolve()` 推导项目根目录，数据与输出目录均为项目内相对路径默认值 |
| 是否默认运行正式长训练 | 否 | 默认 `rounds=1`、`local_epochs=1`、`num_clients=3`，且未提供 `--setting/--task/--device` 时不会自动运行 |
| 是否写入正式数据目录 | 否 | 写入目标是 `results/real_data_experiments/compute_time_profile/` 及其子目录 |
| 是否只输出到 results profiling 目录 | 是 | 默认输出根目录为 `results/real_data_experiments/compute_time_profile/`，也允许用户显式传入 `--output-dir` |
| 是否通过 py_compile | 是 | `python -m py_compile real_data_experiments/profile_tensor_experiments.py` 已通过 |

## 4. 参数与输出路径

默认参数整体安全：

- `num_clients=3`
- `rounds=1`
- `local_epochs=1`
- `batch_size=32`
- `sequence_length=12`
- `prediction_horizon=1`
- `device`、`setting`、`task` 默认不自动设定，必须显式传参或使用 `--run-all`

输出目录定位为：

`results/real_data_experiments/compute_time_profile/`

各次 profiling 会进一步写入形如以下目录：

- `profile_cpu_grid_cell_main_r1e1`
- `profile_cpu_cluster_ablation_r1e1`

以及汇总文件：

- `profile_*_*.csv`
- `hardware_summary.json`
- `dataset_scale_summary.json`
- `compute_time_estimation_summary.csv`
- `profiling_summary.json`

额外说明：

- `--run-all` 会显式触发完整 CPU/GPU profiling 矩阵，但这不是默认行为；
- 当用户显式选择 `device=cuda` 且当前环境 `torch.cuda.is_available() == False` 时，脚本会记录 `cuda_unavailable` 并跳过真实 GPU 运行；
- 脚本不会把 profiling/smoke 结果写成论文正式结果。

## 5. 结论

结论：

- `profile_tensor_experiments.py` 可以认定为独立 profiling 工具脚本；
- 它只是包装现有实验入口，不是训练核心逻辑修改；
- 未发现 FedAvg、模型结构、数据预处理或正式 tensor 数据生成被该脚本篡改；
- 未发现依赖 C 盘用户 site-packages 或个人绝对路径。

因此：

- **建议后续单独提交该 profiling 工具脚本**

本轮审查中做了一项轻量修正：

- 补充了文件顶部 docstring，明确该脚本仅用于 profiling / wall-clock time estimation，不生成论文正式结果，默认输出到 `results/real_data_experiments/compute_time_profile/`，CUDA profiling 需在 CUDA 版 PyTorch 环境下单独运行。

## 6. 建议提交范围

如果后续决定提交，建议只提交：

```bash
git add real_data_experiments/profile_tensor_experiments.py
git add real_data_experiments/profile_tensor_experiments_review_zh.md
```

不要使用：

```bash
git add .
```

## 7. 后续建议

1. 该脚本后续可单独提交
2. CUDA 版 PyTorch 后续单独处理
3. 若要运行 GPU profiling，应先确认 `torch.cuda.is_available() == True`
