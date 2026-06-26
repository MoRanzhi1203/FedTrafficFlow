# 当前真实数据实验 CPU/GPU 耗时估算报告

## 1. 本阶段范围

本阶段只做耗时估算，不运行任何训练、不运行 profiling、不运行 Step 3/4、不修改代码、不生成 results。

本阶段也不修改训练代码、不修改 FedAvg 聚合逻辑、不修改 LaTeX、不修改 `simulation_experiments`、不修改 conda 环境。以下所有数字仅用于工程排程，不是正式实验结果，也不是论文指标。

## 2. 参考资料

本次估算参考了以下既有文件与结果：

- `results/real_data_experiments/gpu_light_profile/grid_cell_main_r3e1/profiling_summary.json`
- `results/real_data_experiments/gpu_light_profile/grid_cell_ablation_r2e1/profiling_summary.json`
- `results/real_data_experiments/compute_time_profile/profiling_summary.json`
- `real_data_experiments/compute_time_estimation_i7_3060_zh.md`
- `real_data_experiments/gpu_light_profiling_plan_zh.md`
- `real_data_experiments/gpu_light_profile_step1_time_estimate_zh.md`
- `real_data_experiments/gpu_light_profile_step2_time_estimate_zh.md`
- `real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md`
- `real_data_experiments/RUN_REAL_EXPERIMENTS_zh.md`
- `real_data_experiments/tensor_only_experiment_plan_zh.md`

## 3. 当前硬件与环境

- CPU：`Intel64 Family 6 Model 141 Stepping 1, GenuineIntel`
- GPU：`NVIDIA GeForce RTX 3060 Laptop GPU`
- GPU 显存：`6 GB`
- 当前显存占用：`1447 MiB / 6144 MiB`（读取时）
- 系统内存：约 `39.71 GB`
- torch 版本：`2.8.0+cu126`
- `torch.version.cuda`：`12.6`
- `cuda_available`：`True`
- 当前正式环境路径：`E:\anaconda3\envs\FedTrafficFlow`

说明：

- 当前 GPU 已可用，但桌面图形界面与常驻程序仍占用部分显存。
- 这会对 GPU 正式实验耗时产生一定波动，尤其在 Windows 笔记本环境下更明显。

## 4. 已有 GPU 实测基线

当前已有两条真实 GPU profiling 基线：

- `grid_cell main r3e1`
  - `wall_time_sec = 19.939549`
  - `num_clients = 3`
  - `rounds = 3`
  - `local_epochs = 1`
  - `batch_size = 16`
  - `sequence_length = 12`
- `grid_cell ablation r2e1`
  - `wall_time_sec = 49.222596`
  - `num_clients = 3`
  - `rounds = 2`
  - `local_epochs = 1`
  - `batch_size = 16`
  - `sequence_length = 12`
  - `variant_count = 4`

说明：

- 当前还没有 `cluster` 的 GPU profiling 实测。
- 因此 `cluster` 的 GPU 时间只能给区间估算，不能写成精确预测值。

## 5. CPU 耗时估算

下表优先采用 `results/real_data_experiments/compute_time_profile/profiling_summary.json` 与 `real_data_experiments/compute_time_estimation_i7_3060_zh.md` 中已经给出的 CPU 外推结果。

| 实验 | CPU 估算 | 依据 | 风险 |
|---|---:|---|---|
| grid_cell main full | 12.62 min | CPU 1r1e 实测 `7.573047s`，按 `5 clients / 20 rounds / 3 epochs` 外推 | 可接受，白天可跑 |
| grid_cell ablation full | 17.80 min | CPU 1r1e ablation 实测 `10.678193s`，4 个 variants 已包含在实测路径中 | 可接受，但批量运行更耗时 |
| cluster main capped | 4.87 min | CPU capped 1r1e 实测 `4.874957s`，保持 `--max-samples-per-client-split 2048` 外推 | 风险低，建议优先保留 cap |
| cluster ablation capped | 9.39 min | CPU capped ablation 1r1e 实测 `9.392968s`，4 个 variants 已包含在实测路径中 | 风险低，建议优先保留 cap |
| cluster main uncapped | 12.05 h | capped CPU 基线按全量样本放大 `148.34x` 外推 | 风险高，不建议白天直接跑 |
| cluster ablation uncapped | 23.22 h | capped CPU ablation 基线按全量样本放大 `148.34x` 外推 | 风险最高，建议单独夜间安排 |

补充说明：

- 上述 CPU 数值与既有文档中的 `12.62 min / 17.80 min / 4.87 min / 9.39 min / 12.05 h / 23.22 h` 一致，无需修正。
- 这些值是当前 CPU 正式实验最稳妥的排程参考。

## 6. GPU 耗时估算

### 6.1 估算方法

`grid_cell main` 采用 Step 1 GPU 实测线性放大：

- 基线：`19.939549s`，对应 `3 clients / 3 rounds / 1 epoch`
- 正式配置：`5 clients / 20 rounds / 3 epochs`
- 放大系数：`(5 * 20 * 3) / (3 * 3 * 1) = 33.333333`
- 估算：`19.939549 * 33.333333 = 664.651633s`

`grid_cell ablation` 采用 Step 2 GPU 实测线性放大：

- 基线：`49.222596s`，对应 `3 clients / 2 rounds / 1 epoch`
- 正式配置：`5 clients / 20 rounds / 3 epochs`
- 放大系数：`(5 * 20 * 3) / (3 * 2 * 1) = 50`
- 估算：`49.222596 * 50 = 2461.1298s`

`cluster` 当前没有 GPU Step 3/4 实测，因此只给粗略区间：

- cluster main capped GPU：约 CPU capped 的 `0.5 - 1.0` 倍
- cluster ablation capped GPU：约 CPU capped 的 `0.5 - 1.2` 倍
- cluster main uncapped GPU：约 CPU uncapped 的 `0.25 - 0.6` 倍
- cluster ablation uncapped GPU：约 CPU uncapped 的 `0.25 - 0.6` 倍

### 6.2 估算结果

| 实验 | GPU 估算 | 依据 | 风险 |
|---|---:|---|---|
| grid_cell main full | 11.08 min，保守范围约 7.75 - 16.62 min | Step 1 GPU 实测 `19.939549s` 线性放大 | 中低风险，建议优先用 GPU 跑正式主实验 |
| grid_cell ablation full | 41.02 min，保守范围约 28.71 - 61.53 min | Step 2 GPU 实测 `49.222596s` 线性放大，已含 4 个 variants | 中等风险，仍明显优于长时间 CPU 批量运行 |
| cluster main capped | 2.44 - 4.87 min | 参考 CPU capped `4.87 min`，按 `0.5 - 1.0x` 区间估算 | 缺少 GPU 实测，仅作粗略排程 |
| cluster ablation capped | 4.70 - 11.27 min | 参考 CPU capped `9.39 min`，按 `0.5 - 1.2x` 区间估算 | 缺少 GPU 实测，仅作粗略排程 |
| cluster main uncapped | 3.01 - 7.23 h | 参考 CPU uncapped `12.05 h`，按 `0.25 - 0.6x` 区间估算 | 风险高，6GB 显存与 I/O 波动都需谨慎 |
| cluster ablation uncapped | 5.81 - 13.93 h | 参考 CPU uncapped `23.22 h`，按 `0.25 - 0.6x` 区间估算 | 风险最高，强烈建议夜间独立安排 |

重要边界：

- `cluster` 的 GPU 估算目前缺少 Step 3/Step 4 实测支撑，因此只能作为粗略排程估算。
- 不能把这些 `cluster GPU` 区间写成精确 wall time。

## 7. 推荐运行顺序

建议后续正式真实数据实验按以下顺序推进：

1. `grid_cell main full`
2. `grid_cell ablation full`
3. `cluster main capped`
4. `cluster ablation capped`
5. `cluster uncapped` 最后再决定，且建议夜间运行

补充建议：

- `grid_cell` 已有 GPU 实测基线，优先用 GPU 跑更稳。
- `cluster capped` 可以先在 CPU 或 GPU 下做受控正式实验。
- `cluster uncapped` 不建议白天直接启动，应先评估显存、功耗和连续运行窗口。

## 8. 风险说明

- 所有估算仅用于工程排程。
- 所有估算都不是正式实验结果。
- 所有估算都不是论文指标。
- `cluster uncapped` 风险最高，尤其是 ablation uncapped。
- GPU 估算会受到显存占用、I/O、batch size、Windows 调度、温度和后台进程影响。
- 当前 `RTX 3060 Laptop GPU` 只有 `6GB` 显存，对 `cluster uncapped` 仍需谨慎。
- `grid_cell` 的 GPU 估算相对更可信，因为已经有 Step 1/Step 2 两条实测基线。
- `cluster` 的 GPU 估算可信度明显低于 `grid_cell`，后续若要精确排程，必须先跑一个 `cluster capped` GPU smoke / profiling。

## 9. 结论

- CPU 跑 `grid_cell` 可以接受，尤其 `grid_cell main full` 约 `12.62 min`、`grid_cell ablation full` 约 `17.80 min`。
- GPU 跑 `grid_cell` 更稳，正式主实验预计约 `11.08 min`，正式 ablation 预计约 `41.02 min`。
- `cluster capped` 在 CPU 和 GPU 下都属于可接受范围，适合作为正式实验的下一步候选。
- `cluster uncapped` 无论 CPU 还是 GPU 都是高风险长任务，不建议白天直接跑，建议单独安排夜间实验。
- 如果后续要精确估算 `cluster GPU`，需要先跑一个 `cluster capped` 的 GPU smoke / profiling，再决定是否进入 uncapped 正式实验。
