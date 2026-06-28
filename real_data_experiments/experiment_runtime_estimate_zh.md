# 真实数据实验运行时间估算与 CPU/GPU 检查

## 1. 目的

本文件用于给出新实验 1-6 的可排期运行时间估算，并核查每条实验线当前主要使用 CPU 还是 GPU/CUDA。

## 2. 估算边界

- 本文件不运行完整训练。
- 本文件基于已有 `run_config.json`、`split_summary.json`、`profiling_summary.json`、`compute_time_estimation_summary.csv`、`profile_*.csv`、README 和 config 默认参数做静态估算。
- 本文件用于排期和资源规划，不等同于正式 benchmark。
- 当前 `results` 中已经存在部分真实 elapsed 字段，不是完全凭目录名猜测。
- 但 grouped-client / global-partition 的 GPU direct profile 仍然缺失，因此实验 3-6 的 GPU 数字属于“部分实测 + 参数化换算”，置信度低于实验 1-2 的 grid-cell GPU 估算。
- `nvidia-smi` 可用只说明机器层面支持 GPU，不能直接说明每条实验线实际用了 GPU；具体设备判断仍以 `run_config.json`、结果路径和已有 profile 为主。

### 已找到的真实 elapsed 证据

- `results/real_data_experiments/compute_time_profile/profiling_summary.json`
- `results/real_data_experiments/compute_time_profile/profile_grid_cell_main_cpu.csv`
- `results/real_data_experiments/compute_time_profile/profile_grid_cell_ablation_cpu.csv`
- `results/real_data_experiments/compute_time_profile/profile_cluster_main_cpu.csv`
- `results/real_data_experiments/compute_time_profile/profile_cluster_ablation_cpu.csv`
- `results/real_data_experiments/gpu_light_profile/grid_cell_main_r3e1/profiling_summary.json`
- `results/real_data_experiments/gpu_light_profile/grid_cell_ablation_r2e1/profiling_summary.json`

## 3. 当前运行环境

| 项目 | 检查结果 |
|---|---|
| torch version | `N/A`，当前 shell 中 `import torch` 失败 |
| cuda_available | `N/A`，当前 shell 中 `import torch` 失败 |
| cuda_device_count | `N/A`，当前 shell 中 `import torch` 失败 |
| cuda_device_name | `N/A`，当前 shell 中 `import torch` 失败 |
| nvidia-smi | 可用，检测到 `NVIDIA GeForce RTX 3060 Laptop GPU`，Driver `560.70`，CUDA `12.6` |

## 4. 估算公式与依据

### 4.1 主实验公式

对比实验采用如下公式：

```text
estimated_time =
base_profile_time
* (target_rounds / profile_rounds)
* (target_local_epochs / profile_local_epochs)
* (target_selected_clients_per_round / profile_selected_clients_per_round)
* data_scale_factor
* safety_factor
```

本文件将 `data_scale_factor` 做成 batch-adjusted 形式，以避免 profile 与目标配置的 `batch_size` 不同导致失真：

```text
data_scale_factor =
(target_avg_train_samples_per_selected_client / target_batch_size)
/
(profile_avg_train_samples_per_selected_client / profile_batch_size)
```

### 4.2 消融实验公式

消融实验统一按完整模型同设置时间做放大：

```text
estimated_ablation_time =
estimated_full_model_time
* variant_count
* variant_complexity_factor
```

本轮固定采用：

```text
variant_count = 4
variant_complexity_factor = 0.75-1.0
```

因此：

```text
estimated_ablation_time ~= estimated_full_model_time * 3.0-4.0
```

### 4.3 GPU 转换因子

- 直接实测 GPU wall-clock 只覆盖 grid-cell 主实验与 grid-cell 消融实验。
- grouped-client / global-partition 的 GPU direct profile 缺失，因此实验 3-6 的 GPU 估算采用“CPU cluster profile + 实测 grid-cell CPU/GPU throughput transfer factor”。
- transfer factor 来自以下两条真实 profile：
  - CPU：`profile_cpu_grid_cell_main_r1e1`，`wall_time_sec = 7.573047`，`rounds = 1`，`batch_size = 32`
  - GPU：`profile_gpu_grid_cell_main_r3e1`，`wall_time_sec = 19.939549`，`rounds = 3`，`batch_size = 16`
- 归一化到相同 `r1e1 + batch_size=32` 后：

```text
GPU equivalent r1e1 b32 time = 19.939549 / 3 * (16 / 32) = 3.323258 sec
CPU/GPU transfer factor = 7.573047 / 3.323258 = 2.2798
```

- 因此实验 3-6 的 GPU cluster 基线时间采用：

```text
pseudo_gpu_cluster_time = cpu_cluster_profile_time / 2.2798
```

### 4.4 Safety Factor

- CUDA：`1.15-1.35`
- CPU：`1.30-1.80`

### 4.5 参考 profile 基线

| profile | 实测 wall_time_sec | rounds | local_epochs | clients | batch_size | total_train_samples | 备注 |
|---|---:|---:|---:|---:|---:|---:|---|
| `profile_cpu_grid_cell_main_r1e1` | 7.573047 | 1 | 1 | 3 | 32 | 12261 | 真实 CPU elapsed |
| `profile_cpu_grid_cell_ablation_r1e1` | 10.678193 | 1 | 1 | 3 | 32 | 12261 | 真实 CPU elapsed，整次运行已包含 4 个变体 |
| `profile_cpu_cluster_main_r1e1` | 4.874957 | 1 | 1 | 3 | 32 | 6144 | 真实 CPU elapsed，`max_samples_per_client_split=2048` |
| `profile_cpu_cluster_ablation_r1e1` | 9.392968 | 1 | 1 | 3 | 32 | 6144 | 真实 CPU elapsed，整次运行已包含 4 个变体 |
| `profile_gpu_grid_cell_main_r3e1` | 19.939549 | 3 | 1 | 3 | 16 | 12261 | 真实 GPU elapsed |
| `profile_gpu_grid_cell_ablation_r2e1` | 49.222596 | 2 | 1 | 3 | 16 | 12261 | 真实 GPU elapsed，整次运行已包含 4 个变体 |

## 5. 新实验 1-6 总览

| 新编号 | 实验含义 | 当前结果设备 | 建议正式设备 | 当前配置依据 | smoke 估算时间 | formal 估算时间 CUDA | formal 估算时间 CPU | 置信度 |
|---|---|---|---|---|---|---|---|---|
| 新实验 1 | 单个网格作为单个客户端的对比实验 | CUDA | CUDA | formal v4 + `r40/r60/r80_e2` diagnostics | CUDA: `8.5-22.4` 分钟；CPU: `21.9-68.2` 分钟 | `6.4-7.5` 分钟 | `16.4-22.7` 分钟 | 中-高 |
| 新实验 2 | 单个网格作为单个客户端的消融实验 | CPU | CUDA | smoke 用 `single_intersection_ablation_tensor/run_config.json`；formal 用新实验 1 formal 同设置乘 `3.0-4.0` | CUDA: `7.7-12.0` 秒；CPU: `19.7-36.4` 秒 | `19.1-29.9` 分钟 | `49.2-90.9` 分钟 | 中 |
| 新实验 3 | 多个相似网格合并为一个客户端的对比实验 | CUDA | CUDA | smoke 用 `full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda`；formal 用 `rfc_config.py` 默认 `80x2` | CUDA: `6.1-7.1` 分钟；CPU: `15.7-21.7` 分钟 | `16.2-19.0` 小时 | `41.8-57.9` 小时 | 中 |
| 新实验 4 | 多个相似网格合并为一个客户端的消融实验 | 未确认 | CUDA | 无独立 formal 结果目录；按新实验 3 同设置乘 `3.0-4.0` | CUDA: `18.2-28.6` 分钟；CPU: `47.0-86.8` 分钟 | `48.7-76.2` 小时 | `125.3-231.4` 小时 | 低 |
| 新实验 5 | 全局所有网格按相似度划分为客户端的对比实验 | CPU | CUDA | smoke 用 `region_client_tensor_smoke`；formal 用 `cluster_main_full_without_cap` | CUDA: `1.2-1.4` 秒；CPU: `3.2-4.4` 秒 | `6.1-7.1` 小时 | `15.7-21.7` 小时 | 中 |
| 新实验 6 | 全局所有网格按相似度划分为客户端的消融实验 | CPU | CUDA | smoke 用 `region_ablation_tensor_smoke`；formal 用新实验 5 formal 同设置乘 `3.0-4.0` | CUDA: `3.7-5.8` 秒；CPU: `9.5-17.6` 秒 | `18.2-28.6` 小时 | `47.0-86.8` 小时 | 中 |

该表中的“当前结果设备”指已有结果目录实际使用或体现出的设备；不代表 formal 建议设备。formal 建议设备以“当前已有结果设备与 formal 建议设备”表为准。

## 当前已有结果设备与 formal 建议设备

| 实验 | 当前已有结果设备 | 当前设备依据 | 后续 formal 建议设备 | 说明 |
|---|---|---|---|---|
| 实验 1 | CUDA / GPU | `grid_cell_main_full_cuda_v4` 与 diagnostics 目录均为 CUDA 结果 | CUDA / GPU | 当前 formal/diagnostics 已经走 CUDA，后续继续用 CUDA 保持一致 |
| 实验 2 | CPU | `single_intersection_ablation` / `single_intersection_ablation_tensor` 现有结果偏 CPU | CUDA / GPU | CPU 可用于 smoke 或轻量验证，但 formal 消融建议 CUDA |
| 实验 3 | CUDA / GPU | `full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda` 与 `full_cells_spatial_k5_smoke_r1_e1_lr5e4_cuda` 为 CUDA smoke | CUDA / GPU | grouped-client formal 数据量大，CPU formal 不建议 |
| 实验 4 | 未确认 | 当前没有独立 ablation formal 结果目录 | CUDA / GPU | 建议沿用实验 3 的 CUDA 优先；当前设备不能硬判 |
| 实验 5 | CPU | `region_client_tensor_smoke` 当前 smoke 偏 CPU | CUDA / GPU | CPU 只适合连通性 smoke，formal 建议 CUDA |
| 实验 6 | CPU | `region_ablation_tensor_smoke` 当前 smoke 偏 CPU | CUDA / GPU | CPU 只适合连通性 smoke，formal 消融建议 CUDA |

## 代码默认设备修正说明

本轮之后，实验 1-6 的代码默认运行设备统一改为 `cuda` 优先。若当前环境中 `torch.cuda.is_available()` 为 `False`，代码会自动 fallback 到 CPU，并在 `run_config.json` 中记录：

- `requested_device`
- `actual_device`
- `cuda_available`
- `cuda_device_name`
- `device_fallback_reason`

注意：历史结果目录中的设备记录不会被重写。因此“当前已有结果设备”仍然保持原样：实验 1/3 为 CUDA，实验 2/5/6 为 CPU，实验 4 未确认；但“后续 formal 默认设备”已经统一改为 CUDA 优先。

## 6. 分实验说明

### 实验 1：单个网格作为单个客户端的对比实验

- 目录：`real_data_experiments/single_intersection_client/`
- 入口：`python -m real_data_experiments.single_intersection_client.sic_core`
- 目标配置：
  - formal 来源：`results/real_data_experiments/formal/grid_cell_main_full_cuda_v4/run_config.json`
  - rounds：`20`
  - local_epochs：`3`
  - batch_size：`32`
  - num_clients：`5`
  - selected_clients_per_round：`5`
  - device：`cuda`
  - train_total：`20435`
- 参考 profile：
  - CPU：`profile_cpu_grid_cell_main_r1e1/profile_grid_cell_main_cpu.csv`
  - GPU：`gpu_light_profile/grid_cell_main_r3e1/profiling_summary.json`
- 放大系数：
  - formal 相对 CPU profile：`rounds_factor=20.0`，`local_epoch_factor=3.0`，`client_factor=1.6667`，`data_scale_factor=1.0`，`safety_factor=1.30-1.80`
  - formal 相对 GPU profile：`rounds_factor=6.6667`，`local_epoch_factor=3.0`，`client_factor=1.6667`，`data_scale_factor=0.5`，`safety_factor=1.15-1.35`
- 现有 smoke 配置：
  - `r40/e3/b32`：CUDA `12.7-15.0` 分钟，CPU `32.8-45.4` 分钟
  - `r60/e3/b32`：CUDA `19.1-22.4` 分钟，CPU `49.2-68.2` 分钟
  - `r80/e2/b64`：CUDA `8.5-10.0` 分钟，CPU `21.9-30.4` 分钟
- 估算结果：
  - CUDA smoke：`8.5-22.4` 分钟
  - CUDA formal：`6.4-7.5` 分钟
  - CPU smoke：`21.9-68.2` 分钟
  - CPU formal：`16.4-22.7` 分钟

### 实验 2：单个网格作为单个客户端的消融实验

- 目录：`real_data_experiments/single_intersection_ablation/`
- 入口：`python -m real_data_experiments.single_intersection_ablation.sia_core`
- 目标配置：
  - smoke 来源：`results/real_data_experiments/single_intersection_ablation_tensor/run_config.json`
  - smoke rounds：`1`
  - smoke local_epochs：`1`
  - smoke batch_size：`32`
  - smoke num_clients：`2`
  - smoke train_total：`8174`
  - formal 来源：新实验 1 formal 同设置，再乘 `3.0-4.0`
- 参考 profile：
  - 完整模型基线：沿用实验 1 的 main profile
  - 交叉验证：`profile_cpu_grid_cell_ablation_r1e1` 与 `profile_gpu_grid_cell_ablation_r2e1`
- 放大系数：
  - smoke 同设置完整模型：`rounds_factor=1.0`，`local_epoch_factor=1.0`，`client_factor=0.6667`，`data_scale_factor=1.0`
  - ablation：`variant_count=4`，`variant_complexity_factor=0.75-1.0`
- 估算结果：
  - CUDA smoke：`7.7-12.0` 秒
  - CUDA formal：`19.1-29.9` 分钟
  - CPU smoke：`19.7-36.4` 秒
  - CPU formal：`49.2-90.9` 分钟

### 实验 3：多个相似网格合并为一个客户端的对比实验

- 目录：`real_data_experiments/region_client_full_cells/`
- 入口：`python -m real_data_experiments.region_client_full_cells.rfc_core`
- 目标配置：
  - smoke 来源：`results/real_data_experiments/diagnostics/full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda/run_config.json`
  - smoke rounds：`1`
  - smoke local_epochs：`1`
  - smoke batch_size：`32`
  - smoke num_clients：`5`
  - smoke selected_clients_per_round：`5`
  - smoke train_total：`911401`
  - formal 来源：`real_data_experiments/region_client_full_cells/rfc_config.py`
  - formal rounds：`80`
  - formal local_epochs：`2`
  - formal batch_size：`32`
  - formal num_clients：`5`
  - formal train_total：`911401`
- 参考 profile：
  - CPU：`profile_cpu_cluster_main_r1e1/profile_cluster_main_cpu.csv`
  - GPU：无 direct cluster GPU profile，采用 `CPU/GPU transfer factor = 2.2798`
- 放大系数：
  - smoke：`rounds_factor=1.0`，`local_epoch_factor=1.0`，`client_factor=1.6667`，`data_scale_factor=89.0040`
  - formal：`rounds_factor=80.0`，`local_epoch_factor=2.0`，`client_factor=1.6667`，`data_scale_factor=89.0040`
- 估算结果：
  - CUDA smoke：`6.1-7.1` 分钟
  - CUDA formal：`16.2-19.0` 小时
  - CPU smoke：`15.7-21.7` 分钟
  - CPU formal：`41.8-57.9` 小时

### 实验 4：多个相似网格合并为一个客户端的消融实验

- 目录：`real_data_experiments/region_client_full_cells/`
- 当前状态：文档与 inventory 层已补位，但独立 ablation 训练入口/正式结果目录尚不完整。
- 目标配置：
  - smoke：按新实验 3 当前 smoke 同设置，再乘 `3.0-4.0`
  - formal：按新实验 3 formal 同设置，再乘 `3.0-4.0`
- 参考 profile：
  - 完整模型基线：沿用实验 3 的 cluster main CPU profile 与 cluster GPU transfer factor
  - 变体放大：`variant_count=4`，`variant_complexity_factor=0.75-1.0`
- 估算结果：
  - CUDA smoke：`18.2-28.6` 分钟
  - CUDA formal：`48.7-76.2` 小时
  - CPU smoke：`47.0-86.8` 分钟
  - CPU formal：`125.3-231.4` 小时
- 置信度说明：
  - 当前没有独立 formal ablation 结果目录。
  - 当前没有 direct grouped-client GPU ablation profile。

### 实验 5：全局所有网格按相似度划分为客户端的对比实验

- 目录：`real_data_experiments/region_client/`
- 入口：`python -m real_data_experiments.region_client.rc_core`
- 目标配置：
  - smoke 来源：`results/real_data_experiments/region_client_tensor_smoke/run_config.json`
  - smoke rounds：`1`
  - smoke local_epochs：`1`
  - smoke batch_size：`32`
  - smoke num_clients：`3`
  - smoke selected_clients_per_round：`3`
  - smoke train_total：`3072`
  - formal 来源：`results/real_data_experiments/compute_time_profile/compute_time_estimation_summary.csv` 中 `cluster_main_full_without_cap`
  - formal rounds：`20`
  - formal local_epochs：`3`
  - formal num_clients：`3`
  - formal full_train_total：`911401`
- 参考 profile：
  - CPU：`profile_cpu_cluster_main_r1e1/profile_cluster_main_cpu.csv`
  - GPU：无 direct cluster GPU profile，采用 `CPU/GPU transfer factor = 2.2798`
- 放大系数：
  - smoke：`rounds_factor=1.0`，`local_epoch_factor=1.0`，`client_factor=1.0`，`data_scale_factor=0.5`
  - formal：`rounds_factor=20.0`，`local_epoch_factor=3.0`，`client_factor=1.0`，`data_scale_factor=148.3400`
- 估算结果：
  - CUDA smoke：`1.2-1.4` 秒
  - CUDA formal：`6.1-7.1` 小时
  - CPU smoke：`3.2-4.4` 秒
  - CPU formal：`15.7-21.7` 小时

### 实验 6：全局所有网格按相似度划分为客户端的消融实验

- 目录：`real_data_experiments/region_ablation/`
- 入口：`python -m real_data_experiments.region_ablation.ra_core`
- 目标配置：
  - smoke 来源：`results/real_data_experiments/region_ablation_tensor_smoke/run_config.json`
  - smoke rounds：`1`
  - smoke local_epochs：`1`
  - smoke batch_size：`32`
  - smoke num_clients：`3`
  - smoke selected_clients_per_round：`3`
  - smoke train_total：`3072`
  - formal 来源：按新实验 5 formal 同设置，再乘 `3.0-4.0`
- 参考 profile：
  - 完整模型基线：沿用实验 5 的 cluster main CPU profile 与 cluster GPU transfer factor
  - 变体放大：`variant_count=4`，`variant_complexity_factor=0.75-1.0`
- 估算结果：
  - CUDA smoke：`3.7-5.8` 秒
  - CUDA formal：`18.2-28.6` 小时
  - CPU smoke：`9.5-17.6` 秒
  - CPU formal：`47.0-86.8` 小时

## 7. 建议补跑的最小 profile（仅建议，不自动执行）

当前若要把实验 3-6 的 GPU 估算从“中/低置信度”提升到“中-高置信度”，建议单独补跑下列最小 profile，并显式记录 `elapsed_seconds` 或 `wall_time_sec`：

```powershell
# 实验 1 单网格主实验 CUDA profile
Measure-Command {
  python -m real_data_experiments.single_intersection_client.sic_core `
    --workflow train `
    --rounds 3 `
    --local-epochs 1 `
    --device cuda `
    --output-dir results/real_data_experiments/time_profile/grid_cell_main_cuda_r3e1
}

# 实验 3 grouped-client CUDA profile
Measure-Command {
  python -m real_data_experiments.region_client_full_cells.rfc_core `
    --workflow train `
    --rounds 3 `
    --local-epochs 1 `
    --device cuda `
    --output-dir results/real_data_experiments/time_profile/grouped_client_cuda_r3e1
}
```

## 8. 结论

- 已发现真实 elapsed 字段，但它们是部分覆盖而不是全覆盖。
- 因此当前文件不是纯猜测，但也不是全实验线完整实测 benchmark；它属于“部分实测 + 参数化外推”的可排期估算。

### 设备判断结论

当前已有结果设备：

- 实验 1：CUDA / GPU。
- 实验 2：CPU。
- 实验 3：CUDA / GPU。
- 实验 4：未确认，因为没有独立结果目录。
- 实验 5：CPU。
- 实验 6：CPU。

后续 formal 正式运行建议：

- 实验 1：建议 CUDA / GPU。
- 实验 2：建议 CUDA / GPU。
- 实验 3：必须优先 CUDA / GPU。
- 实验 4：必须优先 CUDA / GPU。
- 实验 5：必须优先 CUDA / GPU。
- 实验 6：必须优先 CUDA / GPU。

说明：

- “当前已有结果设备”不等于“后续 formal 应使用设备”。
- 实验 2、5、6 当前已有结果偏 CPU，主要是 smoke 或轻量结果；这不代表 formal 也建议 CPU。
- 实验 4 没有独立结果目录，因此当前设备不能硬判，只能按 grouped-client 消融的规模建议 CUDA。
- 实验 3-6 的 CPU formal 时间过长，因此 formal 不建议使用 CPU。

CPU 适合 smoke 或轻量验证的实验：

- 新实验 2：当前已有结果偏 CPU，但 formal 消融建议 CUDA。
- 新实验 5：当前 smoke 偏 CPU，但 formal 建议 CUDA。
- 新实验 6：当前 smoke 偏 CPU，但 formal 消融建议 CUDA。
