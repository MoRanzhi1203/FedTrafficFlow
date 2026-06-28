# CUDA 默认设备改造后的环境验证报告

## 1. 验证目的

本报告用于验证 `ec43e87 feat(real-data): default formal experiments to CUDA` 提交后，真实数据实验 1-6 的默认设备是否已经 CUDA 优先，以及当前 Python / Conda 环境是否能够实际使用 GPU。

本轮验证为只读静态验证与环境检查：

- 不运行完整训练。
- 不运行 formal。
- 不运行 smoke。
- 不修改 `results/` 历史结果。

## 2. Git 状态

- 最近提交：
  - `ec43e87 feat(real-data): default formal experiments to CUDA`
  - `da1ace1 docs(real-data): clarify runtime device classification`
  - `164803f docs(real-data): refine runtime estimates with concrete ranges`
- 当前未跟踪文件：
  - `.dbg/rfc-smoke-stall.env`
  - `.dbg/similarity_fedavg_probe.py`
  - `.dbg/trae-debug-log-rfc-smoke-stall.ndjson`
  - `debug-rfc-smoke-stall.md`
- 是否有未提交 tracked 修改：否。

## 3. Python / Conda 环境

| 项目 | 结果 |
|---|---|
| python executable | `E:\anaconda3\envs\FedTrafficFlow\python.exe` |
| python version | `Python 3.9.23` |
| conda env | `FedTrafficFlow`，通过显式解释器路径验证；当前 `conda env list` 输出中未直接显示该名称 |
| torch import | 成功 |
| torch version | `2.8.0+cu126` |
| torch cuda available | `True` |
| torch cuda device count | `1` |
| torch cuda device name | `NVIDIA GeForce RTX 3060 Laptop GPU` |
| nvidia-smi | 可用，Driver `560.70`，CUDA Version `12.6` |

补充说明：

- 当前 shell 默认解释器为 `E:\anaconda3\python.exe`，`python --version` 为 `3.12.3`。
- 当前 shell 默认解释器中 `import torch` 失败，说明它不是本项目的训练环境。
- 正确训练环境应使用 `E:\anaconda3\envs\FedTrafficFlow\python.exe`，并保持 `PYTHONNOUSERSITE=1`。

## 4. resolve_device 验证

| requested | actual | cuda_available | cuda_device_name | fallback_reason |
|---|---|---|---|---|
| `cuda` | `cuda` | `True` | `NVIDIA GeForce RTX 3060 Laptop GPU` | `None` |
| `gpu` | `cuda` | `True` | `NVIDIA GeForce RTX 3060 Laptop GPU` | `None` |
| `auto` | `cuda` | `True` | `NVIDIA GeForce RTX 3060 Laptop GPU` | `None` |
| `cpu` | `cpu` | `True` | `NVIDIA GeForce RTX 3060 Laptop GPU` | `None` |

结论：

- `resolve_device("cuda")` 在正确训练环境下返回 `actual_device='cuda'`。
- `resolve_device("gpu")` 与 `cuda` 等价。
- `resolve_device("auto")` 在 CUDA 可用时正确解析为 `cuda`。
- `resolve_device("cpu")` 始终保持 `cpu`。

## 5. 实验默认设备检查

| 实验 | Config | 默认 device | 是否符合 CUDA 优先 |
|---|---|---|---|
| 实验 1 | `single_intersection_client.sic_config.ExperimentConfig` | `cuda` | 是 |
| 实验 2 | `single_intersection_ablation.sia_config.ExperimentConfig` | `cuda` | 是 |
| 实验 3/4 | `region_client_full_cells.rfc_config.ExperimentConfig` | `cuda` | 是 |
| 实验 5 | `region_client.rc_config.ExperimentConfig` | `cuda` | 是 |
| 实验 6 | `region_ablation.ra_config.ExperimentConfig` | `cuda` | 是 |

补充说明：

- `--help` 静态验证已通过。
- 五条实验入口的 `--device` 参数均支持：`cuda`、`gpu`、`cpu`、`auto`。
- 帮助信息中已明确写出：默认设备为 `cuda`，若 CUDA 不可用则自动 fallback 到 CPU。

## 6. run_config 元数据出口检查

五条 core 均已包含以下字段出口或对应写入逻辑：

- `requested_device`
- `actual_device`
- `cuda_available`
- `cuda_device_name`
- `device_fallback_reason`

静态命中结果覆盖以下文件：

- `real_data_experiments/single_intersection_client/sic_core.py`
- `real_data_experiments/single_intersection_ablation/sia_core.py`
- `real_data_experiments/region_client_full_cells/rfc_core.py`
- `real_data_experiments/region_client/rc_core.py`
- `real_data_experiments/region_ablation/ra_core.py`

结论：

- 后续新生成的 `run_config.json` 可同时记录请求设备与实际设备。
- 若后续在无 GPU 环境请求 `cuda`，代码应写出：
  - `requested_device="cuda"`
  - `actual_device="cpu"`
  - `device_fallback_reason=<具体原因>`

## 7. 结论

### 情况 A：CUDA 可用

当前验证结果属于本情况。

- 正确训练环境已找到：`E:\anaconda3\envs\FedTrafficFlow\python.exe`
- `torch` 可正常导入，版本为 `2.8.0+cu126`
- `torch.cuda.is_available()` 为 `True`
- `resolve_device("cuda")` 返回 `actual_device='cuda'`
- 真实数据实验 1-6 的默认设备均已改为 `cuda` 优先
- CPU fallback 仍保留，可用于无 GPU 环境或显式 `--device cpu`

因此，后续 formal 可以按默认 CUDA 逻辑运行；若需要下一步验证，建议安排最小 CUDA smoke，但本轮不执行。

## 8. 建议下一步 smoke 验证命令

以下命令仅作为下一步建议，不在本轮执行。

```powershell
# 实验 1 CUDA smoke
python -m real_data_experiments.single_intersection_client.sic_core `
  --workflow train `
  --rounds 1 `
  --local-epochs 1 `
  --device cuda `
  --output-dir results/real_data_experiments/cuda_smoke/exp1_sic_r1e1_cuda

# 实验 2 CUDA smoke
python -m real_data_experiments.single_intersection_ablation.sia_core `
  --workflow train `
  --rounds 1 `
  --local-epochs 1 `
  --device cuda `
  --output-dir results/real_data_experiments/cuda_smoke/exp2_sia_r1e1_cuda

# 实验 3/4 CUDA smoke
python -m real_data_experiments.region_client_full_cells.rfc_core `
  --workflow train `
  --rounds 1 `
  --local-epochs 1 `
  --device cuda `
  --output-dir results/real_data_experiments/cuda_smoke/exp3_rfc_r1e1_cuda

# 实验 5 CUDA smoke
python -m real_data_experiments.region_client.rc_core `
  --workflow train `
  --rounds 1 `
  --local-epochs 1 `
  --device cuda `
  --output-dir results/real_data_experiments/cuda_smoke/exp5_rc_r1e1_cuda

# 实验 6 CUDA smoke
python -m real_data_experiments.region_ablation.ra_core `
  --workflow train `
  --rounds 1 `
  --local-epochs 1 `
  --device cuda `
  --output-dir results/real_data_experiments/cuda_smoke/exp6_ra_r1e1_cuda
```
