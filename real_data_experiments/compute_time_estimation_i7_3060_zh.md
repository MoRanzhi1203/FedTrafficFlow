# i7 + RTX 3060 笔记本真实数据实验计算量与耗时估算

## 1. 本阶段范围

本阶段只做 GPU/CPU profiling、时间估算和资源评估，不运行完整正式长训练，不修改训练逻辑，不修改模型结构，不修改标准 FedAvg 聚合，不修改 LaTeX，不修改 `simulation_experiments/`，不重新生成正式 tensor 数据。

当前 profiling 通过独立脚本 `real_data_experiments/profile_tensor_experiments.py` 完成，统一输出到 `results/real_data_experiments/compute_time_profile/`。

## 2. 硬件与软件环境

| 项目 | 值 |
|---|---|
| CPU | Intel64 Family 6 Model 141 Stepping 1, GenuineIntel |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU |
| VRAM | 6 GB |
| System RAM | 39.71 GB |
| OS | Windows-10-10.0.19042-SP0 |
| Python | 3.12.3（旧 profiling 阶段环境） |
| PyTorch | 2.12.0+cpu（旧 profiling 阶段环境） |
| CUDA available | False |
| CUDA version in PyTorch | None |
| NVIDIA Driver / Runtime | Driver 560.70 / CUDA Runtime 12.6 |

说明：

- 上表保留的是最初 profiling 阶段的硬件与软件快照，其中 Python / PyTorch 版本属于历史记录，不代表当前正式环境。
- 当时机器物理上已具备 RTX 3060 6GB，但 profiling 所在 PyTorch 环境为 CPU-only，因此没有完成真实 GPU profiling。

## 2.1 当前正式环境状态

本报告最初基于 profiling 阶段的临时环境生成；后续项目环境已完成隔离修复与依赖重装。当前正式 CPU 运行环境如下：

| 项目 | 当前状态 |
|---|---|
| Conda 环境路径 | `E:\anaconda3\envs\FedTrafficFlow` |
| Python 解释器 | `E:\anaconda3\envs\FedTrafficFlow\python.exe` |
| Python 版本 | `3.9` |
| PyTorch 版本 | `2.8.0+cpu` |
| torch CUDA | `None` |
| cuda_available | `False` |
| Jupyter kernel | `Python (FedTrafficFlow)` |
| PYTHONNOUSERSITE | `1` |
| ENABLE_USER_SITE | `False` |

因此，本文档中的 GPU 内容应理解为“CUDA 不可用条件下的 CPU profiling 与 GPU 理论建议”，而不是实际 GPU profiling 结果。

## 3. 数据规模

| 项目 | 值 |
|---|---|
| tensor shape | `(2, 630, 5856)` |
| dtype | `torch.float32` |
| finite | `true` |
| C / R / T | `2 / 630 / 5856` |
| total regions | `630` |
| active regions | `223` |
| sequence length | `12` |
| horizon | `1` |
| train_end / val_end / test_end | `4099 / 4977 / 5856` |
| estimated tensor memory | `28.147 MB` |

补充：

- cluster-level profiling 使用 `--num-clients 3 --partition-method spatial_block --max-samples-per-client-split 2048`。
- 该 capped profiling 的总训练样本数为 `6144`。
- 若 cluster-level 正式运行不设 cap，按同样 3-client 划分估算，总训练样本数约为 `911401`，相对 profiling 放大约 `148.34x`。

## 4. 两类客户端设置

| 设置 | 英文名称 | client 定义 | 对应代码 |
|---|---|---|---|
| 网格单元级客户端联邦学习设置 | Grid-cell-level Client Federated Learning Setting | 每个 client = 1 个 pooled grid cell | `single_intersection_client/*`, `single_intersection_ablation/*` |
| 簇级客户端联邦学习设置 | Cluster-level Client Federated Learning Setting | 每个 client = 一组 pooled grid cells / pooled grid regions | `region_client/*`, `region_ablation/*` |

## 5. GPU profiling 结果

当前正式环境为 `torch 2.8.0+cpu`，`torch.cuda.is_available() = False`。虽然 `nvidia-smi` 可以识别 RTX 3060，但当前并没有安装 CUDA 版 PyTorch，因此四类 GPU profiling 都未执行真实 GPU 训练，CSV 仅记录了 `cuda_unavailable` 状态。

| setting | task | time/sec | samples | GPU memory | 状态 |
|---|---|---:|---:|---:|---|
| grid_cell | main | N/A | N/A | N/A | CUDA unavailable，需要 CUDA 版 PyTorch 复测 |
| grid_cell | ablation | N/A | N/A | N/A | CUDA unavailable，需要 CUDA 版 PyTorch 复测 |
| cluster | main | N/A | N/A | N/A | CUDA unavailable，需要 CUDA 版 PyTorch 复测 |
| cluster | ablation | N/A | N/A | N/A | CUDA unavailable，需要 CUDA 版 PyTorch 复测 |

理论判断：

- 正式 tensor 本体仅约 `28.15 MB`，模型也较轻，`RTX 3060 6GB` 从容量角度看大概率足以支撑 grid-cell-level 与 cluster-level capped 方案。
- 但由于当前正式环境仍是 CPU-only PyTorch，本阶段不能给出可信的 `gpu_max_allocated_MB` / `gpu_max_reserved_MB` 实测值，也不能把 GPU 耗时写成实测结果。

## 6. CPU-only profiling 结果

| setting | task | time/sec | samples | RAM | 状态 |
|---|---|---:|---:|---:|---|
| grid_cell | main | 7.57 | 12261 | RSS `321.6 -> 613.9 MB` | ok |
| grid_cell | ablation | 10.68 | 12261 | RSS `613.9 -> 614.4 MB` | ok |
| cluster | main | 4.87 | 6144 | RSS `614.6 -> 638.9 MB` | ok |
| cluster | ablation | 9.39 | 6144 | RSS `638.9 -> 617.7 MB` | ok |

说明：

- CPU profiling 记录的是同一 profiling 进程的运行前后 RSS，不是严格峰值内存。
- 从本次结果看，CPU-only capped profiling 的进程级内存占用大致在 `0.6 GB` 左右，没有出现内存瓶颈迹象。
- grid-cell-level 3 client profiling 总训练样本数为 `12261`。
- cluster-level capped profiling 总训练样本数为 `6144`，每个 client 为 `2048`。

## 7. GPU 与 CPU 对比

| 实验 | GPU 1r1e | CPU 1r1e | 加速比 | GPU full 估算 | CPU full 估算 | 建议 |
|---|---:|---:|---:|---:|---:|---|
| Grid-cell-level main | CUDA unavailable | 7.57 s | N/A | CUDA unavailable | 12.62 min | CPU 可接受，GPU 可作为加速选项 |
| Grid-cell-level ablation | CUDA unavailable | 10.68 s | N/A | CUDA unavailable | 17.80 min | CPU 可接受，批量任务时更建议 GPU |
| Cluster-level main with cap | CUDA unavailable | 4.87 s | N/A | CUDA unavailable | 4.87 min | CPU 可接受，建议保留 cap |
| Cluster-level main without cap | CUDA unavailable | 4.87 s | N/A | CUDA unavailable | 12.05 h | 强烈建议 GPU，且建议先保留 cap |
| Cluster-level ablation with cap | CUDA unavailable | 9.39 s | N/A | CUDA unavailable | 9.39 min | CPU 可接受，建议保留 cap |
| Cluster-level ablation without cap | CUDA unavailable | 9.39 s | N/A | CUDA unavailable | 23.22 h | 强烈建议 GPU + 夜间运行 |

说明：

- 由于没有真实 GPU 1r1e 数据，当前无法给出可信的 GPU/CPU 加速比。
- 只有在安装 CUDA 版 PyTorch 并复测后，才能填入可靠的 GPU wall time、显存峰值和加速比。

## 8. 正式实验耗时估算

本报告中的耗时估算基于 CPU-only profiling。由于当前正式环境为 `torch 2.8.0+cpu`，GPU 相关内容仅作为后续安装 CUDA 版 PyTorch 后的运行建议，不代表已经完成 GPU 实测。

估算公式采用：

`estimated_time = measured_wall_time * (target_rounds / measured_rounds) * (target_local_epochs / measured_local_epochs) * sample_scale`

其中：

- grid-cell-level 从 3 clients 外推到 5 clients 时，`sample_scale = 5 / 3`。
- cluster-level capped 方案保留 `--max-samples-per-client-split 2048` 时，`sample_scale = 1`。
- cluster-level uncapped 方案用 `full_train_samples / profiled_train_samples = 911401 / 6144 ≈ 148.34` 外推。

结果如下：

- grid-cell-level quick
  - GPU：CUDA unavailable，需要 CUDA 版 PyTorch 复测。
  - CPU：约 `113.60 s`，约 `1.89 min`。
- grid-cell-level full
  - GPU：CUDA unavailable，需要 CUDA 版 PyTorch 复测。
  - CPU：约 `757.30 s`，约 `12.62 min`。
- grid-cell-level ablation
  - GPU：CUDA unavailable，需要 CUDA 版 PyTorch 复测。
  - CPU：约 `1067.82 s`，约 `17.80 min`。
- cluster-level quick
  - 保留 cap 时：
    - GPU：CUDA unavailable，需要 CUDA 版 PyTorch 复测。
    - CPU：约 `73.12 s`，约 `1.22 min`。
  - 不保留 cap 时：
    - GPU：CUDA unavailable，需要 CUDA 版 PyTorch 复测。
    - CPU：约 `10847.27 s`，约 `180.79 min`，约 `3.01 h`。
- cluster-level full with cap
  - GPU：CUDA unavailable，需要 CUDA 版 PyTorch 复测。
  - CPU：约 `292.50 s`，约 `4.87 min`。
- cluster-level full without cap
  - GPU：CUDA unavailable，需要 CUDA 版 PyTorch 复测。
  - CPU：约 `43389.07 s`，约 `723.15 min`，约 `12.05 h`。
- cluster-level ablation with cap
  - GPU：CUDA unavailable，需要 CUDA 版 PyTorch 复测。
  - CPU：约 `563.58 s`，约 `9.39 min`。
- cluster-level ablation without cap
  - GPU：CUDA unavailable，需要 CUDA 版 PyTorch 复测。
  - CPU：约 `83601.18 s`，约 `1393.35 min`，约 `23.22 h`。

## 9. i7 + RTX 3060 笔记本建议

- 哪些实验可以 CPU 跑：
  - `grid-cell-level quick`
  - `grid-cell-level full`
  - `grid-cell-level ablation full`
  - `cluster-level quick with cap`
  - `cluster-level full with cap`
  - `cluster-level ablation with cap`
- 哪些实验强烈建议 GPU：
  - `cluster-level quick` 不设 cap
  - `cluster-level full without cap`
  - `cluster-level ablation without cap`
  - 在需要批量重复运行 grid ablation 时，也建议 GPU
- 哪些实验建议夜间运行：
  - `cluster-level full without cap`
  - `cluster-level ablation without cap`
  - 若后续装好 CUDA 并改成 GPU 跑，以上两类仍建议夜间跑
- batch_size 是否保持 32：
  - 目前 profiling 未发现 batch size 32 带来的明显内存压力。
  - 对 grid-cell-level 和 cluster-level capped 方案，建议保持 `batch_size=32`。
  - 对 cluster-level uncapped 方案，当前首要问题是总样本量与总时间，而不是 batch size；优先保留 cap 或降低 rounds / local_epochs，不建议先动 batch size。
- 是否建议 `--max-samples-per-client-split`：
  - 强烈建议 cluster-level 在笔记本上保留 `--max-samples-per-client-split 2048`。
  - 若必须做 uncapped 正式实验，建议先装 CUDA 版 PyTorch，再做单独 GPU profiling 复测。
- 是否建议先跑 quick 再跑 full：
  - 是。建议严格先跑 quick，再决定是否进入 full。
  - 对 cluster-level，尤其要先跑 capped quick，再决定是否尝试 uncapped。
- 温度、电源模式、散热建议：
  - 插电运行，避免电池供电降频。
  - Windows 设为高性能或最佳性能模式。
  - 关闭高占用浏览器标签和视频会议进程，当前桌面程序已占用部分 GPU 显存。
  - 建议抬高机身或使用散热底座，避免长时间满载导致降频。
  - 夜间长跑时定期检查温度、风扇噪声和系统休眠设置。

## 10. 风险与限制

- 估算基于 `1 round / 1 epoch` profiling 外推，仍然属于一阶近似。
- Windows 笔记本的 CPU/GPU 性能会受温度、功耗墙和后台进程影响。
- 当前正式环境为 `torch 2.8.0+cpu`，GPU wall time、显存峰值、GPU/CPU 加速比都不能做可信实测。
- cluster-level without cap 的样本量相对 capped profiling 放大约 `148.34x`，耗时会显著增加。
- ablation 含 `4` 个模型变体，耗时天然约为对应单模型 main 的多倍。
- CPU profiling 记录的是进程前后 RSS，不是严格峰值内存。
- 如需可信 GPU 估算，必须安装 CUDA 版 PyTorch 后复测四类 GPU profiling。
