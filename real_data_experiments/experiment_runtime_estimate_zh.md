# 真实数据实验运行时间估算与 CPU/GPU 检查

## 1. 目的

本文件用于估算新实验 1-6 的代码运行时间，并核查每个实验当前主要使用 CPU 还是 GPU/CUDA。

## 2. 估算边界

- 本文件不运行完整训练。
- 本文件基于配置、README、`run_config.json`、`split_summary.json`、已有结果目录和环境检查估算。
- 估算结果用于排期和资源规划，不等同于正式 benchmark。
- 当前 shell 环境中的 `python` 无法直接 `import torch`，因此环境节中的 `torch` 字段以当前 shell 检查结果为准；设备判断主要依赖 `nvidia-smi`、README 命令和已有 `run_config.json`。
- `nvidia-smi` 可用只能说明机器层面支持 GPU，不能直接说明每条实验线都实际使用了 GPU；每个实验的设备判断以 `run_config.json`、结果路径和 README 命令为主。
- 估算依据优先级从高到低如下：
  - 已有结果目录中的 `run_config.json`
  - 已有结果目录中的 `split_summary.json`、`main_metrics.csv`、`ablation_metrics.csv`
  - README 或 migration 文档中的 smoke / formal 命令
  - `config.py` 中的默认参数，例如 `rounds`、`local_epochs`、`batch_size`、`num_clients`、`partition_method`、`device`
  - 若无真实 elapsed 字段，则只给区间估算，不写精确秒数

## 3. 当前运行环境

| 项目 | 检查结果 |
|---|---|
| torch version | `N/A`，当前 shell 中 `import torch` 失败 |
| cuda_available | `N/A`，当前 shell 中 `import torch` 失败 |
| cuda_device_count | `N/A`，当前 shell 中 `import torch` 失败 |
| cuda_device_name | `N/A`，当前 shell 中 `import torch` 失败 |
| nvidia-smi | 可用，检测到 `NVIDIA GeForce RTX 3060 Laptop GPU`，Driver `560.70`，CUDA `12.6` |

## 4. 新实验 1-6 总览

| 新编号 | 实验含义 | 目录 | 入口脚本 | 当前已有结果 | 当前设备判断 | smoke 估算 | formal 估算 | 置信度 |
|---|---|---|---|---|---|---|---|---|
| 新实验 1 | 单个网格作为单个客户端的对比实验 | `real_data_experiments/single_intersection_client/` | `python -m real_data_experiments.single_intersection_client.sic_core` | `grid_cell_main_full_cuda_v4`、`experiment1_fedavg_rounds_smoke_r40_cuda`、`experiment1_fedavg_rounds_smoke_r60_cuda`、`experiment1_metric_opt_k5_r80_e2_lr5e4_cuda` | GPU/CUDA | 数分钟级 | `30` 分钟到数小时 | 高 |
| 新实验 2 | 单个网格作为单个客户端的消融实验 | `real_data_experiments/single_intersection_ablation/` | `python -m real_data_experiments.single_intersection_ablation.sia_core` | `single_intersection_ablation`、`single_intersection_ablation_tensor` | CPU | `5-20` 分钟 | `2-6` 小时 | 中 |
| 新实验 3 | 多个相似网格合并为一个客户端的对比实验 | `real_data_experiments/region_client_full_cells/` | `python -m real_data_experiments.region_client_full_cells.rfc_core` | `full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda`、`full_cells_spatial_k5_smoke_r1_e1_lr5e4_cuda` | GPU/CUDA | 数分钟到十几分钟 | 数小时级，重设置时可能接近过夜 | 中 |
| 新实验 4 | 多个相似网格合并为一个客户端的消融实验 | `real_data_experiments/region_client_full_cells/` | 当前尚无独立 ablation 入口冻结为单独正式线 | 暂无独立 formal ablation 结果目录 | 未确认，按当前代码推定 GPU 优先 | `30-90` 分钟 | `1-2` 天级 | 低 |
| 新实验 5 | 全局所有网格按相似度划分为客户端的对比实验 | `real_data_experiments/region_client/` | `python -m real_data_experiments.region_client.rc_core` | `region_client_tensor_smoke` | CPU | CPU 连通性检查可在 `1-10` 分钟内完成 | 数小时级，正式跑建议改 CUDA | 中 |
| 新实验 6 | 全局所有网格按相似度划分为客户端的消融实验 | `real_data_experiments/region_ablation/` | `python -m real_data_experiments.region_ablation.ra_core` | `region_ablation_tensor_smoke` | CPU | CPU 连通性检查可在 `5-20` 分钟内完成 | 约为实验 5 同设置的 `3-4` 倍，正式跑建议改 CUDA | 中 |

## 5. 分实验说明

### 实验 1：单个网格作为单个客户端的对比实验

- 目录：`real_data_experiments/single_intersection_client/`
- 入口：`python -m real_data_experiments.single_intersection_client.sic_core`
- 已有结果：
  - `results/real_data_experiments/formal/grid_cell_main_full_cuda_v4/`
  - `results/real_data_experiments/diagnostics/experiment1_fedavg_rounds_smoke_r40_cuda/`
  - `results/real_data_experiments/diagnostics/experiment1_fedavg_rounds_smoke_r60_cuda/`
  - `results/real_data_experiments/diagnostics/experiment1_metric_opt_k5_r80_e2_lr5e4_cuda/`
- CPU/GPU 判断：
  - `sic_config.py` 默认 `device = "auto"`。
  - 现有 formal 与 diagnostics 的 `run_config.json` 都明确写 `device = "cuda"`。
  - 结果目录名也包含 `_cuda`。
  - 结论：当前已有主结果链是 GPU/CUDA。
- 运行时间估算：
  - smoke：数分钟级。
  - formal：`30` 分钟到数小时。
- 估算依据：
  - `grid_cell_main_full_cuda_v4/run_config.json`：`num_clients=5`、`batch_size=32`、`local_epochs=3`、`communication_rounds=20`、`device=cuda`。
  - `grid_cell_main_full_cuda_v4/split_summary.json`：5 个 client 每个训练集约 `4087` 个样本，合计约 `20435` 个 train windows。
  - 按 `batch_size=32` 估算，每轮约 `5 * ceil(4087/32) ≈ 640` 个 client-local batches；formal 约 `20 * 3 * 640 ≈ 38400` 个 local batches。
  - diagnostics 中还存在 `r40`、`r60`、`r80/e2/lr5e-4` 的 CUDA 结果，说明该实验线确实以 CUDA 反复运行过。
- 风险点：
  - 如果改用 CPU 跑 formal，耗时会明显拉长到数小时甚至过夜，不建议。
  - 当前 shell 无法 `import torch`，因此本次环境核验不能直接复现实验时的 `torch.cuda.is_available()`。

### 实验 2：单个网格作为单个客户端的消融实验

- 目录：`real_data_experiments/single_intersection_ablation/`
- 入口：`python -m real_data_experiments.single_intersection_ablation.sia_core`
- 已有结果：
  - `results/real_data_experiments/single_intersection_ablation/`
  - `results/real_data_experiments/single_intersection_ablation_tensor/`
- CPU/GPU 判断：
  - `sia_config.py` 默认 `device = "auto"`。
  - 两个已有结果目录的 `run_config.json` 都写 `device = "cpu"`。
  - 现有结果没有 `_cuda` 命名。
  - 结论：当前已有结果以 CPU 为主。
- 运行时间估算：
  - smoke：`5-20` 分钟。
  - formal：`2-6` 小时。
- 估算依据：
  - `single_intersection_ablation_tensor/run_config.json`：`num_clients=2`、`batch_size=32`、`local_epochs=1`、`communication_rounds=1`、`device=cpu`、4 个变体。
  - `single_intersection_ablation_tensor/split_summary.json`：2 个 client，每个 train 约 `4087` 样本，4 个消融变体。
  - 单轮单变体约 `2 * ceil(4087/32) ≈ 256` 个 local batches；4 个变体约 `1024` 个 local batches，适合作为 CPU smoke。
  - 辅助 profile 证据：
    - `compute_time_profile/profile_cpu_grid_cell_ablation_r1e1/run_config.json` 为 CPU。
    - `gpu_light_profile/grid_cell_ablation_r2e1/.../run_config.json` 为 CUDA。
  - 因此 formal 如果扩展到更多 rounds / 更大 client 集并保持 4 个变体，总耗时通常约为对应对比实验的 `3-4` 倍，建议改 CUDA。
- 风险点：
  - 现有结果是 CPU 小规模结果，不能直接等同于正式大规模正式跑时长。
  - 消融实验的总耗时会被 4 个变体线性放大。

### 实验 3：多个相似网格合并为一个客户端的对比实验

- 目录：`real_data_experiments/region_client_full_cells/`
- 入口：`python -m real_data_experiments.region_client_full_cells.rfc_core`
- 已有结果：
  - `results/real_data_experiments/diagnostics/full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda/`
  - `results/real_data_experiments/diagnostics/full_cells_spatial_k5_smoke_r1_e1_lr5e4_cuda/`
- CPU/GPU 判断：
  - `rfc_config.py` 默认 `device = "cuda"`，CLI 默认也是 `--device cuda`。
  - 两个现有 smoke 结果 `run_config.json` 都写 `device = "cuda"`。
  - 结论：当前 smoke 明确是 GPU/CUDA。
- 运行时间估算：
  - smoke：数分钟到十几分钟。
  - formal：数小时级；在更重设置下可能接近过夜。
- 估算依据：
  - `full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda/run_config.json`：`num_clients=5`、`batch_size=32`、`local_epochs=1`、`communication_rounds=1`、`device=cuda`。
  - `split_summary.json`：`used_region_count=223`，5 个 grouped clients 的 train sample 总量约 `911401`。
  - 以 `batch_size=32` 估算，单轮单 epoch 约 `911401 / 32 ≈ 28482` 个 local batches，远大于新实验 1。
  - `rfc_config.py` 默认 formal 倾向是 `rounds=80`、`local_epochs=2`、`device=cuda`，若直接按默认正式规模扩展，理论训练步数约是 smoke 的 `160` 倍。
- 风险点：
  - 这是当前 6 条线里数据量最重的一类之一。
  - 即便使用 CUDA，formal 也很容易进入过夜级。

### 实验 4：多个相似网格合并为一个客户端的消融实验

- 目录：`real_data_experiments/region_client_full_cells/`
- 入口：当前尚无独立的正式 ablation 入口冻结为单独主线；现阶段只能参照 `rfc_core` 与实验 2 的变体数推估。
- 已有结果：
  - 当前没有独立的 formal ablation 结果目录。
  - 当前只有新实验 3 的 grouped-client 对比 smoke 结果和目录级实现。
- CPU/GPU 判断：
  - 现有目录的默认配置来自 `rfc_config.py`，其默认设备是 `cuda`。
  - 但由于新实验 4 还没有独立结果目录，不能把新实验 3 的 `device=cuda` 直接等同为新实验 4 的已验证事实。
  - 结论：当前设备判断写为未确认，但按现有代码结构推定 GPU/CUDA 优先。
- 运行时间估算：
  - smoke：`30-90` 分钟。
  - formal：`1-2` 天级。
- 估算依据：
  - 新实验 4 的训练规模至少接近新实验 3。
  - 若沿用实验 2 的 4 个变体，则总耗时大致是新实验 3 同等设置的 `3-4` 倍。
  - 因新实验 3 的 full-cells grouped-client 线本身已接近过夜级，叠加 4 个消融变体后，formal 很容易达到过夜到多日级。
- 风险点：
  - 当前没有独立 ablation 训练入口和正式结果目录。
  - 因此本实验的运行时间估算置信度低于实验 1/2/3/5/6。

### 实验 5：全局所有网格按相似度划分为客户端的对比实验

- 目录：`real_data_experiments/region_client/`
- 入口：`python -m real_data_experiments.region_client.rc_core`
- 已有结果：
  - `results/real_data_experiments/region_client_tensor_smoke/`
- CPU/GPU 判断：
  - `rc_config.py` 默认 `device = "cpu"`。
  - README smoke 命令明确写 `--device cpu`。
  - `region_client_tensor_smoke/run_config.json` 也写 `device = "cpu"`。
  - 结论：当前 smoke 是 CPU。
- 运行时间估算：
  - smoke：`1-10` 分钟。
  - formal：数小时级，正式跑建议改 CUDA。
- 估算依据：
  - `run_config.json`：`num_clients=3`、`batch_size=32`、`local_epochs=1`、`communication_rounds=1`、`device=cpu`。
  - `split_summary.json`：每个 client 的 `client_sample_counts` 当前都被 cap 到 `1024`，3 个 client 总 train sample 仅 `3072`，对应总 batches 约 `96`。
  - `compute_time_profile/profile_cpu_cluster_main_r1e1/run_config.json` 也显示当前 profile 线是 CPU。
  - 但如果 future formal 去掉 `max_samples_per_client_split` 限制、改用更高 rounds 或 `flow_kmeans` 正式划分，总量会明显增大，因此正式跑更适合 CUDA。
- 风险点：
  - 当前 smoke 是轻量 capped-subset 结果，不代表 full-data 正式训练时长。
  - README 与当前 run_config 的 `partition_method` 都是 `spatial_block` smoke，不是 full formal 的最终形态。

### 实验 6：全局所有网格按相似度划分为客户端的消融实验

- 目录：`real_data_experiments/region_ablation/`
- 入口：`python -m real_data_experiments.region_ablation.ra_core`
- 已有结果：
  - `results/real_data_experiments/region_ablation_tensor_smoke/`
- CPU/GPU 判断：
  - `ra_config.py` 默认 `device = "cpu"`。
  - README smoke 命令明确写 `--device cpu`。
  - `region_ablation_tensor_smoke/run_config.json` 也写 `device = "cpu"`。
  - 结论：当前 smoke 是 CPU。
- 运行时间估算：
  - smoke：`5-20` 分钟。
  - formal：约为实验 5 同设置的 `3-4` 倍，正式跑建议改 CUDA。
- 估算依据：
  - `run_config.json`：`num_clients=3`、`batch_size=32`、`local_epochs=1`、`communication_rounds=1`、`device=cpu`、4 个变体。
  - `split_summary.json`：当前同样使用 `1024` cap 的轻量 smoke 数据规模。
  - `compute_time_profile/profile_cpu_cluster_ablation_r1e1/run_config.json` 也说明当前 profile 线是 CPU。
  - 与实验 5 相比，本实验再乘 4 个结构变体，因此 smoke 比 experiment 5 更长，formal 也会被放大到数小时甚至过夜边缘。
- 风险点：
  - 当前结果是 CPU smoke，不能直接当成 full-data、full-rounds 正式时长。
  - 一旦放开 sample cap 并保持 4 个变体，耗时会快速上升。

## 6. 推荐运行顺序

- 推荐先跑现有 smoke 线，再决定是否扩大到 formal。
- 若只做设备与链路检查，优先级从低到高建议为：
  - 新实验 5 CPU smoke
  - 新实验 1 CUDA smoke
  - 新实验 2 CPU smoke
  - 新实验 6 CPU smoke
  - 新实验 3 CUDA smoke
  - 新实验 4 grouped-client ablation smoke
- 若进入 formal，建议的资源优先级从低到高为：
  - 新实验 1 formal
  - 新实验 2 formal
  - 新实验 5 formal
  - 新实验 6 formal
  - 新实验 3 formal
  - 新实验 4 formal

## 7. 结论

- 当前已有结果判断为 GPU/CUDA 的主线是：新实验 1、新实验 3。
- 当前已有结果判断为 CPU 的主线是：新实验 2、新实验 5、新实验 6。
- 新实验 4 当前缺少独立 formal ablation 结果目录，设备判断只能写为未确认，但按实现结构推定正式运行必须优先考虑 CUDA。
- 对正式排期的建议是：
  - 新实验 1：formal 可用 CUDA 执行，属于几十分钟到数小时级。
  - 新实验 2：现有 CPU smoke 足够做轻量验证；若扩展到 formal，建议 CUDA。
  - 新实验 3：formal 基本应视为 CUDA 必需，且接近过夜级。
  - 新实验 4：formal 应视为 CUDA 必需，且可能进入 1-2 天级。
  - 新实验 5：CPU smoke 足够，但 formal 不建议继续用 CPU。
  - 新实验 6：CPU smoke 足够，但 formal 不建议继续用 CPU。
