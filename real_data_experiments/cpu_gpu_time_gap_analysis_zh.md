# 真实数据实验 CPU/GPU 耗时差异分析与优化建议

## 1. 本阶段范围

本阶段只做静态分析和已有结果解释，不运行训练、不运行 profiling、不运行 Step 3/4、不修改代码、不生成新的 `results`。

本阶段也不修改训练代码、不修改标准样本量加权 FedAvg、不修改 LaTeX、不修改 `simulation_experiments`、不修改 conda 环境，不安装或卸载任何包。本文档仅用于解释当前真实数据实验的 CPU/GPU 耗时差异，并给出不改变论文主线的工程优化建议。

## 2. 已有耗时依据

本阶段分析主要基于以下已有资料：

- Step 1 `grid_cell main r3e1` GPU 实测：`wall_time_sec = 19.939549`
- Step 2 `grid_cell ablation r2e1` GPU 实测：`wall_time_sec = 49.222596`
- CPU 侧已有估算来源：`real_data_experiments/compute_time_estimation_i7_3060_zh.md`
- CPU profiling 汇总来源：`results/real_data_experiments/compute_time_profile/profiling_summary.json`
- 当前 CPU/GPU 总估算来源：`real_data_experiments/real_data_cpu_gpu_time_estimate_zh.md`

补充说明：

- 当前 GPU 实测只覆盖 `grid_cell` 的轻量 profiling，尚未覆盖 `cluster`。
- 当前分析还静态查看了 `profile_tensor_experiments.py`、`tensor_dataset.py`、`region_tensor_dataset.py`、`sic_core.py`、`sia_core.py`、`rc_core.py`、`ra_core.py`、`client.py`、`trainer.py`、`fedavg.py`、`result_writer.py` 等文件。
- 因此，本文档中的 `cluster GPU` 判断仍然属于“静态结构分析 + 区间估算”，不是新的 GPU 实测结果。

## 3. 为什么真实数据实验仍然耗时较长

真实数据正式实验比 smoke 和 light profiling 慢，不是单一因素造成的，而是多个训练单元叠加后的总耗时结果。

### 3.1 client 数放大

轻量 GPU profiling 的 `grid_cell` 基线只使用了 `3 clients`，而正式实验通常按 `5 clients` 运行。联邦学习中 client 数增加后，并不是只多一点计算，而是会让每一轮的本地训练、参数拷贝、FedAvg 聚合、验证评估都成比例增加。

### 3.2 rounds 与 local_epochs 放大

轻量 profiling 中的 `grid_cell main r3e1` 只跑了 `3 rounds / 1 local epoch`，但正式实验通常要到 `20 rounds / 3 local epochs`。这意味着单个任务的训练循环单元数会明显放大，wall-clock 时间自然会显著增长。

从工程结构上看，当前项目的训练主循环不是一次连续的大训练，而是：

- 每一轮 server 下发全局参数；
- 每个 client 各自完成 local train；
- server 再做一次标准样本量加权 FedAvg；
- 然后还要做验证与测试指标计算。

这类“多 client + 多 round + 多 local epoch”的碎片化循环，本身就比单机单模型连续训练更容易累积调度开销。

### 3.3 ablation variant 数放大

`ablation` 任务不是在 `main` 基础上只多一点判断逻辑，而是要把以下 4 个变体串行完整跑完：

- `full`
- `without_attention`
- `without_cnn`
- `without_lstm`

也就是说，ablation 的本质更接近“4 套完整训练链路顺序执行”，而不是“1 套训练多导出几个指标”。因此 ablation 比 main 更重，是结构性增重，不是偶然波动。

### 3.4 样本量与 cluster uncapped 放大

根据已有 CPU profiling 与估算资料，`cluster capped` profiling 的总训练样本数约为 `6144`，而 `cluster uncapped` 正式训练样本数约为 `911401`，放大约 `148.34x`。

这说明：

- `cluster capped` 和 `cluster uncapped` 的耗时不是同一量级；
- `cluster uncapped` 是当前真实数据实验中最大的耗时风险；
- 即使 GPU 能提供一定加速，面对 `148.34x` 的样本量放大，整体任务依然可能达到小时级。

### 3.5 数据加载与时间窗口构造仍有 CPU 成分

静态代码显示，数据张量首先通过 `torch.load(..., map_location="cpu")` 加载到 CPU；随后：

- `GridTensorWindowDataset` 在 `__getitem__` 中按样本执行时间窗切片；
- `RegionClientWindowDataset` 在 `__getitem__` 中按 `region_id` 和 `target_time` 执行窗口索引与目标值提取；
- `DataLoader` 默认构造里没有显式设置 `num_workers` 或 `pin_memory`。

这意味着 GPU 并没有接管整个数据链路。真实实验中，Dataset / DataLoader / 时间窗口构造仍然有相当一部分工作发生在 CPU 端。

### 3.6 指标计算与文件输出会增加非训练耗时

当前项目在训练完成后会写出较多文件，包括但不限于：

- `run_config.json`
- `environment_summary.json`
- `split_summary.json`
- `selected_regions.csv`
- `main_metrics.csv/json`
- `main_summary.csv/json`
- `client_metrics.csv/json`
- `convergence_history.csv/json`
- `prediction_samples.csv/json`
- `ablation_metrics.csv`
- `ablation_summary.csv`
- `ablation_client_metrics.csv`
- `region_assignment.csv`
- `client_distribution_summary.csv`
- `non_iid_summary.csv`
- `experiment_notes_zh.md`

这些写入与指标汇总本身不一定是主耗时，但在多任务、多 variant、长时间实验里会持续累积 wall-clock 成本，尤其在 Windows 笔记本环境下更容易体现为 I/O 与调度噪声。

### 3.7 为什么真实数据实验比 smoke / light profiling 慢

综合来看，真实数据正式实验比 smoke / light profiling 慢，主要是因为：

- 正式实验的 client 数更多；
- rounds 和 local_epochs 更高；
- ablation 需要串行跑多个 variant；
- cluster 尤其 uncapped 的样本量显著膨胀；
- 数据构造、DataLoader、指标计算和文件输出并未完全转移到 GPU；
- 联邦学习的 client-local train + server aggregation 结构天然更碎片化。

因此，真实数据实验耗时长是符合当前任务结构的，不意味着 GPU 环境异常，也不意味着代码出现错误。

## 4. 为什么部分任务 CPU 可能比 GPU 更快

GPU 并不保证在所有任务上都优于 CPU。GPU 擅长的是大 batch、大矩阵、连续高强度计算；而当前项目中有不少任务更接近“小 batch、小模型、多轮碎片化调度”。

### 4.1 batch_size 较小时，GPU 调度开销占比会偏高

现有 GPU 轻量 profiling 基线使用的是 `batch_size = 16`。当 batch 很小时：

- 单次前向和反向计算量有限；
- GPU kernel 启动开销更难被摊薄；
- CPU 到 GPU 的数据搬运与同步成本占比会上升。

因此，`batch_size = 16` 的情况下，GPU 未必能够充分吃满，甚至可能出现“GPU 利用不高，但 wall time 仍然不短”的现象。

### 4.2 模型规模不够大时，GPU 优势不一定明显

当前主模型是相对轻量的 `CNN + LSTM + Attention` 回归器，hidden_dim 也不大。对于这种规模较小的模型：

- GPU 的理论吞吐优势不一定能完全释放；
- CPU 在缓存命中、调度稳定性、小任务切换方面反而可能更直接；
- 小模型配合小 batch 时，更容易出现“GPU 算得快，但总流程没有明显更快”的结果。

### 4.3 联邦训练循环是碎片化的，不是连续大训练

静态代码表明，当前联邦训练存在明显碎片化特征：

- 每轮对每个 client 依次做 local training；
- 每个 client 训练时都要重新实例化模型并加载全局参数；
- 每个 batch 再执行 `features.to(device)` 与 `targets.to(device)`；
- 每轮结束后还要做 FedAvg 聚合和验证评估。

这种结构意味着 GPU 大量时间并不是在持续执行大矩阵运算，而是在频繁响应许多较小的训练片段与数据传输请求。对于这类任务，CPU 可能更稳定，甚至在某些较小任务上更快。

### 4.4 DataLoader 与样本构造仍在 CPU

当前 `DataLoader` 没有显式设置：

- `num_workers`
- `pin_memory`

同时 Dataset 的窗口切片、索引与样本构造逻辑主要也在 CPU 端完成。结果就是：

- GPU 只负责模型前向、反向和参数更新；
- 数据准备阶段仍由 CPU 驱动；
- 如果 CPU 供数不够连续，GPU 会出现“等数据”的情况。

因此，部分任务即使放到 GPU，也不一定能拿到明显加速。

### 4.5 Windows + RTX 3060 Laptop GPU + 6GB VRAM 也有现实约束

当前环境是 Windows 笔记本，GPU 为 `RTX 3060 Laptop GPU 6GB`。这类环境的特点是：

- WDDM 图形栈和桌面程序会持续占用部分显存；
- 笔记本功耗墙、温度墙和后台进程更容易引入波动；
- 显存余量和系统调度不如独立服务器稳定。

因此，GPU 虽然可用，但在“小任务 + 多轮切换 + 多 client + 背景程序存在”的工程场景下，不一定在所有任务上都明显优于 CPU。

### 4.6 为什么部分 GPU 估算不一定比 CPU 更快

结合现有估算结果可以看出：

- `grid_cell main` 的 GPU 预计略优于 CPU；
- 但 `grid_cell ablation` 的 GPU 估算显著长于 CPU 估算；
- `cluster capped` 的 GPU 目前只能给区间，且区间上界并不保证优于 CPU；
- `cluster uncapped` 虽然理论上更适合 GPU，但仍然可能被样本量、I/O 与调度成本拖成小时级任务。

因此，“GPU 不一定总比 CPU 快”在当前项目是合理判断，而不是异常现象。

## 5. 各类实验的 CPU/GPU 适配判断

| 实验类型 | 更适合 CPU/GPU | 原因 | 建议 |
|---|---|---|---|
| grid_cell main | CPU/GPU 都可，偏向 GPU | 已有 GPU 实测基线；任务规模中等；正式实验仍会被多轮 FedAvg 与评估开销放大，但 GPU 预计略优于 CPU | 若白天希望更快完成，可优先 GPU；若追求稳定与简洁排程，CPU 也可接受 |
| grid_cell ablation | CPU 或夜间 GPU | ablation 要串行跑 4 个 variants；任务碎片化明显；小 batch 下 GPU 调度优势不一定稳定释放 | 若强调可控与稳定，可优先 CPU；若需要与后续 cluster 统一走 GPU，可放夜间 GPU |
| cluster main capped | 偏向 GPU，但 CPU 也可 | capped 后样本量可控；cluster 比 grid 更接近 GPU 擅长的较大样本任务；但目前缺少 GPU 实测 | 建议先按 GPU 候选排程；若需要更稳妥判断，先做一次单独 capped smoke 再决定正式设备 |
| cluster ablation capped | CPU/GPU 都可，偏向 GPU 夜间 | capped 使样本量仍可控，但 ablation 的 4 个 variants 会明显放大总耗时；缺少 GPU 实测 | 可先 GPU，但更适合夜间窗口；如果白天资源紧张，CPU 也可作为保守方案 |
| cluster uncapped | 偏向 GPU，但风险最高 | 样本量相对 capped 放大约 `148.34x`；这是当前最接近 GPU 适用场景的大样本任务，但 6GB 显存、I/O、Windows 调度都可能成为风险 | 不建议白天直接跑；应最后再决定，优先夜间独立运行，并保留中止观察策略 |

补充判断：

- 当前任务结构整体并不是“非常理想的 GPU 任务”，因为它不是连续大 batch 训练，而是联邦学习下的碎片化循环。
- 但当任务样本量逐步放大到 `cluster` 尤其 `uncapped` 时，GPU 仍然更有潜在收益。
- 小模型、小 batch、多循环任务更容易让 CPU 表现得“更稳”，而不一定“更慢”。

## 6. 不改变论文主线的优化建议

以下建议只属于工程优化方向，不改变正式实验主线，不改变标准样本量加权 FedAvg，需要单独 smoke 后再决定是否用于正式实验。

### 6.1 GPU 任务可尝试把 batch_size 从 16 提升到 32

当前 GPU 轻量 profiling 基线采用的是 `batch_size = 16`。对于 GPU 来说，这个 batch 偏小，容易导致调度开销占比过高。

因此可考虑：

- 仅在 GPU 任务中，先做一次独立 smoke；
- 将 `batch_size` 从 `16` 试探到 `32`；
- 观察 wall time、显存占用和是否稳定。

这属于工程调优，不改变模型结构，也不改变 FedAvg 聚合逻辑。

### 6.2 对 GPU 任务尝试 `pin_memory` / `num_workers`

当前 DataLoader 未显式设置 `pin_memory` 或 `num_workers`。如果后续希望提高 GPU 任务的数据供给效率，可考虑：

- 仅针对 GPU 路径尝试 `pin_memory=True`；
- 仅小步测试 `num_workers`，如 `1` 或 `2`；
- 每次只改一个参数，并保留 smoke 日志。

这类优化主要作用于数据搬运与供数效率，不改变实验主线，但必须先独立验证在 Windows 笔记本环境下是否稳定。

### 6.3 减少 profiling 阶段不必要的频繁写入

当前 wrapper 和训练脚本都会输出大量 CSV / JSON / MD。对于 profiling 或 smoke 阶段，可以优先减少不必要的频繁文件写入，降低 I/O 干扰。

这里强调：

- 这是针对 profiling 或工程验证阶段的建议；
- 正式实验如需保留完整审计材料，应谨慎处理；
- 不应为了提速而削弱正式实验需要保留的核心可追溯信息。

### 6.4 cluster 任务先 capped 后 uncapped

这是当前最重要的工程排程建议之一：

- 先做 `cluster main capped`
- 再做 `cluster ablation capped`
- 最后再判断是否进入 `cluster uncapped`

这样可以先验证 GPU/CPU 在较大样本但可控规模下的行为，再决定是否进入高风险全量任务。

### 6.5 grid_cell ablation 可优先考虑 CPU 或夜间 GPU

从当前结构判断，`grid_cell ablation` 属于“4 个变体串行 + 小 batch + 联邦循环碎片化”的典型任务。它并不是最能体现 GPU 优势的类型。

因此：

- 白天如追求稳定与可控，可优先 CPU；
- 若要统一 GPU 路线，可放夜间运行；
- 不建议把 `grid_cell ablation` 当成判断 cluster GPU 价值的唯一依据。

### 6.6 uncapped cluster 建议夜间运行

`cluster uncapped` 是当前最大风险源。即使使用 GPU，也仍可能是小时级长任务。

建议：

- 放夜间或独立长窗口运行；
- 保持电源、散热和系统休眠策略稳定；
- 不建议白天直接启动全量 uncapped 任务。

### 6.7 每次只改变一个工程参数，并保留日志

这是最重要的执行纪律：

- 每次只改一个工程参数；
- 先做最小 smoke；
- 记录 wall time、显存、是否稳定；
- 保留参数、日志和结论；
- 如果收益不明确，就回退到原始正式口径。

这样才能保证后续正式实验仍具备可解释性和可复现性。

## 7. 不建议直接做的修改

以下改动虽然可能带来表面提速，但会影响实验可比性、论文主线或结果解释，当前不建议直接做：

- 不改标准样本量加权 FedAvg；
- 不为了提速改变 client 划分；
- 不为了提速减少正式实验 `rounds`；
- 不为了提速减少正式实验 `local_epochs`；
- 不为了提速删除 ablation variants；
- 不把 capped 结果冒充 uncapped 正式结果；
- 不把 profiling 结果作为论文正式指标；
- 不为了提速修改模型结构或训练目标；
- 不把工程 smoke 结果写成正式实验结论。

这些边界必须明确保持，否则后续论文主线和实验可比性都会受到影响。

## 8. 推荐运行策略

结合当前已有估算、GPU 基线和静态结构分析，推荐后续正式实验按以下顺序推进：

1. `grid_cell main full`
2. `grid_cell ablation full`
3. `cluster main capped`
4. `cluster ablation capped`
5. `cluster uncapped` 最后再决定，建议夜间运行

具体说明如下：

- `grid_cell` 可 CPU/GPU 都接受，其中 `grid_cell main` 更适合作为正式实验起点；
- `grid_cell ablation` 虽然可以上 GPU，但并不是最能体现 GPU 优势的任务，CPU 也完全可接受；
- `cluster capped` 更值得优先考虑 GPU，因为它比 `grid_cell` 更接近 GPU 擅长的较大样本场景；
- `cluster uncapped` 不建议白天直接跑，无论 CPU 还是 GPU 都应最后再决定；
- 如果后续要精确判断 `cluster GPU` 是否值得投入，最稳妥的做法是先跑一个 `capped cluster smoke`，再决定是否进入正式全量任务。

## 9. 结论

- GPU 不一定总比 CPU 快，尤其在当前这种小模型、小 batch、联邦碎片化循环较多的工程场景下更是如此。
- 当前真实数据实验的主要瓶颈，很可能不只是模型训练本身，还包括 Dataset / DataLoader / 时间窗口构造、CPU/GPU 调度、指标计算和文件 I/O。
- `batch_size = 16` 时，GPU kernel 启动和数据搬运开销占比可能偏高，这会削弱 GPU 的优势。
- CPU 在小模型、小 batch、多 client、多轮次的小碎片任务中可能更稳，甚至部分任务更快。
- GPU 仍然适合样本量更大、训练更连续的任务，尤其是 `cluster` 方向，特别是更接近全量样本的场景。
- `ablation` 比 `main` 更重，是因为它要把 `full / without_attention / without_cnn / without_lstm` 四个变体串行完整跑完。
- `cluster uncapped` 的样本量放大是当前最大耗时风险，应作为最后、最谨慎的任务处理。
- 后续优化应优先保证实验可复现性、论文主线一致性和标准样本量加权 FedAvg 不变，而不是单纯追求 wall time 更短。
