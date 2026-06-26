# 真实数据正式实验运行策略与分批执行边界

## 1. 本阶段范围

本阶段只制定真实数据正式实验运行策略，不运行实验、不运行 profiling、不修改训练代码、不修改论文 LaTeX、不修改 `simulation_experiments`、不修改 conda 环境、不生成新的 `results`。

本阶段只基于已有 smoke、profiling、耗时估算、差异分析和正式运行计划文档，明确：

- 哪些实验可以作为论文正式结果；
- 哪些结果只能作为工程验证；
- 推荐运行顺序、设备选择、输出目录命名、提交边界与中止规则；
- 每批正式实验的统一前置核验与后置收口模板。

## 2. 已有依据

本策略文档基于以下已有资料制定：

- GPU smoke 报告：`real_data_experiments/gpu_cuda_smoke_report_zh.md`
- Step 1 profiling 报告：`real_data_experiments/gpu_light_profile_step1_grid_cell_main_report_zh.md`
- Step 2 profiling 报告：`real_data_experiments/gpu_light_profile_step2_grid_cell_ablation_report_zh.md`
- 轻量 GPU profiling 方案：`real_data_experiments/gpu_light_profiling_plan_zh.md`
- CPU/GPU 耗时估算报告：`real_data_experiments/real_data_cpu_gpu_time_estimate_zh.md`
- CPU/GPU 耗时差异分析文档：`real_data_experiments/cpu_gpu_time_gap_analysis_zh.md`
- 历史 CPU 侧计算量与耗时估算文档：`real_data_experiments/compute_time_estimation_i7_3060_zh.md`
- Tensor-only 正式运行计划：`real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md`
- Tensor-only 配置冻结方案：`real_data_experiments/tensor_only_experiment_plan_zh.md`
- 总体真实实验运行说明：`real_data_experiments/RUN_REAL_EXPERIMENTS_zh.md`

当前静态确认到的工程状态如下：

- CUDA 环境已配置完成；
- GPU smoke 已通过；
- `grid_cell main r3e1` 与 `grid_cell ablation r2e1` 的 GPU 轻量 profiling 已完成；
- CPU/GPU 耗时估算与差异分析文档已生成并提交；
- `main` 当前应与 `origin/main` 同步；
- 工作区当前应保持干净，正式实验尚未启动。

## 3. 实验类型划分

当前真实数据正式实验分为以下五类：

### 3.1 grid_cell main

- 代码入口：`python -m real_data_experiments.single_intersection_client.sic_core --workflow all`
- 可视化入口：`python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all`
- 语义：网格单元级客户端联邦学习设置主实验
- 正式常用参数：
  - `--device`
  - `--selected-clients`
  - `--output-dir`
  - `--num-clients`
  - `--rounds`
  - `--local-epochs`
  - `--batch-size`

### 3.2 grid_cell ablation

- 代码入口：`python -m real_data_experiments.single_intersection_ablation.sia_core --workflow all`
- 可视化入口：`python -m real_data_experiments.single_intersection_ablation.sia_visualization --workflow all`
- 语义：网格单元级客户端联邦学习设置消融实验
- 正式常用参数：
  - `--device`
  - `--selected-clients`
  - `--variants`
  - `--output-dir`
  - `--num-clients`
  - `--rounds`
  - `--local-epochs`
  - `--batch-size`

### 3.3 cluster main capped

- 代码入口：`python -m real_data_experiments.region_client.rc_core --workflow all`
- 可视化入口：`python -m real_data_experiments.region_client.rc_visualization --workflow all`
- 语义：簇级客户端联邦学习设置主实验，保留样本上限 cap
- 正式常用参数：
  - `--device`
  - `--partition-method`
  - `--max-samples-per-client-split`
  - `--output-dir`
  - `--num-clients`
  - `--rounds`
  - `--local-epochs`
  - `--batch-size`

### 3.4 cluster ablation capped

- 代码入口：`python -m real_data_experiments.region_ablation.ra_core --workflow all`
- 可视化入口：`python -m real_data_experiments.region_ablation.ra_visualization --workflow all`
- 语义：簇级客户端联邦学习设置消融实验，保留样本上限 cap
- 正式常用参数：
  - `--device`
  - `--partition-method`
  - `--variants`
  - `--max-samples-per-client-split`
  - `--output-dir`
  - `--num-clients`
  - `--rounds`
  - `--local-epochs`
  - `--batch-size`

### 3.5 cluster uncapped

- 代码入口仍为：
  - `python -m real_data_experiments.region_client.rc_core --workflow all`
  - `python -m real_data_experiments.region_ablation.ra_core --workflow all`
- 与 capped 的唯一区别不是入口变化，而是：
  - 不传 `--max-samples-per-client-split`
  - 或显式使其为 `0 / None`
- 当前风险判断：
  - `cluster uncapped` 是最高耗时、最高调度风险、最高夜间运行风险任务；
  - 在作者明确批准前，不直接进入正式运行。

## 4. 正式结果与非正式结果边界

### 4.1 可以作为论文正式结果的内容

以下内容只有在“按冻结配置完整运行”后，才可以进入论文正式结果链：

- 按冻结配置完整运行的 `grid_cell main full`
- 按冻结配置完整运行的 `grid_cell ablation full`
- 按冻结配置完整运行的 `cluster main capped` 或 `cluster main uncapped`
- 按冻结配置完整运行的 `cluster ablation capped` 或 `cluster ablation uncapped`

补充说明：

- `cluster` 最终采用 capped 还是 uncapped，应由作者最终选择决定；
- 若作者最终选择 capped，则 capped 才能作为正式结果；
- 若作者最终选择 uncapped，则 capped 只能作为工程验证或前置可行性依据，不能冒充 uncapped 正式结果。

### 4.2 不能作为论文正式结果的内容

以下内容只能作为工程验证、排程依据或运行风控材料，不能进入论文正文正式结果表：

- smoke
- profiling
- 耗时估算
- CPU/GPU 差异分析
- capped smoke
- 中途失败或中止的实验
- 未按冻结配置运行的临时实验
- 任意为工程验证而修改 batch size、worker、pin memory 后的临时运行

### 4.3 必须保持的边界

- profiling 结果不得进入论文正文表格；
- smoke 结果不得作为论文正式指标；
- 估算结果只用于工程排程；
- 不得把 capped 结果冒充 uncapped 结果；
- 不得改变标准样本量加权 FedAvg 主线；
- 不得因提速需要而把工程验证运行写成正式实验结果。

## 5. 推荐运行顺序

推荐遵循以下固定原则：

1. 先 `grid_cell`，再 `cluster`
2. 先 `main`，再 `ablation`
3. 先 `capped`，再 `uncapped`
4. `cluster uncapped` 不直接白天运行
5. 每一批只运行一个实验组
6. 每一批运行前必须 `git clean`
7. 每一批运行后先检查 `results` 是否被忽略
8. 每一批都生成独立中文报告
9. 每一批报告单独提交
10. `results/` 不提交

| 批次 | 实验 | 建议设备 | 是否 cap | 是否可作为正式结果 | 预计耗时 | 风险 |
|---|---|---|---|---|---|---|
| Batch 1 | `grid_cell main full` | CPU/GPU 都可，偏向 GPU | 否 | 是 | CPU 约 `12.62 min`；GPU 约 `11.08 min`，保守约 `7.75 - 16.62 min` | 中低风险；GPU 收益可能有限但已有基线 |
| Batch 2 | `grid_cell ablation full` | CPU 或夜间 GPU | 否 | 是 | CPU 约 `17.80 min`；GPU 约 `41.02 min`，保守约 `28.71 - 61.53 min` | 中等风险；4 个 variants 串行导致增重明显 |
| Batch 3 | `cluster main capped` | 建议 GPU | 是 | 是，可作为 capped 正式结果 | GPU 约 `2.44 - 4.87 min`；CPU 约 `4.87 min` | 中等风险；缺少 GPU 正式实测但 cap 可控 |
| Batch 4 | `cluster ablation capped` | 建议 GPU，必要时夜间 | 是 | 是，可作为 capped 正式结果 | GPU 约 `4.70 - 11.27 min`；CPU 约 `9.39 min` | 中高风险；4 个 variants 串行且 cluster 结构更重 |
| Batch 5 | `cluster uncapped feasibility review` | 不运行正式训练，只做作者复核 | 不适用 | 否 | 不执行训练，仅复核是否进入 Batch 6 | 高风险；重点复核样本量、夜间窗口、显存和作者批准 |
| Batch 6 | `cluster uncapped full` | 优先 GPU，最好夜间 | 否 | 仅在作者明确批准后才可作为正式结果 | `cluster main uncapped` GPU 约 `3.01 - 7.23 h`；`cluster ablation uncapped` GPU 约 `5.81 - 13.93 h`；CPU 分别约 `12.05 h` 与 `23.22 h` | 最高风险；样本量放大约 `148.34x`，6GB 显存和 I/O 都需谨慎 |

执行口径说明：

- 如果作者最终决定论文主线使用 capped cluster，则 Batch 3 和 Batch 4 可以进入正式结果链，Batch 6 不进入。
- 如果作者最终决定必须给出 uncapped cluster 正式结果，则 Batch 3 和 Batch 4 应被视为前置验证或阶段性正式参考，Batch 6 才是最终正式结果来源。
- Batch 5 不是训练批次，而是进入 `uncapped` 前的可行性复核关卡。

## 6. 每批运行前核验

每批正式实验启动前，统一执行以下模板命令：

```powershell
git status --short
git status -sb
git log -5 --oneline

(& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate E:\anaconda3\envs\FedTrafficFlow

python -c "import sys, os, site, torch; print('executable=', sys.executable); print('prefix=', sys.prefix); print('PYTHONNOUSERSITE=', os.environ.get('PYTHONNOUSERSITE')); print('ENABLE_USER_SITE=', site.ENABLE_USER_SITE); print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device_name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
nvidia-smi
```

### 中止规则

出现以下任一情况，立即停止，不自动扩大规模、不自动切换任务：

- `git status` 不干净，停止；
- 计划用 GPU，但 `cuda_available = False`，停止；
- 显存异常占用，停止；
- `results` 进入 `git status`，停止并先修 `.gitignore`；
- 脚本报错，停止；
- CUDA OOM，停止；
- 耗时明显异常，停止；
- 不得自动扩大 `clients / rounds / local_epochs / batch_size`；
- 不得自动切换到 `uncapped`；
- 不得自动切换正式论文实验配置；
- 不得自行把验证任务升级为正式结果任务。

## 7. 每批运行后收口

每批正式实验完成后，统一执行以下收口步骤：

1. 检查 `results` 是否已被 `.gitignore` 覆盖，确认不进入 Git。
2. 读取输出目录中的关键摘要文件，例如：
   - `run_config.json`
   - `environment_summary.json`
   - `split_summary.json`
   - `main_summary.csv/json`
   - `ablation_summary.csv`
   - `client_metrics.csv/json`
   - `convergence_history.csv/json`
   - `figure_index.csv`
   - `figure_notes_zh.md`
3. 基于已有输出生成单独中文运行报告。
4. 只提交报告和必要文档，不提交 `results/`。
5. 不提交大文件，不提交中间模型权重，不提交本地临时图像缓存。
6. 每一批报告单独提交，不与下一批混提。

### 当前 `.gitignore` 静态结论

当前 `.gitignore` 已覆盖 `results/real_data_experiments/` 下的主要真实数据实验输出，包括：

- `results/real_data_experiments/gpu_light_profile/`
- `results/real_data_experiments/*_smoke/`
- `results/real_data_experiments/**/*_smoke/`
- `results/real_data_experiments/single_intersection_client_tensor/`
- `results/real_data_experiments/single_intersection_ablation_tensor/`
- `results/real_data_experiments/single_region_client_tensor_*/`
- `results/real_data_experiments/single_region_ablation_tensor_*/`
- `results/real_data_experiments/region_client_tensor_*/`
- `results/real_data_experiments/region_ablation_tensor_*/`
- `results/real_data_experiments/**/*.png`
- `results/real_data_experiments/**/*.pdf`
- `results/real_data_experiments/**/prediction_samples.csv`
- `results/real_data_experiments/**/convergence_history.csv`

因此，按照当前推荐目录命名运行，正式实验输出原则上应被忽略；但每批次运行后仍必须再核验一次 `git status`。

## 8. 设备选择建议

结合已有 GPU smoke、Step 1/Step 2 profiling、CPU/GPU 耗时估算与差异分析，当前建议如下：

- `grid_cell main full`：CPU/GPU 都可，GPU 可作为优先，但收益可能不大；
- `grid_cell ablation full`：CPU 可能更稳，GPU 不一定明显更快；
- `cluster main capped`：建议 GPU；
- `cluster ablation capped`：建议 GPU，但必须保留 cap；
- `cluster uncapped`：不建议直接运行，需单独夜间计划。

更细化说明如下：

### 8.1 grid_cell main full

- 当前已有 `grid_cell main r3e1` GPU 实测基线；
- GPU 预计略优于 CPU，但不是数量级加速；
- 若白天先跑正式主实验，GPU 可以作为优先设备；
- 若更强调稳定和操作简单，CPU 也可接受。

### 8.2 grid_cell ablation full

- `ablation` 包含 `full / without_attention / without_cnn / without_lstm` 四个变体；
- 任务结构是典型的小 batch、碎片化联邦循环；
- 当前分析口径下，CPU 可能更稳，GPU 不一定明显更快；
- 因此可优先 CPU，或安排到夜间 GPU。

### 8.3 cluster capped

- capped 后样本规模可控；
- 相比 `grid_cell`，`cluster` 更接近 GPU 擅长的较大样本任务；
- 但当前缺少 `cluster` 的 GPU 实测；
- 因此建议 GPU 优先，但必须保留 cap，不自动放大到 uncapped。

### 8.4 cluster uncapped

- 这是当前全链路风险最高任务；
- 样本量相对 capped 放大约 `148.34x`；
- `RTX 3060 Laptop GPU 6GB` 对 `uncapped cluster` 仍需谨慎；
- 不建议白天直接运行，不建议在没有作者明确批准前进入正式执行。

## 9. 不允许为提速改变的内容

以下内容不得为提速而擅自改变：

- 不改变标准样本量加权 FedAvg；
- 不改变 client 划分；
- 不减少正式 `rounds`；
- 不减少正式 `local_epochs`；
- 不删除 ablation variants；
- 不把 capped 结果写成 uncapped；
- 不把 profiling / smoke / 估算写成正式指标；
- 不为提速而改训练代码主逻辑；
- 不自动从正式配置切换到“更轻量的临时配置”。

这些边界必须保持，否则后续正式结果将失去可比性和论文解释一致性。

## 10. 下一步建议

建议先执行 `Batch 1: grid_cell main full` 的正式运行准备。

在真正执行前，先单独生成 `Batch 1: grid_cell main full` 的精确 Trae 指令，要求包含：

- 前置 Git 核验；
- Conda 与 CUDA 环境核验；
- 目标输出目录冻结；
- 运行后摘要读取；
- 结果不入 Git 的检查；
- 单独中文报告生成边界；
- 中止规则与异常收口动作。

当前不建议直接跳到 `cluster uncapped`，也不建议在未生成 Batch 1 精确执行指令前直接启动正式训练。
