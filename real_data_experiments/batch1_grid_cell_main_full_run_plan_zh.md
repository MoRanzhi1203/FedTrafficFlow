# Batch 1：grid_cell main full 正式实验运行方案

## 1. 本阶段范围

本阶段只静态确认 Batch 1 正式运行方案，不运行 Batch 1，不运行任何训练，不运行 profiling，不运行 Step 3/4，不修改训练代码，不修改 LaTeX，不修改 `simulation_experiments`，不修改 conda 环境，不生成新的 `results`。

本阶段只基于已有策略文档、正式运行计划文档、代码入口和 `--help` 信息，确认：

- Batch 1 的正式命令来源；
- Batch 1 的实验类型与结果边界；
- Batch 1 的冻结参数；
- Batch 1 的输出目录与 Git 风险；
- Batch 1 的运行前核验模板与中止规则。

## 2. Batch 1 实验定义

- 实验类型：`grid_cell main full`
- 中文名称：网格单元级客户端真实数据主实验
- 代码目录：`single_intersection_client`
- 入口类型：`single_intersection_client / grid_cell main`
- 数据来源：tensor-only 正式数据入口
- 是否正式实验候选：是，前提是按冻结配置完整运行
- 是否 profiling / smoke：否
- 是否 ablation：否
- 是否 cluster：否

本次静态确认的边界结论：

1. Batch 1 不是 profiling；
2. Batch 1 不是 smoke；
3. Batch 1 不是 ablation；
4. Batch 1 不是 cluster；
5. Batch 1 不改变标准样本量加权 FedAvg 主线；
6. Batch 1 不重新生成 tensor 数据；
7. Batch 1 不修改 LaTeX。

## 3. 依据文档

本次静态确认读取了以下依据文档与代码文件：

- `real_data_experiments/real_data_formal_run_strategy_zh.md`
- `real_data_experiments/real_data_cpu_gpu_time_estimate_zh.md`
- `real_data_experiments/cpu_gpu_time_gap_analysis_zh.md`
- `real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md`
- `real_data_experiments/tensor_only_experiment_plan_zh.md`
- `real_data_experiments/RUN_REAL_EXPERIMENTS_zh.md`
- `real_data_experiments/single_intersection_client/README_zh.md`
- `real_data_experiments/single_intersection_client/sic_config.py`
- `real_data_experiments/single_intersection_client/sic_core.py`
- `real_data_experiments/single_intersection_client/sic_visualization.py`
- `real_data_experiments/profile_tensor_experiments.py --help`
- `.gitignore`

## 4. 正式命令来源

### 4.1 推荐正式命令来源

Batch 1 的正式命令应优先引用：

- `real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md`

原因：

- 该文档明确给出了“网格单元级客户端设置主论文正式实验”的完整命令；
- 命令已经包含 `--data-mode tensor`、`--selected-clients` 和独立 `--output-dir`；
- 该命令与 `tensor_only_experiment_plan_zh.md` 中的主论文正式实验方案 B 对齐；
- 该命令不会复用默认 smoke 输出目录。

### 4.2 代码层正式入口

代码层正式训练入口为：

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all
```

配套可视化入口为：

```bash
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all
```

静态代码确认：

- `sic_core.py` 负责正式 tensor 读取、active region 选择、时间顺序划分、标准样本量加权 FedAvg / Independent 训练与结果导出；
- `sic_visualization.py` 只读取已有 CSV 结果并生成图表，不重新训练；
- `sic_core.py` 中 `run_experiment()` 会写出 `run_config.json`、`main_summary.csv/json`、`client_metrics.csv/json`、`convergence_history.csv/json`、`prediction_samples.csv/json`、`experiment_notes_zh.md` 等结果文件；
- `sic_visualization.py` 会额外写出图表文件、`figure_index.csv` 和 `figure_notes_zh.md`。

### 4.3 与 profiling wrapper 的关系

`profile_tensor_experiments.py` 不是 Batch 1 的正式入口，而是 profiling wrapper。

静态代码表明：

- 当 `setting=grid_cell` 且 `task=main` 时，wrapper 会构造模块命令并调用 `real_data_experiments.single_intersection_client.sic_core`；
- 但 wrapper 的语义是 profiling，默认输出目录也是 `results/real_data_experiments/compute_time_profile` 体系；
- 因此 Batch 1 正式实验不应通过 `profile_tensor_experiments.py` 发起，而应直接使用 RUN 文档中的正式命令。

### 4.4 其他可能来源

`real_data_experiments/single_intersection_client/README_zh.md` 中也给出了简化示例：

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all
```

但该 README 示例没有冻结 `selected_clients` 和独立 `output_dir`，因此更适合作为模块说明，不应优先于正式 RUN 文档。

## 5. 拟执行命令草案

以下命令草案基于实际脚本支持参数整理，本阶段不执行。

### 5.1 RUN 文档中的正式主命令

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --num-clients 5 --selected-clients 290,284,318,288,289 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/single_region_client_tensor_main
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all --input-dir results/real_data_experiments/single_region_client_tensor_main --dpi 300
```

### 5.2 设备项的待确认说明

当前文档来源对 `device` 存在口径差异：

- `RUN_TENSOR_ONLY_EXPERIMENTS_zh.md` 的正式主命令写的是 `--device cpu`；
- `tensor_only_experiment_plan_zh.md` 的主论文正式实验方案 B 也与 CPU 正式命令一致；
- `real_data_formal_run_strategy_zh.md` 的 Batch 1 设备建议是“CPU/GPU 都可，偏向 GPU”；
- `real_data_cpu_gpu_time_estimate_zh.md` 则从工程排程角度建议 `grid_cell main full` 可优先用 GPU。

因此，本阶段不擅自改写为 `--device cuda`。若作者下一阶段决定按 GPU 执行，应在正式执行指令中显式确认：

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --num-clients 5 --selected-clients 290,284,318,288,289 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cuda --output-dir results/real_data_experiments/single_region_client_tensor_main
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all --input-dir results/real_data_experiments/single_region_client_tensor_main --dpi 300
```

上面这组 GPU 命令只是基于已支持参数给出的待确认草案，不是本阶段决定。

## 6. 冻结参数

### 6.1 当前可直接冻结的参数

- `setting = grid_cell`
- `task = main`
- `experiment family = single_intersection_client`
- `data-mode = tensor`
- `num_clients = 5`
- `selected_clients = 290,284,318,288,289`
- `rounds = 20`
- `local_epochs = 3`
- `batch_size = 32`
- `sequence_length = 12`
- `prediction_horizon = 1`
- `learning_rate = 0.001`
- `seed = 42`
- `input tensor path = data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- `regions_path = data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`
- `recommended output_dir = results/real_data_experiments/single_region_client_tensor_main`
- `workflow = all`

### 6.2 默认值与文档值的关系

静态确认结果如下：

- `sic_config.py` 的代码默认值并不等于正式运行值；
- 代码默认值为：
  - `batch_size = 64`
  - `local_epochs = 2`
  - `communication_rounds = 3`
  - `device = auto`
  - `output_dir = results/real_data_experiments/single_intersection_client_tensor`
- 正式运行文档冻结值为：
  - `batch_size = 32`
  - `local_epochs = 3`
  - `rounds = 20`
  - `selected_clients = 290,284,318,288,289`
  - `output_dir = results/real_data_experiments/single_region_client_tensor_main`

因此，Batch 1 不应依赖代码默认值，应显式带完整正式参数。

### 6.3 待作者确认项

当前 Batch 1 至少有以下待确认项：

1. `device`
   - 当前正式 RUN 文档为 `cpu`
   - 当前最新策略文档建议“CPU/GPU 都可，偏向 GPU”
   - 需作者明确下一阶段到底按 `cpu` 还是 `cuda` 执行
2. `output_dir`
   - 当前 RUN 文档推荐 `results/real_data_experiments/single_region_client_tensor_main`
   - 用户消息中举例的 `results/real_data_experiments/formal/grid_cell_main_full` 并非当前项目已有正式目录命名
   - 若作者要改成新的 `formal/grid_cell_main_full` 风格，应先单独确认 `.gitignore` 风险
3. `是否在 Batch 1 同步执行 visualization`
   - RUN 文档将训练与可视化作为一组正式命令给出
   - 因此当前更推荐“训练完成后立即执行 visualization”
   - 但若作者希望把训练与可视化分成两个阶段，也应在执行前明确

## 7. 预计耗时

当前 Batch 1 可引用的已有估算如下：

- CPU 估算：约 `12.62 min`
- GPU 估算：约 `11.08 min`
- GPU 保守范围：约 `7.75 - 16.62 min`

建议预留：

- 若按 CPU 执行：至少预留 `15 - 20 min`
- 若按 GPU 执行：至少预留 `15 - 20 min`

说明：

- 以上估算仅用于工程排程，不是正式结果；
- 以上估算不是论文正式指标；
- GPU 对 Batch 1 的收益预计存在，但并非数量级提升；
- 若 Windows 后台显存占用或 I/O 波动明显，GPU wall time 仍可能偏离中心估算值。

## 8. 输出与提交边界

Batch 1 的输出与 Git 边界应保持如下纪律：

- `results/` 不提交；
- 只提交 Batch 1 报告和必要索引文档；
- 不提交 `.pt` / `.npy` / `parquet` / `png` / `pdf` 大文件；
- 不提交模型权重；
- 若 `results` 进入 `git status`，先停止并单独处理 `.gitignore`；
- 不将 smoke / profiling / 估算结果写成 Batch 1 正式结果。

### 8.1 当前输出目录判断

当前项目中有三个与 Batch 1 相关的目录层级：

1. 代码默认目录：
   - `results/real_data_experiments/single_intersection_client_tensor`
   - 不推荐用于 Batch 1，因为可能与历史 smoke 或默认输出混写
2. RUN 文档推荐正式目录：
   - `results/real_data_experiments/single_region_client_tensor_main`
   - 这是当前最稳妥的正式 Batch 1 候选目录
3. 用户消息示例目录：
   - `results/real_data_experiments/formal/grid_cell_main_full`
   - 当前项目文档体系尚未正式采用这一命名

### 8.2 当前 `.gitignore` 风险判断

静态检查 `.gitignore` 后可确认：

- `results/real_data_experiments/single_region_client_tensor_*/` 已被忽略；
- `results/real_data_experiments/single_intersection_client_tensor/` 已被忽略；
- `results/real_data_experiments/**/*.png`、`**/*.pdf`、`**/prediction_samples.csv`、`**/convergence_history.csv` 也已被忽略。

因此：

- 如果 Batch 1 使用 `results/real_data_experiments/single_region_client_tensor_main`，当前 Git 风险较低；
- 如果改用 `results/real_data_experiments/formal/grid_cell_main_full`，当前 `.gitignore` 未静态确认覆盖该路径，存在结果进入 Git 状态的风险；
- 本阶段不修改 `.gitignore`，只记录该风险。

## 9. 运行前核验模板

```powershell
git status --short
git status -sb
git log -5 --oneline

(& conda 'shell.powershell' 'hook') | Out-String | Invoke-Expression
conda activate E:\anaconda3\envs\FedTrafficFlow

python -c "import sys, os, site, torch; print('executable=', sys.executable); print('prefix=', sys.prefix); print('PYTHONNOUSERSITE=', os.environ.get('PYTHONNOUSERSITE')); print('ENABLE_USER_SITE=', site.ENABLE_USER_SITE); print('torch=', torch.__version__); print('torch_cuda=', torch.version.cuda); print('cuda_available=', torch.cuda.is_available()); print('device_name=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
nvidia-smi
```

## 10. 中止规则

Batch 1 下一阶段若开始正式执行，必须遵守以下中止规则：

- `git status` 不干净，停止；
- 计划用 GPU，但 `cuda_available = False`，停止；
- 显存异常占用，停止；
- 脚本报错，停止；
- CUDA OOM，停止；
- `results` 进入 `git status`，停止；
- 不自动扩大规模；
- 不自动改 `rounds / local_epochs / clients / batch_size`；
- 不自动切换 `CPU / GPU`；
- 不自动运行 Batch 2；
- 不自动改写正式输出目录命名；
- 不得把未按冻结配置运行的结果写成正式 Batch 1 结果。

## 11. 是否可以进入 Batch 1 执行阶段

当前结论：

- Batch 1 的正式命令来源、实验类型、主入口、主要参数、默认 tensor 路径、推荐输出目录和 Git 风险点都已静态确认；
- Batch 1 不是 profiling，不是 smoke，不是 ablation，不是 cluster，也不改变标准样本量加权 FedAvg 主线；
- 从工程准备角度看，已经可以进入“等待作者确认后执行”的阶段；
- 但从严格配置冻结角度看，`device` 仍存在 CPU/GPU 口径差异，建议在真正执行前由作者做最终确认。

因此，本文件给出的结论是：

**暂不建议直接执行 Batch 1，原因是正式命令来源对 `device` 仍存在 CPU/GPU 口径差异；建议作者先确认 `cpu` 还是 `cuda`，再进入下一阶段正式执行。**
