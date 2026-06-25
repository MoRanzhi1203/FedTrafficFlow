# 轻量 GPU Profiling 方案与运行边界

## 1. 本阶段范围

本阶段只做方案设计，不运行 profiling、不运行正式训练、不修改训练代码、不修改论文正文。

本阶段也不修改 `simulation_experiments`、不修改 FedAvg 主线、不修改 conda 环境，仅冻结轻量 GPU profiling 的运行边界、任务组合、输出目录、显存观察项和风险控制。

## 2. 当前 CUDA 环境状态

- Conda 环境路径：`E:\anaconda3\envs\FedTrafficFlow`
- Python 解释器：`E:\anaconda3\envs\FedTrafficFlow\python.exe`
- torch 版本：`2.8.0+cu126`
- `torch.version.cuda`：`12.6`
- `cuda_available`：`True`
- GPU device name：`NVIDIA GeForce RTX 3060 Laptop GPU`
- `PYTHONNOUSERSITE`：`1`
- `ENABLE_USER_SITE`：`False`

补充说明：

- 已执行 `profile_tensor_experiments.py --help`，确认参数存在。
- 当前脚本实际支持的 `--setting` 取值为 `grid_cell` 和 `cluster`，不支持 `region`。
- 当前脚本支持 `--max-samples-per-client-split`，因此 cluster-level 任务可按计划加 cap。

## 3. profiling 结果使用边界

- 仅用于工程耗时、显存和运行链路评估。
- 不作为论文正式结果。
- 不用于正文表格。
- 不用于审稿回复中的正式指标。
- 不改变标准 FedAvg 主线。
- 不应把 smoke/profiling 的阶段性输出解释为正式实验结论。

## 4. 输出目录约束

统一输出到：

`results/real_data_experiments/gpu_light_profile/`

约束说明：

- 该目录应被 `.gitignore` 忽略。
- 不提交 `results/`。
- 所有轻量 profiling 任务都应写入该目录下的独立子目录，避免与 smoke 输出混杂。

## 5. 任务组合设计

### 5.1 grid-cell-level main quick profiling

- setting：`grid_cell`
- task：`main`
- device：`cuda`
- num_clients：`3`
- rounds：`3`
- local_epochs：`1`
- batch_size：`16`
- sequence_length：`12`
- `max_samples_per_client_split`：不使用
- output_dir：`results/real_data_experiments/gpu_light_profile/grid_cell_main_r3e1`

### 5.2 grid-cell-level ablation quick profiling

- setting：`grid_cell`
- task：`ablation`
- device：`cuda`
- num_clients：`3`
- rounds：`2`
- local_epochs：`1`
- batch_size：`16`
- sequence_length：`12`
- `max_samples_per_client_split`：不使用
- output_dir：`results/real_data_experiments/gpu_light_profile/grid_cell_ablation_r2e1`

### 5.3 cluster-level main capped profiling

- setting：`cluster`
- task：`main`
- device：`cuda`
- num_clients：`3`
- rounds：`2`
- local_epochs：`1`
- batch_size：`16`
- sequence_length：`12`
- `max_samples_per_client_split`：`1024`
- output_dir：`results/real_data_experiments/gpu_light_profile/cluster_main_r2e1_cap1024`

### 5.4 cluster-level ablation capped profiling

- setting：`cluster`
- task：`ablation`
- device：`cuda`
- num_clients：`3`
- rounds：`1`
- local_epochs：`1`
- batch_size：`16`
- sequence_length：`12`
- `max_samples_per_client_split`：`1024`
- output_dir：`results/real_data_experiments/gpu_light_profile/cluster_ablation_r1e1_cap1024`

## 6. 命令草案

以下命令草案仅用于后续执行阶段参考，本阶段不执行。

### 6.1 Grid-cell-level main quick profiling

```powershell
python real_data_experiments/profile_tensor_experiments.py --setting grid_cell --task main --device cuda --num-clients 3 --rounds 3 --local-epochs 1 --batch-size 16 --sequence-length 12 --output-dir results/real_data_experiments/gpu_light_profile/grid_cell_main_r3e1
```

### 6.2 Grid-cell-level ablation quick profiling

```powershell
python real_data_experiments/profile_tensor_experiments.py --setting grid_cell --task ablation --device cuda --num-clients 3 --rounds 2 --local-epochs 1 --batch-size 16 --sequence-length 12 --output-dir results/real_data_experiments/gpu_light_profile/grid_cell_ablation_r2e1
```

### 6.3 Cluster-level main capped profiling

```powershell
python real_data_experiments/profile_tensor_experiments.py --setting cluster --task main --device cuda --num-clients 3 --rounds 2 --local-epochs 1 --batch-size 16 --sequence-length 12 --max-samples-per-client-split 1024 --output-dir results/real_data_experiments/gpu_light_profile/cluster_main_r2e1_cap1024
```

### 6.4 Cluster-level ablation capped profiling

```powershell
python real_data_experiments/profile_tensor_experiments.py --setting cluster --task ablation --device cuda --num-clients 3 --rounds 1 --local-epochs 1 --batch-size 16 --sequence-length 12 --max-samples-per-client-split 1024 --output-dir results/real_data_experiments/gpu_light_profile/cluster_ablation_r1e1_cap1024
```

命令修正说明：

- 用户草案中的 `--setting region` 已根据脚本实际 `--help` 修正为 `--setting cluster`。
- 修正原因是脚本只接受 `grid_cell` 与 `cluster` 两种 setting，`region` 不在允许值中。

## 7. 显存与中止规则

- 每次任务前后执行 `nvidia-smi`。
- CUDA OOM 立即停止。
- 任一任务耗时明显异常，立即停止。
- 任一任务产生非预期代码修改或数据文件修改，立即停止。
- 任一任务 `results/` 未被 `.gitignore` 忽略，立即停止并报告。
- 不自动扩大 `rounds`、`clients`、`batch_size`。
- region / cluster-level 必须设置 `--max-samples-per-client-split`。
- batch size 从 `16` 起步，不先上调。
- `rounds` 不超过 `3`。
- `local_epochs` 不超过 `1`。
- 执行顺序固定为：
  - Step 1: `grid_cell main r3e1`
  - Step 2: `grid_cell ablation r2e1`
  - Step 3: `cluster main r2e1 cap1024`
  - Step 4: `cluster ablation r1e1 cap1024`
- 不把 smoke/profiling 写成论文正式结果。

## 8. 是否可以进入轻量 GPU profiling 执行阶段

结论：

- 可以进入轻量 GPU profiling 执行阶段。

原因：

- 当前仓库工作区干净，`main` 与 `origin/main` 同步。
- 当前 CUDA 环境状态正常，`torch=2.8.0+cu126`、`torch.version.cuda=12.6`、`cuda_available=True`。
- 最小 GPU smoke 已通过，说明当前 GPU 运行链路可用。
- 脚本 `--help` 已确认存在 `--setting`、`--task`、`--device`、`--num-clients`、`--rounds`、`--local-epochs`、`--batch-size`、`--sequence-length`、`--output-dir`、`--max-samples-per-client-split` 等关键参数。

执行前提：

- 后续执行阶段仍需坚持“先 grid_cell、后 cluster；先 main、后 ablation”的顺序。
- 任一任务异常即中止，不自动扩容。
- 所有结果继续只作为工程验证，不进入论文正式证据链。
