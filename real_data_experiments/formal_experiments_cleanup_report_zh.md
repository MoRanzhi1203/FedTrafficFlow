# 真实数据实验目录清理报告

## 本阶段目标

- 回退本地未推送的 Phase 1 断点续跑提交 `c9bc9f1 feat: add resume checkpoints for single intersection client`。
- 重新收敛 `real_data_experiments/` 目录，仅围绕四个正式实验目录与必要公共依赖保留内容。
- 不运行任何实验，仅执行静态编译与 CLI `--help` 验证。

## Git 回退结论

- 已执行 `git reset --hard origin/main`，本地 `main` 已回退到 `origin/main`。
- 回退后 `HEAD` 为 `c71abe3 docs: design resume checkpoints for real data experiments`。
- 未推送 `c9bc9f1`，未执行 `git push`，未执行 `force push`。
- Phase 1 断点续跑改造未保留；相关提交中的断点续跑实现与阶段性报告未继续保留在当前分支状态中。

## 清理前依赖审计

- 已审计四个正式实验目录的 import 关系：
  - `single_intersection_client`
  - `single_intersection_ablation`
  - `region_client`
  - `region_ablation`
- 审计结果表明，正式实验代码依赖集中在四个正式实验目录本身，以及 `real_data_experiments/common` 公共模块。
- `single_intersection_ablation`、`region_client`、`region_ablation` 还依赖 `single_intersection_client` 中的共享模型与辅助函数，因此该目录必须完整保留。
- `common` 目录中的以下模块为正式实验直接或间接依赖项，因此予以保留：
  - `client.py`
  - `data_splits.py`
  - `fedavg.py`
  - `io_utils.py`
  - `metrics.py`
  - `region_partition.py`
  - `region_tensor_dataset.py`
  - `result_writer.py`
  - `seed.py`
  - `tensor_dataset.py`
  - `trainer.py`

## 清理结果

- 已保留目录：
  - `real_data_experiments/single_intersection_client`
  - `real_data_experiments/single_intersection_ablation`
  - `real_data_experiments/region_client`
  - `real_data_experiments/region_ablation`
  - `real_data_experiments/common`
- 已删除根目录中的工程辅助内容，包括 profiling 脚本、运行说明、阶段计划、环境报告、估算文档、辅助 CSV 与其他非正式实验产物说明文档。
- 已按要求本地删除可能遗留的 ignored smoke 输出目录：
  - `results/real_data_experiments/formal/resume_smoke_sic`
- 未删除正式 Batch 1 结果目录：
  - `results/real_data_experiments/formal/grid_cell_main_full_cuda`

## 验证结果

- 已通过静态编译：
  - `python -m py_compile real_data_experiments\single_intersection_client\sic_core.py`
  - `python -m py_compile real_data_experiments\single_intersection_ablation\sia_core.py`
  - `python -m py_compile real_data_experiments\region_client\rc_core.py`
  - `python -m py_compile real_data_experiments\region_ablation\ra_core.py`
- 已通过 CLI 帮助检查：
  - `python -m real_data_experiments.single_intersection_client.sic_core --help`
  - `python -m real_data_experiments.single_intersection_ablation.sia_core --help`
  - `python -m real_data_experiments.region_client.rc_core --help`
  - `python -m real_data_experiments.region_ablation.ra_core --help`
- 本阶段未运行任何实验，未运行 profiling，未运行 smoke。

## 约束符合性

- 未修改标准样本量加权 FedAvg。
- 未修改模型结构。
- 未修改数据划分。
- 未修改指标计算。
- 未修改 LaTeX。
- 未修改 `simulation_experiments`。
- 未修改 conda 环境。
- `results/` 不纳入提交范围。
- 本阶段未执行 `git add`、`git commit` 或 `git push`。

## 结论

- 本阶段已回退未推送的 Phase 1 断点续跑提交，并完成真实数据实验目录清理。
- 当前 `real_data_experiments/` 已收敛为四个正式实验目录与必要公共依赖，适合进入后续清理确认与提交审查阶段。
