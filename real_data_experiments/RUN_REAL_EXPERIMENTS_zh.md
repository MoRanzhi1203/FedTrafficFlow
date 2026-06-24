# 真实实验运行说明

## 环境准备

- 推荐先安装项目依赖：

```bash
pip install -r requirements.txt
pip install torch matplotlib
```

## 正式数据入口

- 当前正式 tensor-only 输入：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 配套 sidecar：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor_metadata.json`
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_metadata.json`
- 正式 `tensor shape = (2, 630, 5856)`
- 正式 `pool_mode = sum_mean`
- 正式 `layout = standard`
- 当前 `client = pooled-grid-region client`
- `parquet-direct = legacy fallback only`

## 命名说明

- 历史名称“单路口客户端”在 tensor-only 阶段实际表示“单池化网格区域客户端”。
- 论文、报告和图表中建议统一采用“单池化网格区域客户端”或 `single pooled-grid-region client`。
- 代码目录 `single_intersection_client` 和 `single_intersection_ablation` 暂不修改。
- `region_client` / `region_ablation` 表示“区域网格客户端”，每个 client = 一组 pooled grid regions。

## 当前可运行模块

### 单池化网格区域客户端主实验

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all
```

### 单池化网格区域客户端消融实验

```bash
python -m real_data_experiments.single_intersection_ablation.sia_core --workflow all
python -m real_data_experiments.single_intersection_ablation.sia_visualization --workflow all
```

### 区域网格客户端主实验

```bash
python -m real_data_experiments.region_client.rc_core --workflow all
python -m real_data_experiments.region_client.rc_visualization --workflow all
```

### 区域网格客户端消融实验

```bash
python -m real_data_experiments.region_ablation.ra_core --workflow all
python -m real_data_experiments.region_ablation.ra_visualization --workflow all
```

### 快速 smoke test

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --num-clients 2 --rounds 1 --local-epochs 1 --batch-size 32 --sequence-length 12 --device cpu
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all --input-dir results/real_data_experiments/single_intersection_client_tensor
python -m real_data_experiments.single_intersection_ablation.sia_core --workflow all --data-mode tensor --num-clients 2 --rounds 1 --local-epochs 1 --batch-size 32 --sequence-length 12 --device cpu
python -m real_data_experiments.single_intersection_ablation.sia_visualization --workflow all --input-dir results/real_data_experiments/single_intersection_ablation_tensor
python -m real_data_experiments.region_client.rc_core --workflow all --data-mode tensor --partition-method spatial_block --num-clients 3 --rounds 1 --local-epochs 1 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/region_client_tensor_smoke
python -m real_data_experiments.region_client.rc_visualization --workflow all --input-dir results/real_data_experiments/region_client_tensor_smoke --dpi 150
python -m real_data_experiments.region_ablation.ra_core --workflow all --data-mode tensor --partition-method spatial_block --num-clients 3 --rounds 1 --local-epochs 1 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/region_ablation_tensor_smoke
python -m real_data_experiments.region_ablation.ra_visualization --workflow all --input-dir results/real_data_experiments/region_ablation_tensor_smoke --dpi 150
```

- 若只做 agent / CI 级 smoke test，可在 `rc_core.py` 与 `ra_core.py` 命令后追加：
  `--max-samples-per-client-split 1024`

## 正式运行计划入口

- 配置冻结方案见 `real_data_experiments/tensor_only_experiment_plan_zh.md`
- 正式命令计划见 `real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md`
- 固定 region 清单见 `real_data_experiments/selected_regions_fixed_plan.csv`

## 固定 region 建议

- 推荐正式 top-3 regions：
  `290, 284, 318`
- 推荐正式 top-5 regions：
  `290, 284, 318, 288, 289`
- 若需要固定客户端，可通过 `--selected-clients` 传入逗号分隔的 `region_id` 列表。

## 输出目录

- 当前 smoke test 默认结果目录：
  `results/real_data_experiments/single_intersection_client_tensor/`
  `results/real_data_experiments/single_intersection_ablation_tensor/`
  `results/real_data_experiments/region_client_tensor_smoke/`
  `results/real_data_experiments/region_ablation_tensor_smoke/`
- 为避免覆盖 smoke test，正式运行建议额外指定 `--output-dir`。
- 推荐正式目录：
  `results/real_data_experiments/single_region_client_tensor_quick/`
  `results/real_data_experiments/single_region_client_tensor_main/`
  `results/real_data_experiments/single_region_client_tensor_seed15/`
  `results/real_data_experiments/single_region_client_tensor_seed42/`
  `results/real_data_experiments/single_region_client_tensor_seed48/`
  `results/real_data_experiments/single_region_ablation_tensor_main/`

## 当前已生成的关键结果文件

- `run_config.json`
- `run_commands.txt`
- `environment_summary.json`
- `split_summary.json`
- `selected_regions.csv`
- `main_metrics.csv`
- `main_summary.csv`
- `client_metrics.csv`
- `convergence_history.csv`
- `prediction_samples.csv`
- `experiment_notes_zh.md`
- `figure_index.csv`
- `figure_notes_zh.md`

## 常见问题

- 若提示找不到 `final_sum_mean_standard/node_flow_grid_tensor.pt`，请先确认正式 tensor-only 预处理链路已完成。
- 若训练过慢，可先执行 smoke test 或方案 A。
- 正式实验不建议复用默认输出目录，以免与现有 smoke test 结果混写。

## 当前范围控制

- 当前 `FedAvg` 仍然是标准样本量加权 `FedAvg`。
- 区域实验当前迁移主线为：tensor-only 输入、区域客户端多 region 划分、标准 `FedAvg`、不把 `FedProx` / personalization / server damping 放入默认流程。
- 本阶段不修改 LaTeX，不修改 `simulation_experiments/`，不改变标准 `FedAvg` 主线。
