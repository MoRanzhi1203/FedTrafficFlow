# 真实实验运行说明

## 环境准备

- 推荐先安装项目依赖：

```bash
pip install -r requirements.txt
pip install torch matplotlib
```

## 数据准备

- 当前正式单路口实验默认读取：
- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 配套 sidecar：
- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`
- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor_metadata.json`
- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_metadata.json`
- 正式 tensor shape 为 `(2, 630, 5856)`。
- `parquet-direct` 版本仅保留为历史 smoke test fallback。

## 当前可运行模块

### 单路口客户端主实验

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all
```

### 快速 smoke test

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --num-clients 2 --rounds 1 --local-epochs 1 --batch-size 32 --sequence-length 12 --device cpu
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all --input-dir results/real_data_experiments/single_intersection_client_tensor
python -m real_data_experiments.single_intersection_ablation.sia_core --workflow all --data-mode tensor --num-clients 2 --rounds 1 --local-epochs 1 --batch-size 32 --sequence-length 12 --device cpu
python -m real_data_experiments.single_intersection_ablation.sia_visualization --workflow all --input-dir results/real_data_experiments/single_intersection_ablation_tensor
```

## 输出目录

- 单路口客户端结果：
- `results/real_data_experiments/single_intersection_client_tensor/`
- 单路口消融结果：
- `results/real_data_experiments/single_intersection_ablation_tensor/`

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

## 常见错误

- 若提示找不到 `final_sum_mean_standard/node_flow_grid_tensor.pt`，请先确认正式 tensor-only 预处理链路已完成。
- 若训练过慢，可通过 `--num-clients`、`--rounds`、`--device cpu` 进行 smoke test。
- 若需要固定客户端，可通过 `--selected-clients` 指定 region ID。

## 如何复现当前图表

1. 先运行 `sic_core.py` 导出 CSV/JSON 结果。
2. 再运行 `sic_visualization.py` 生成图表。
3. 图表与索引默认写回 `results/real_data_experiments/single_intersection_client_tensor/`。

## 当前说明

- 当前单路口实验中的 `client` 表示 `pooled-grid-region client`。
- `FedAvg` 仍然是标准样本量加权 `FedAvg`。
- 区域实验仍待后续阶段迁移，本阶段未改动。
