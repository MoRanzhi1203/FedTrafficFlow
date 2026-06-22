# 真实实验运行说明

## 环境准备

- 推荐先安装项目依赖：

```bash
pip install -r requirements.txt
pip install torch matplotlib
```

## 数据准备

- 当前真实实验最小可运行版本直接读取：
- `data/analysis/node_intersection_flow_parquet/`
- 不再依赖 notebook 中缺失的 `6.池化网格张量.pt`。

## 当前可运行模块

### 单路口客户端主实验

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all
```

### 快速 smoke test

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --max-chunks 2 --num-clients 2 --rounds 1 --local-epochs 1 --batch-size 32 --sequence-length 12 --device cpu
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all
```

## 输出目录

- 单路口客户端结果：
- `results/real_data_experiments/single_intersection_client/`

## 当前已生成的关键结果文件

- `run_config.json`
- `run_commands.txt`
- `environment_summary.json`
- `split_summary.json`
- `main_metrics.csv`
- `main_summary.csv`
- `client_metrics.csv`
- `convergence_history.csv`
- `prediction_samples.csv`
- `experiment_notes_zh.md`
- `figure_index.csv`
- `figure_notes_zh.md`

## 常见错误

- 若提示找不到 `data/analysis/node_intersection_flow_parquet/`，请先确认真实数据预处理链路已完成。
- 若训练过慢，可通过 `--max-chunks`、`--num-clients`、`--rounds`、`--device cpu` 进行 smoke test。
- 若需要固定客户端，可通过后续版本的 `--selected-clients` 指定节点 ID。

## 如何复现当前图表

1. 先运行 `sic_core.py` 导出 CSV/JSON 结果。
2. 再运行 `sic_visualization.py` 生成图表。
3. 图表与索引默认写回 `results/real_data_experiments/single_intersection_client/`。

## 待补齐模块

- `single_intersection_ablation`
- `region_client`
- `region_ablation`

上述模块目录骨架已创建，但本轮首次交付尚未完成正式迁移与重跑。
