# 区域客户端主实验

## 当前定义

- 当前 `region_client` 表示区域网格客户端。
- 每个 `client` = 一组 pooled grid regions。
- 当前正式输入为 tensor-only：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 当前正式 `regions_path`：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`
- 当前正式 `pool_mode = sum_mean`
- 当前正式 `layout = standard`
- 当前正式 `tensor shape = (2, 630, 5856)`

## 默认主流程

- `partition_method = spatial_block`
- `num_clients = 3`
- `sequence_length = 12`
- `prediction_horizon = 1`
- `use_channels = [0, 1]`
- `target_channel = 0`
- `FedAvg = standard sample-size weighted FedAvg`
- `Independent baseline = enabled`
- `split = temporal_contiguous_by_target_time`

## 目录文件

- `rc_config.py`：区域主实验配置
- `rc_core.py`：区域主实验训练与结果导出
- `rc_visualization.py`：区域主实验图表生成
- `region_notebook_migration_zh.md`：原 notebook 到 py 的迁移映射
- `historical_notes_zh.md`：历史探索逻辑记录

## smoke test

```bash
python -m real_data_experiments.region_client.rc_core --workflow all --data-mode tensor --partition-method spatial_block --num-clients 3 --rounds 1 --local-epochs 1 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/region_client_tensor_smoke
python -m real_data_experiments.region_client.rc_visualization --workflow all --input-dir results/real_data_experiments/region_client_tensor_smoke --dpi 150
```

- 若只做 agent / CI 级 smoke test，可追加：
  `--max-samples-per-client-split 1024`
- 该参数只截断每个 client 的 split 样本数用于加速联调，不改变区域划分、客户端定义和标准样本量加权 `FedAvg` 的权重计算方式。

## 说明

- 当前正式主流程不引入 `FedProx`、`server damping`、`personalization`。
- 历史 notebook 中出现的探索逻辑仅记录在 `historical_notes_zh.md`，默认不运行。
