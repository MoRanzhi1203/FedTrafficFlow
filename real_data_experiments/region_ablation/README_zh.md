# 簇级客户端设置消融实验

## 当前定义

- 当前 `region_ablation` 在文档中统一解释为：簇级客户端联邦学习设置 / Cluster-level Client Federated Learning Setting。
- 每个 `client` = 一组 pooled grid regions。
- 当前正式输入为 tensor-only：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 当前正式 `regions_path`：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_regions.csv`
- 当前正式 `pool_mode = sum_mean`
- 当前正式 `layout = standard`

## 默认主流程

- `partition_method = spatial_block`
- `num_clients = 3`
- `sequence_length = 12`
- `prediction_horizon = 1`
- `use_channels = [0, 1]`
- `target_channel = 0`
- `FedAvg = standard sample-size weighted FedAvg`

## 默认消融变体

- `Full`
- `Without Attention`
- `Without CNN / Spatial Encoder`
- `Without LSTM`

## 目录文件

- `ra_config.py`：区域消融实验配置
- `ra_core.py`：区域消融实验训练与结果导出
- `ra_visualization.py`：区域消融图表生成
- `region_ablation_notebook_migration_zh.md`：原 notebook 到 py 的迁移映射
- `historical_notes_zh.md`：历史说明

## smoke test

```bash
python -m real_data_experiments.region_ablation.ra_core --workflow all --data-mode tensor --partition-method spatial_block --num-clients 3 --rounds 1 --local-epochs 1 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/region_ablation_tensor_smoke
python -m real_data_experiments.region_ablation.ra_visualization --workflow all --input-dir results/real_data_experiments/region_ablation_tensor_smoke --dpi 150
```

- 若只做 agent / CI 级 smoke test，可追加：
  `--max-samples-per-client-split 1024`
- 该参数只截断每个 client 的 split 样本数用于加速联调，不改变区域划分、客户端定义和标准样本量加权 `FedAvg` 的权重计算方式。
