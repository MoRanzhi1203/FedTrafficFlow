# Tensor-Only 正式实验运行计划

## 当前状态更新

- `single_intersection_client` / `single_intersection_ablation` 已完成 tensor-only Python 化。
- `region_client` / `region_ablation` 已完成 tensor-only Python 化，并已通过 smoke test。
- 当前 smoke test 结果仅用于验证代码链路、输出文件和可视化流程，不作为论文正式结果。
- smoke test 结果不作为论文正式结果。

## 1. 适用范围

- 本文用于两类真实数据客户端组织设置的 tensor-only 正式运行计划。
- `single_intersection_client` / `single_intersection_ablation` 在文档中统一解释为：
  网格单元级客户端联邦学习设置 / Grid-cell-level Client Federated Learning Setting。
- `region_client` / `region_ablation` 在文档中统一解释为：
  簇级客户端联邦学习设置 / Cluster-level Client Federated Learning Setting。
- 当前正式数据入口固定为：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 当前正式 `pool_mode = sum_mean`
- 当前正式 `layout = standard`
- 当前正式 `tensor shape = (2, 630, 5856)`
- 当前 `FedAvg = standard sample-size weighted FedAvg`
- `parquet-direct = legacy fallback only`

## 2. 命名对照

| 代码目录 | 正式中文名称 | 正式英文名称 | 客户端定义 |
|---|---|---|---|
| `single_intersection_client` / `single_intersection_ablation` | 网格单元级客户端联邦学习设置 | Grid-cell-level Client Federated Learning Setting | 每个 client = 1 个 pooled grid cell |
| `region_client` / `region_ablation` | 簇级客户端联邦学习设置 | Cluster-level Client Federated Learning Setting | 每个 client = 一组 temporal-similarity-clustered grid cells |

## 3. 固定 region 建议

- top-3 固定 region IDs：
  `290,284,318`
- top-5 固定 region IDs：
  `290,284,318,288,289`
- 固定推荐清单文件：
  `real_data_experiments/selected_regions_fixed_plan.csv`

## 4. 输出目录保护

- 为避免覆盖 smoke test，正式运行建议使用独立输出目录。
- 当前代码已支持 `--output-dir`。
- 推荐目录：
  `results/real_data_experiments/single_region_client_tensor_quick/`
  `results/real_data_experiments/single_region_client_tensor_main/`
  `results/real_data_experiments/single_region_client_tensor_seed15/`
  `results/real_data_experiments/single_region_client_tensor_seed42/`
  `results/real_data_experiments/single_region_client_tensor_seed48/`
  `results/real_data_experiments/single_region_ablation_tensor_main/`

## 5. 网格单元级客户端设置快速正式实验

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --num-clients 3 --selected-clients 290,284,318 --rounds 5 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/single_region_client_tensor_quick
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all --input-dir results/real_data_experiments/single_region_client_tensor_quick --dpi 300
```

## 6. 网格单元级客户端设置主论文正式实验

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --num-clients 5 --selected-clients 290,284,318,288,289 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/single_region_client_tensor_main
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all --input-dir results/real_data_experiments/single_region_client_tensor_main --dpi 300
```

## 7. 网格单元级客户端设置消融正式实验

```bash
python -m real_data_experiments.single_intersection_ablation.sia_core --workflow all --data-mode tensor --num-clients 5 --selected-clients 290,284,318,288,289 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/single_region_ablation_tensor_main
python -m real_data_experiments.single_intersection_ablation.sia_visualization --workflow all --input-dir results/real_data_experiments/single_region_ablation_tensor_main --dpi 300
```

## 8. 网格单元级客户端设置稳健性 / 多 seed 实验

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --seed 15 --num-clients 5 --selected-clients 290,284,318,288,289 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/single_region_client_tensor_seed15
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --seed 42 --num-clients 5 --selected-clients 290,284,318,288,289 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/single_region_client_tensor_seed42
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --seed 48 --num-clients 5 --selected-clients 290,284,318,288,289 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/single_region_client_tensor_seed48
```

## 9. 与既有默认命令的对应关系

- 若忽略 `--selected-clients` 与 `--output-dir`，则下面这些是当前代码层面可直接运行的基础命令。
- 但正式运行不建议直接使用默认输出目录，因为会与现有 smoke test 结果混写。

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --num-clients 3 --rounds 5 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all --input-dir results/real_data_experiments/single_intersection_client_tensor --dpi 300

python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --num-clients 5 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all --input-dir results/real_data_experiments/single_intersection_client_tensor --dpi 300

python -m real_data_experiments.single_intersection_ablation.sia_core --workflow all --data-mode tensor --num-clients 5 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu
python -m real_data_experiments.single_intersection_ablation.sia_visualization --workflow all --input-dir results/real_data_experiments/single_intersection_ablation_tensor --dpi 300
```

## 10. 簇级客户端设置 smoke test 命令

区域主实验 smoke：

```bash
python -m real_data_experiments.region_client.rc_core --workflow all --data-mode tensor --partition-method spatial_block --num-clients 3 --rounds 1 --local-epochs 1 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/region_client_tensor_smoke
python -m real_data_experiments.region_client.rc_visualization --workflow all --input-dir results/real_data_experiments/region_client_tensor_smoke --dpi 150
```

区域消融 smoke：

```bash
python -m real_data_experiments.region_ablation.ra_core --workflow all --data-mode tensor --partition-method spatial_block --num-clients 3 --rounds 1 --local-epochs 1 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/region_ablation_tensor_smoke
python -m real_data_experiments.region_ablation.ra_visualization --workflow all --input-dir results/real_data_experiments/region_ablation_tensor_smoke --dpi 150
```

说明：上述命令仅用于 smoke test，不作为论文正式结果。

## 11. 簇级客户端设置正式训练计划

- 簇级客户端主实验正式训练建议后续单独确认：
  `num_clients`、`partition_method`、`rounds`、`local_epochs`、是否多 seed、是否采用 `spatial_block` 或 `flow_kmeans`。
- 在作者确认前，不将 smoke test 指标写入论文正式结果表。

## 12. 执行顺序建议

1. 先跑“快速正式实验”，确认固定 region、结果导出、图表链路和耗时。
2. 再跑“主论文正式实验”。
3. 然后跑“网格单元级客户端设置消融正式实验”。
4. 最后按需要补跑“稳健性 / 多 seed 实验”。

## 13. 本阶段说明

- 本文只生成正式运行计划，不直接触发正式长训练。
- 当前真实数据两类实验线均已完成 tensor-only Python 化；smoke test 结果仅用于验证代码链路、输出文件和可视化流程，不作为论文正式结果。
- 为避免将客户端组织方式误解为新的联邦聚合机制，本文统一使用“setting”而非“mechanism”描述两类真实数据实验。
- 两类设置共享相同的 tensor-only 数据入口、时间顺序切分、本地模型结构和标准样本量加权 `FedAvg` 聚合流程，区别仅在于客户端组织粒度不同。
- 不修改 LaTeX，不修改 `simulation_experiments/`，不改变标准 `FedAvg` 主线。
