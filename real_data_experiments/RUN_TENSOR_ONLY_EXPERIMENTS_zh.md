# Tensor-Only 正式实验运行计划

## 1. 适用范围

- 本文仅用于 tensor-only 单池化网格区域客户端实验的正式运行计划。
- 当前 `client` 表示 `pooled-grid-region client`，不是原始路口节点客户端。
- 当前正式数据入口固定为：
  `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
- 当前正式 `pool_mode = sum_mean`
- 当前正式 `layout = standard`
- 当前正式 `tensor shape = (2, 630, 5856)`
- 当前 `FedAvg = standard sample-size weighted FedAvg`
- `parquet-direct = legacy fallback only`

## 2. 固定 region 建议

- top-3 固定 region IDs：
  `290,284,318`
- top-5 固定 region IDs：
  `290,284,318,288,289`
- 固定推荐清单文件：
  `real_data_experiments/selected_regions_fixed_plan.csv`

## 3. 输出目录保护

- 为避免覆盖 smoke test，正式运行建议使用独立输出目录。
- 当前代码已支持 `--output-dir`。
- 推荐目录：
  `results/real_data_experiments/single_region_client_tensor_quick/`
  `results/real_data_experiments/single_region_client_tensor_main/`
  `results/real_data_experiments/single_region_client_tensor_seed15/`
  `results/real_data_experiments/single_region_client_tensor_seed42/`
  `results/real_data_experiments/single_region_client_tensor_seed48/`
  `results/real_data_experiments/single_region_ablation_tensor_main/`

## 4. 快速正式实验

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --num-clients 3 --selected-clients 290,284,318 --rounds 5 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/single_region_client_tensor_quick
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all --input-dir results/real_data_experiments/single_region_client_tensor_quick --dpi 300
```

## 5. 主论文正式实验

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --num-clients 5 --selected-clients 290,284,318,288,289 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/single_region_client_tensor_main
python -m real_data_experiments.single_intersection_client.sic_visualization --workflow all --input-dir results/real_data_experiments/single_region_client_tensor_main --dpi 300
```

## 6. 单区域消融正式实验

```bash
python -m real_data_experiments.single_intersection_ablation.sia_core --workflow all --data-mode tensor --num-clients 5 --selected-clients 290,284,318,288,289 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/single_region_ablation_tensor_main
python -m real_data_experiments.single_intersection_ablation.sia_visualization --workflow all --input-dir results/real_data_experiments/single_region_ablation_tensor_main --dpi 300
```

## 7. 稳健性 / 多 seed 实验

```bash
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --seed 15 --num-clients 5 --selected-clients 290,284,318,288,289 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/single_region_client_tensor_seed15
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --seed 42 --num-clients 5 --selected-clients 290,284,318,288,289 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/single_region_client_tensor_seed42
python -m real_data_experiments.single_intersection_client.sic_core --workflow all --data-mode tensor --seed 48 --num-clients 5 --selected-clients 290,284,318,288,289 --rounds 20 --local-epochs 3 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/single_region_client_tensor_seed48
```

## 8. 与既有默认命令的对应关系

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

## 9. 执行顺序建议

1. 先跑“快速正式实验”，确认固定 region、结果导出、图表链路和耗时。
2. 再跑“主论文正式实验”。
3. 然后跑“单区域消融正式实验”。
4. 最后按需要补跑“稳健性 / 多 seed 实验”。

## 10. 本阶段说明

- 本文只生成正式运行计划，不直接触发正式长训练。
- 当前不迁移区域实验，不修改 LaTeX，不修改 `simulation_experiments/`，不改变标准 `FedAvg` 主线。
