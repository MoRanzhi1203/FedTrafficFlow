# 新实验 6：全局所有网格按相似度划分为客户端的消融实验

## 当前定义

- 本目录固定对应新实验 6：`global similarity partition ablation`。
- 旧新映射：原实验 4 -> 新实验 6。
- 当前文档语义强调：沿用新实验 5 的全局覆盖式客户端划分，只做结构消融。
- 形式化边界为：所有网格必须进入某个 client，且每个网格只属于一个 client。

## 默认主流程

- 默认输入为 tensor-only：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 当前实现复用全局覆盖式客户端划分数据构造逻辑。
- 默认联邦聚合仍为标准样本量加权 `FedAvg`。
- 默认只比较结构消融，不引入非主线聚合。

## 默认消融变体

- `Full`
- `Without Attention`
- `Without CNN / Spatial Encoder`
- `Without LSTM`

## 结果路径归属

- `results/real_data_experiments/region_ablation_tensor_smoke/` 归入新实验 6 的 smoke 结果。
- 旧结果路径不移动，只新增新编号说明。

## 目录文件

- `ra_config.py`：区域消融实验配置。
- `ra_core.py`：区域消融实验训练与结果导出。
- `ra_visualization.py`：区域消融图表生成。
- `region_ablation_notebook_migration_zh.md`：原 notebook 到 Python 的迁移映射。
- `historical_notes_zh.md`：历史说明。

## smoke test

```bash
python -m real_data_experiments.region_ablation.ra_core --workflow all --data-mode tensor --partition-method spatial_block --num-clients 3 --rounds 1 --local-epochs 1 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/region_ablation_tensor_smoke
python -m real_data_experiments.region_ablation.ra_visualization --workflow all --input-dir results/real_data_experiments/region_ablation_tensor_smoke --dpi 150
```
