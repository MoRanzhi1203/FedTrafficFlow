# 新实验 5：全局所有网格按相似度划分为客户端的对比实验

## 当前定义

- 本目录固定对应新实验 5：`global similarity partition comparison`。
- 旧新映射：原实验 3 -> 新实验 5。
- 当前文档语义强调：全局所有网格按相似度划分为客户端。
- 形式化边界为：`All grid cells are partitioned into K non-overlapping clients.`
- 该实验线关注全局覆盖式客户端划分，且 `client_i and client_j are non-overlapping when i != j.`

## 代码与实现说明

- `rc_config.py`：区域主实验配置。
- `rc_core.py`：区域主实验训练与结果导出。
- `rc_visualization.py`：区域主实验图表生成。
- `region_notebook_migration_zh.md`：原 notebook 到 Python 的迁移映射。
- `historical_notes_zh.md`：历史探索逻辑记录。

## 当前主流程

- 默认输入为 tensor-only：`data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`。
- 当前实现会把参与划分的 pooled regions 分配到多个 client，并输出 assignment / distribution / non-IID 汇总文件。
- 代码保留 `spatial_block` 与 `flow_kmeans` 两种全局划分实现；在新的实验编号语义中，`flow_kmeans` 对应相似度划分主线，`spatial_block` 作为同目录内保留的全局划分实现与 smoke 入口。
- 默认联邦聚合仍为标准样本量加权 `FedAvg`。
- 数据划分仍为按 target time 的连续切分。

## 结果路径归属

- `results/real_data_experiments/region_client_tensor_smoke/` 归入新实验 5 的 smoke 结果。
- 旧结果路径不移动，只在文档和 inventory 中新增新编号对应关系。

## smoke test

```bash
python -m real_data_experiments.region_client.rc_core --workflow all --data-mode tensor --partition-method spatial_block --num-clients 3 --rounds 1 --local-epochs 1 --batch-size 32 --sequence-length 12 --learning-rate 0.001 --device cpu --output-dir results/real_data_experiments/region_client_tensor_smoke
python -m real_data_experiments.region_client.rc_visualization --workflow all --input-dir results/real_data_experiments/region_client_tensor_smoke --dpi 150
```
