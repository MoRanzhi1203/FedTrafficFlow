# 新实验 3 / 4：多个相似网格合并为一个客户端的实验线

## 当前定位

- 本目录在新的实验编号体系下承接两条语义。
- 新实验 3：`similar grid group client comparison`。
- 新实验 4：`similar grid group client ablation`。
- 旧新映射：原实验 5 -> 新实验 3；基于原实验 5 的客户端组织方式补齐消融 -> 新实验 4。
- 当前阶段优先保持目录不移动，只更新编号、README、inventory、历史说明与结果归属。

## 客户端组织边界

- 本目录的实验语义强调：`client_k = {grid_cell_a, grid_cell_b, grid_cell_c, ...}`。
- 重点是把若干相似 grid cells 合并到同一个 client 中。
- 文档口径不把本实验线写成新实验 5/6 那种全局覆盖式完整划分主线。
- 即使当前实现依赖 full-cell inventory 与 partition 文件，本次重构后的实验语义仍固定为 grouped-client 线路。

## 当前实现内容

- `rfc_full_cell_inventory.py`：只读盘点全部有效 cells，输出 inventory CSV/报告。
- `rfc_partition.py`：生成 `spatial` / `similarity` 两类 partition JSON。
- `rfc_dataset.py`：根据 partition 文件构造多-cell client 数据集与 split summary。
- `rfc_core.py`：对比实验训练入口，输出 `FedAvg` / `Independent` / `NaiveLastValue` 三类结果。
- `rfc_compare_report.py`：生成 grouped-client smoke 对比报告。
- `rfc_eval.py` / `rfc_report.py`：结果读取与 markdown 辅助。

## 新实验 3 与新实验 4 的分工

- 新实验 3 对应 grouped-client 对比实验，当前已有 inventory、partition、dataset、core 与 smoke 结果。
- 新实验 4 对应基于同一客户端组织方式的消融补齐。
- 本阶段仅补齐编号与文档说明，不新增独立 ablation 训练入口，不运行训练。

## 结果路径归属

- `results/real_data_experiments/diagnostics/full_cells_similarity_k5_smoke_r1_e1_lr5e4_cuda/` 归入新实验 3 的 similarity diagnostic/smoke 结果。
- `results/real_data_experiments/diagnostics/full_cells_spatial_k5_smoke_r1_e1_lr5e4_cuda/` 保留为同目录下的辅助空间对照结果；旧路径不移动。
- 新实验 4 当前尚无独立结果目录，本阶段仅完成文档和 inventory 补位。

## 边界声明

- FedAvg 仍是标准样本量加权 FedAvg。
- 不引入 FedProx、个性化联邦、相似度加权或 loss 加权。
- 不修改模型结构。
- 不修改数据划分原则。
- 不删除 `NaiveLastValue`，不删除或替换 `289`。
- 不改动已有 `results/` 结果文件。

## 运行示例

```bash
# CUDA formal / recommended
python -m real_data_experiments.region_client_full_cells.rfc_core --workflow all --device cuda --tensor-path data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt --partition-file real_data_experiments/region_client_full_cells/partitions/similarity_k5.json --output-dir results/real_data_experiments/region_client_full_cells_formal_cuda

# CPU smoke / connectivity only
python -m real_data_experiments.region_client_full_cells.rfc_core --workflow all --device cpu --rounds 1 --local-epochs 1 --tensor-path data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt --partition-file real_data_experiments/region_client_full_cells/partitions/similarity_k5.json --output-dir results/real_data_experiments/region_client_full_cells_smoke_cpu
```

## 设备默认值

- 当前代码默认设备已统一改为 `cuda` 优先。
- 若当前环境中 `torch.cuda.is_available()` 为 `False`，代码会自动 fallback 到 `cpu`。
- 新实验 4 当前与新实验 3 共用 `rfc_core.py` 的设备解析逻辑。
