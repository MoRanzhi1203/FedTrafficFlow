# 全量有效 grid cells 多客户端组织实验

## 目标

- 本目录承载一个新的独立真实数据实验：
  使用全部有效 pooled grid cells，将多个 cells 组织成一个 client。
- 本实验不覆盖 `single_intersection_client`，也不替换当前 K=5 单 grid-cell 结果。
- 本实验同时支持两种 client 组织方式：
  - `spatial`：空间连续或相邻 cells 组成一个 client。
  - `similarity`：时间模式相似的 cells 组成一个 client。

## 关键文件

- `rfc_full_cell_inventory.py`：只读盘点全部有效 cells，输出 inventory CSV/报告。
- `rfc_partition.py`：生成 `spatial` / `similarity` 两类 partition JSON。
- `rfc_dataset.py`：根据 partition 文件构造多-cell client 数据集与 split summary。
- `rfc_core.py`：训练入口，输出 `FedAvg` / `Independent` / `NaiveLastValue` 三类结果。
- `rfc_compare_report.py`：生成 full-cells smoke 对比报告。
- `rfc_eval.py` / `rfc_report.py`：结果读取与 markdown 辅助。

## 边界

- FedAvg 仍是标准样本量加权 FedAvg。
- 不引入 FedProx、个性化联邦、相似度加权或 loss 加权。
- 不修改模型结构。
- 不修改数据划分原则。
- 所有训练输出只进入 `results/real_data_experiments/diagnostics/`。

