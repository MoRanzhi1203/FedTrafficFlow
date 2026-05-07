# FedTrafficFlow

## 项目当前进度

当前仓库已完成交通路网基础数据的预处理脚本整理，并生成了可直接用于后续建模与联邦学习实验的中间结果。

已完成内容：

- 完成 `link_gps` 原始数据清洗与格式化。
- 完成 `road_network_sub-dataset` 路网结构数据清洗、字段重命名与节点坐标推导。
- 完成交通速度数据与路网信息的分块合并脚本。
- 产出可复用的处理后 CSV 文件，供后续训练、分析与联邦切分使用。

## 当前目录结构

```text
FedTrafficFlow/
├─ data/
│  ├─ raw/
│  │  ├─ link_gps.v2
│  │  ├─ road_network_sub-dataset.v2
│  │  └─ traffic_speed_sub-dataset.v2
│  └─ processed/
│     ├─ link_gps_processed.csv
│     ├─ rnsd_processed.csv
│     └─ speed_data_chunks/
├─ preprocessing_scripts/
│  ├─ process_link_gps.py
│  ├─ process_rnsd.py
│  └─ merge_speed_data.py
└─ README.md
```

## 预处理脚本说明

### `preprocessing_scripts/process_link_gps.py`

功能：

- 读取 `data/raw/link_gps.v2`
- 去重并删除关键字段缺失值
- 转换字段类型
- 输出 `data/processed/link_gps_processed.csv`

### `preprocessing_scripts/process_rnsd.py`

功能：

- 读取 `data/raw/road_network_sub-dataset.v2`
- 将英文列名重命名为中文字段
- 清洗缺失值与重复路段
- 转换数值字段类型
- 根据方向和长度推导起点/终点经纬度
- 输出 `data/processed/rnsd_processed.csv`

### `preprocessing_scripts/merge_speed_data.py`

功能：

- 读取已处理的 `link_gps_processed.csv` 与 `rnsd_processed.csv`
- 生成用于关联交通速度数据的路网信息
- 使用 `Polars` 按块读取超大规模速度数据
- 将速度数据和路网信息合并后输出到 `data/processed/speed_data_chunks/`

## 当前产出文件

目前已生成以下可直接使用的处理结果：

- `data/processed/link_gps_processed.csv`
- `data/processed/rnsd_processed.csv`

交通速度数据由于体量非常大，分块结果保存在：

- `data/processed/speed_data_chunks/`

## 大文件说明

仓库中的交通速度原始数据与分块合并结果体量较大：

- `data/raw/traffic_speed_sub-dataset.v2` 约 `7.9 GB`
- `data/processed/speed_data_chunks/` 总体约 `28.5 GB`

这两部分内容不适合直接纳入普通 GitHub 仓库版本管理，因此当前已在 `.gitignore` 中忽略。仓库将保留：

- 预处理脚本
- 轻量级处理结果
- 项目说明文档

如需复现完整数据处理流程，可在本地准备原始数据后执行脚本重新生成。

## 使用方式

建议按以下顺序执行：

```bash
python preprocessing_scripts/process_link_gps.py
python preprocessing_scripts/process_rnsd.py
python preprocessing_scripts/merge_speed_data.py
```

## 下一步建议

- 增加环境依赖说明，例如 `pandas`、`numpy`、`polars`
- 补充联邦学习任务定义与数据划分方案
- 增加训练入口脚本与实验配置说明
