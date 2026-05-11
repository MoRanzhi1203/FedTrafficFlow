# FedTrafficFlow

## 项目当前进度

当前仓库已完成交通路网基础数据预处理，并新增了基于速度分片结果的统计分析与可视化脚本，可直接用于后续建模、数据理解与联邦学习实验设计。

已完成内容：

- 完成 `link_gps` 原始数据清洗与格式化。
- 完成 `road_network_sub-dataset` 路网结构数据清洗、字段重命名与节点坐标推导。
- 完成交通速度数据与路网信息的分块合并脚本。
- 产出可复用的处理后 CSV 文件，供后续训练、分析与联邦切分使用。
- 新增速度等级总体统计、按日期时段统计、分时段直方图与 P99.5 可视化分析结果。

## 当前目录结构

```text
FedTrafficFlow/
├─ analysis_scripts/
│  ├─ summarize_speed_stats.py
│  ├─ visualize_speed_hist_by_period.py
│  └─ add_p995_to_speed_histogram.py
├─ data/
│  ├─ analysis/
│  │  ├─ speed_class_overall_stats.csv
│  │  ├─ speed_class_daily_period_stats.csv
│  │  ├─ speed_histograms_by_class_p995.csv
│  │  ├─ speed_histograms_by_class_with_p995_percent.png
│  │  └─ speed_histograms_by_period_by_class/
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

## 分析脚本说明

### `analysis_scripts/summarize_speed_stats.py`

功能：

- 读取 `data/processed/speed_data_chunks/` 下的所有速度分片
- 基于 `Polars` 流式引擎统计各速度等级的总体速度分布
- 按日期、时段、速度等级生成分组统计结果
- 输出 `data/analysis/speed_class_overall_stats.csv`
- 输出 `data/analysis/speed_class_daily_period_stats.csv`

### `analysis_scripts/visualize_speed_hist_by_period.py`

功能：

- 对所有速度分片按速度等级和时段进行聚合
- 生成各速度等级在 6 个时段下的平均速度频率直方图
- 输出到 `data/analysis/speed_histograms_by_period_by_class/`

### `analysis_scripts/add_p995_to_speed_histogram.py`

功能：

- 统计各速度等级的平均速度分布百分比直方图
- 基于细粒度分箱估计各速度等级的 `P99.5`
- 输出 `data/analysis/speed_histograms_by_class_p995.csv`
- 输出 `data/analysis/speed_histograms_by_class_with_p995_percent.png`

## 当前产出文件

目前已生成以下可直接使用的处理结果：

- `data/processed/link_gps_processed.csv`
- `data/processed/rnsd_processed.csv`

交通速度数据由于体量非常大，分块结果保存在：

- `data/processed/speed_data_chunks/`

当前新增的分析结果包括：

- `data/analysis/speed_class_overall_stats.csv`
- `data/analysis/speed_class_daily_period_stats.csv`
- `data/analysis/speed_histograms_by_class_p995.csv`
- `data/analysis/speed_histograms_by_class_with_p995_percent.png`
- `data/analysis/speed_histograms_by_period_by_class/`

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
python analysis_scripts/summarize_speed_stats.py
python analysis_scripts/visualize_speed_hist_by_period.py
python analysis_scripts/add_p995_to_speed_histogram.py
```

建议环境依赖至少包括：

- `polars`
- `matplotlib`
- `seaborn`

## 下一步建议

- 补充 `requirements.txt` 或环境安装说明，固定分析与预处理依赖版本
- 增加分析图表样例说明，帮助快速理解各速度等级的分布特征
- 补充联邦学习任务定义、客户端划分方案与评估指标
- 增加训练入口脚本与实验配置说明
