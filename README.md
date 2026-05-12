# FedTrafficFlow

## 项目当前进度

当前仓库已完成交通路网基础数据预处理，并新增了基于速度分片结果的统计分析与可视化脚本，可直接用于后续建模、数据理解与联邦学习实验设计。

已完成内容：

- 完成 `link_gps` 原始数据清洗与格式化。
- 完成 `road_network_sub-dataset` 路网结构数据清洗、字段重命名与节点坐标推导。
- 完成交通速度数据与路网信息的分块合并脚本。
- 产出可复用的处理后 CSV 文件，供后续训练、分析与联邦切分使用。
- 新增速度等级总体统计、按日期时段统计、分时段直方图与 P99.5 可视化分析结果。
- 新增基于 Greenshields 模型的密度、车辆数与流量计算脚本及查询脚本。

## 当前目录结构

```text
FedTrafficFlow/
├─ analysis_scripts/
│  ├─ add_p995_to_speed_histogram.py
│  ├─ compute_greenshields_density.py
│  ├─ summarize_speed_stats.py
│  ├─ visualize_speed_hist_by_period.py
├─ dataset_inspection_scripts/
│  ├─ check_density_time_order.py
│  ├─ inspect_density_metrics_chunks.py
│  └─ inspect_speed_data_chunks.py
├─ docs/
│  └─ greenshields_speed_density_scheme.md
├─ data/
│  ├─ analysis/
│  │  ├─ density_metrics_chunks/
│  │  ├─ speed_histogram_counts_by_period_by_class.csv
│  │  ├─ speed_histograms_by_class_p995.csv
│  │  ├─ speed_histograms_by_class_with_p995_percent.png
│  │  └─ speed_histograms_by_period_by_class/
│  ├─ params/
│  │  ├─ beijing_capacity_params.csv
│  │  └─ speed_class_density_params.csv
│  ├─ raw/
│  │  ├─ link_gps.v2
│  │  └─ road_network_sub-dataset.v2
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
- 生成可供后续可视化和校验使用的聚合统计结果

### `analysis_scripts/visualize_speed_hist_by_period.py`

功能：

- 对所有速度分片按速度等级和时段进行聚合
- 生成各速度等级在 6 个时段下的平均速度频率直方图
- 输出 `data/analysis/speed_histogram_counts_by_period_by_class.csv`
- 输出到 `data/analysis/speed_histograms_by_period_by_class/`

### `analysis_scripts/add_p995_to_speed_histogram.py`

功能：

- 统计各速度等级的平均速度分布百分比直方图
- 基于细粒度分箱估计各速度等级的 `P99.5`
- 输出 `data/analysis/speed_histograms_by_class_p995.csv`
- 输出 `data/analysis/speed_histograms_by_class_with_p995_percent.png`

### `analysis_scripts/compute_greenshields_density.py`

功能：

- 按 `100 / 80 / 60 km/h` 三档为速度等级分配标准化自由流速度
- 为不同速度等级分配每车道临界密度参数
- 对超过自由流速度的观测做截断，避免出现负密度
- 输出路段密度、车辆数估计和 15 分钟流量等衍生指标
- 输出 `data/params/speed_class_density_params.csv`
- 读取 `data/params/beijing_capacity_params.csv`
- 按 `时间段`、`路段ID`、`速度等级` 升序输出密度分块结果
- 输出 `data/analysis/density_metrics_chunks/`

## 查询脚本说明

### `dataset_inspection_scripts/inspect_speed_data_chunks.py`

功能：

- 查看 `data/processed/speed_data_chunks/` 下分块文件概览
- 输出样例分块的形状、列名、数据类型、空值统计和样例值
- 输出前 50 行与后 20 行数据预览

### `dataset_inspection_scripts/inspect_density_metrics_chunks.py`

功能：

- 查看 `data/analysis/density_metrics_chunks/` 下分块文件概览
- 输出样例分块的形状、列名、数据类型、空值统计和样例值
- 输出前 50 行与后 20 行数据预览

### `dataset_inspection_scripts/check_density_time_order.py`

功能：

- 逐文件读取 `density_metrics_chunks` 中的 `时间段` 列
- 按文件中实际出现顺序压缩连续重复值并输出时间段序列
- 校验每个文件的 `时间段` 是否连续升序
- 输出异常说明及全量汇总结果，便于确认分块顺序是否正确

## 密度建模文档

### `docs/greenshields_speed_density_scheme.md`

文档内容包括：

- 速度等级到自由流速度和临界密度的标准化映射
- 为什么 `P99.5` 只用于校准档位而不直接作为 `v_f`
- Greenshields 密度计算公式与超速截断规则
- 车辆数、流量与 15 分钟流量的派生计算方法

## 当前产出文件

目前已生成以下可直接使用的处理结果：

- `data/processed/link_gps_processed.csv`
- `data/processed/rnsd_processed.csv`

交通速度数据由于体量非常大，分块结果保存在：

- `data/processed/speed_data_chunks/`

当前新增的分析结果包括：

- `data/params/beijing_capacity_params.csv`
- `data/params/speed_class_density_params.csv`
- `data/analysis/speed_histogram_counts_by_period_by_class.csv`
- `data/analysis/speed_histograms_by_class_p995.csv`
- `data/analysis/speed_histograms_by_class_with_p995_percent.png`
- `data/analysis/speed_histograms_by_period_by_class/`
- `data/analysis/density_metrics_chunks/`

用于数据核查的脚本位于：

- `dataset_inspection_scripts/inspect_speed_data_chunks.py`
- `dataset_inspection_scripts/inspect_density_metrics_chunks.py`
- `dataset_inspection_scripts/check_density_time_order.py`

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
python analysis_scripts/visualize_speed_hist_by_period.py
python analysis_scripts/add_p995_to_speed_histogram.py
python analysis_scripts/compute_greenshields_density.py
python dataset_inspection_scripts/inspect_speed_data_chunks.py
python dataset_inspection_scripts/inspect_density_metrics_chunks.py
python dataset_inspection_scripts/check_density_time_order.py
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
