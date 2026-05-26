# 参数文件说明

## 概述

本文档说明 `data/params/` 目录下参数文件的角色、字段含义、当前使用位置以及维护建议。

当前目录包含两个参数文件：

- `data/params/speed_class_density_params.csv`
- `data/params/beijing_capacity_params.csv`

两者的定位不同：

- `speed_class_density_params.csv`：当前 `compute_greenshields_density.py` 的直接输入参数表
- `beijing_capacity_params.csv`：容量与临界速度的参考参数表，当前主要用于文档说明和后续扩展参考

## 1. `speed_class_density_params.csv`

### 作用

该文件是当前 Greenshields 密度计算流程真正使用的参数源。

对应脚本：

- `analysis_scripts/compute_greenshields_density.py`

脚本读取位置：

- `PARAM_PATH = data/params/speed_class_density_params.csv`

### 当前字段

| 字段名 | 含义 |
| --- | --- |
| `速度等级` | 原始路网中的速度等级编码 |
| `P99.5速度` | 由速度分布统计得到的等级高分位速度 |
| `自由流速度` | 用于密度计算的标准化自由流速度 |
| `每车道临界密度` | 每车道临界密度参数 |
| `速度档位` | 对应的标准速度档说明 |
| `说明` | 人工可读的参数解释 |

### 当前数据含义

项目当前采用三档标准自由流速度：

```text
速度等级 2、3       -> 100 km/h
速度等级 4          -> 80 km/h
速度等级 5、6、7、8 -> 60 km/h
```

Greenshields 计算中实际使用的核心字段为：

- `速度等级`
- `P99.5速度`
- `自由流速度`
- `每车道临界密度`

脚本内部会进一步派生：

- `p995_class`
- `v_f`
- `k_c_lane`
- `k_j_lane = 2 * k_c_lane`

### 与脚本的关系

`compute_greenshields_density.py` 会做以下兼容处理：

- 若文件列名中没有 `P99.5速度`，但存在 `第99.5百分位速度`，会自动重命名
- 会校验关键字段是否存在且为正值
- 会将表中的参数映射到每条速度观测记录，再进行密度与流量计算

### 维护建议

- 若重新统计新的 P99.5 结果，应同步更新本文件
- 若调整自由流速度档位或临界密度，也应直接更新本文件
- 文档和代码应保持一致，避免 README 中把该文件误写成“脚本输出”

## 2. `beijing_capacity_params.csv`

### 作用

该文件保存标准速度档下的参考通行能力、临界密度和临界速度。

当前字段：

| 字段名 | 含义 |
| --- | --- |
| `标准速度` | 标准速度档 |
| `通行能力` | 该速度档下的参考通行能力 |
| `临界密度` | 临界密度参考值 |
| `临界速度` | 临界速度参考值 |

### 当前定位

在当前代码版本中，这个文件**没有被主流程脚本直接读取**。

它更适合作为：

- 参数推导参考表
- 文档中的工程背景参数
- 后续容量约束、状态分级或模型扩展时的候选输入

### 建议用法

若未来需要将容量约束正式纳入计算，可考虑以下方向：

- 在 `compute_greenshields_density.py` 中显式读取该文件
- 用 `标准速度` 与 `自由流速度` 建立映射
- 将 `通行能力`、`临界速度` 作为额外输出字段或归一化约束依据

在未接入代码前，应避免在 README 中把它描述成“当前主流程必读输入”。

## 3. 参数文件与统计结果的关系

当前参数链路可概括为：

```text
speed_data_chunks
-> add_p995_to_speed_histogram.py
-> speed_histograms_by_class_p995.csv
-> 人工整理/确认
-> speed_class_density_params.csv
-> compute_greenshields_density.py
```

其中：

- `speed_histograms_by_class_p995.csv` 是统计结果
- `speed_class_density_params.csv` 是整理后的建模参数表

两者不应混为同一种文件。

## 4. 推荐维护原则

- 将“统计产物”和“人工确认后的参数表”分开管理
- 将“当前脚本真实使用的参数文件”和“仅供参考的参数文件”分开说明
- 每次更新参数表后，同步检查：
  - `README.md`
  - `docs/greenshields_speed_density_scheme.md`
  - `analysis_scripts/compute_greenshields_density.py`

## 相关文件

- `data/params/speed_class_density_params.csv`
- `data/params/beijing_capacity_params.csv`
- `analysis_scripts/compute_greenshields_density.py`
- `analysis_scripts/add_p995_to_speed_histogram.py`
- `docs/greenshields_speed_density_scheme.md`
