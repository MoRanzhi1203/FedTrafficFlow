# 仿真数据集构造总览图说明

旧图处理：
- 已弃用旧组合图 `simulation_dataset_construction_overview.png`
- 已弃用旧组合图 `simulation_dataset_construction_overview.pdf`
- 旧图已移动至 `deprecated/` 目录，仅保留历史记录，不再用于正文

当前推荐图件：
- base_simulation_dataset_overview.png
- base_simulation_dataset_overview.pdf
- enhanced_simulation_noniid_overview.png
- enhanced_simulation_noniid_overview.pdf

图件使用位置：
- `base_simulation_dataset_overview.png` 用于“仿真数据构造、客户端划分与 Non-IID 设置”小节中的基础仿真数据集说明图
- `enhanced_simulation_noniid_overview.png` 用于同一小节中的增强仿真数据集 Non-IID 说明图

数据来源说明：
- 基础仿真图基于 `simulation_experiments/cnn_fed_base/cfb_core.py` 中的 `generate_base_traffic_data()` 生成，并结合 `simulation_experiments/gcn_fed_base/gfb_core.py` 中的基础图结构拓扑
- 增强仿真图基于 `simulation_experiments/cnn_fed_enhanced_experiments/cfe_core.py` 中的 `CLIENT_CONFIGS_BASE`、`generate_traffic_flow()` 与 `build_sequences()` 生成
- 本脚本仅用于生成数据说明图，不触发模型训练，不修改已有实验 CSV

说明：
- 基础图强调轻度样本量不平衡、受控弱异质性与基础路网结构
- 增强图强调样本量不平衡、目标值分布差异、峰型差异以及噪声/事件扰动共同构成的 Non-IID 来源
- 图中不涉及 Proposed、Loss-weighted 或 Data-loss weighted
