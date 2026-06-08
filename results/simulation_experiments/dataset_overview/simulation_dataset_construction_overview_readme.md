# 仿真数据集构造总览图说明

本图用于论文“仿真数据构造、客户端划分与 Non-IID 设置”小节。

图件：
- simulation_dataset_construction_overview.png
- simulation_dataset_construction_overview.pdf

图件内容：
- 第一行展示基础仿真数据集：5 个客户端、8 个节点、每客户端 200 个样本、输入窗口 24、预测步长 1。
- 第二行展示增强仿真数据集：5 个客户端、样本量 600/500/700/550/450、分布族 normal/student-t/chi-square/gaussian_mixture/log_normal，以及样本量、分布族、噪声、高峰和事件扰动构成的联合 Non-IID 设置。

说明：
- 本图是实验设计说明图，不是模型性能结果图。
- 本图不涉及 Proposed、Loss-weighted 或 Data-loss weighted。
- 本图用于帮助读者理解基础仿真数据集和增强仿真数据集的差异。
