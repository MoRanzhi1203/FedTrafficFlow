# 仿真实验证据表 v2

| 论文结论 | 使用实验组 | 结果文件 | 证据状态 | 是否进入主文 | 注意事项 |
|---|---|---|---|---|---|
| 标准 FedAvg 是本文仿真实验唯一主线聚合方式 | 全部仿真实验 | `simulation_experiments/cnn_fed_base/cfb_core.py`；`simulation_experiments/gcn_fed_base/gfb_core.py`；`paper_revision/00_NATURE_SKILLS_PRECHECK.md` | 已核验 | 是 | v2 已排除 `Proposed`、`Loss-weighted`、`Data-loss weighted` 的主文方法地位 |
| 基础实验训练/验证/测试比例为 70%/10%/20% | CNN 基础联邦；GCN 基础联邦 | `simulation_experiments/cnn_fed_base/cfb_core.py`；`simulation_experiments/gcn_fed_base/gfb_core.py` | 已核验 | 是 | 已修正 v1 中出现过的 70%/15%/15% 表述 |
| CNN 基础实验 FedAvg 平均 MSE 为 0.00018285，优于 Independent 的 0.00024686 | CNN 基础联邦 | `results/simulation_experiments/cnn_fed_base/main_summary.csv` | 已核验 | 是 | 直接取 `FedAvg` 与 `Independent` 行 |
| CNN 基础实验中客户端 3 的联邦收益最明显 | CNN 基础联邦 | `results/simulation_experiments/cnn_fed_base/main_metrics.csv` | 已核验 | 是 | 客户端 3 的 MSE 由 0.00048777 降至 0.00019087 |
| GCN 基础实验 FedAvg 在汇总指标上优于 Independent | GCN 基础联邦 | `results/simulation_experiments/gcn_fed_base/main_summary.csv` | 已核验 | 是 | 四项指标均可直接追溯到 CSV |
| 基础合成路网包含 8 个节点、10 条边，图密度为 0.3571 | GCN 基础联邦 | `results/simulation_experiments/gcn_fed_base/base_graph_summary.csv` | 已核验 | 是 | 用于说明 GCN 基础图结构设定 |
| CNN 基础实验收敛记录为 15 轮，而非笼统 10 轮 | CNN 基础联邦 | `simulation_experiments/cnn_fed_base/cfb_core.py`；`results/simulation_experiments/cnn_fed_base/convergence_history.csv` | 已核验 | 是 | 主训练轮次 10，收敛记录文件为 15 轮 |
| GCN 基础实验收敛记录为 10 轮 | GCN 基础联邦 | `simulation_experiments/gcn_fed_base/gfb_core.py`；`results/simulation_experiments/gcn_fed_base/convergence_history.csv` | 已核验 | 是 | 需与 CNN 明确区分 |
| 增强异质性默认场景下，FedAvg 相对 Independent 仍保持一定优势 | CNN 增强实验 | `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_main_summary.csv` | 已核验 | 是 | 仅使用 `FedAvg` 与 `Independent` 行；`Proposed` 不进入主文 |
| 低/中/高 Non-IID 分层结果可作为 FedAvg 主文结论 | CNN 增强实验 | `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_noniid_summary.csv` | 缺少 FedAvg 行 | 否 | 当前汇总文件只有 `Proposed` 行，不能将数值写成 FedAvg 结论 |
| 3/5/8 客户端数量实验可作为 FedAvg 主文结论 | CNN 增强实验 | `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_client_scale_summary.csv` | 缺少 FedAvg 行 | 否 | 当前汇总文件只有 `Proposed` 行 |
| 特征消融结果可作为 FedAvg 主文结论 | CNN 增强实验 | `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation_summary.csv` | 缺少 FedAvg 行 | 否 | 当前汇总文件只有 `Proposed` 行 |
| 鲁棒性实验可用于说明掉线、延迟和梯度噪声扰动下的 FedAvg 表现 | 联邦鲁棒性 | `results/simulation_experiments/fed_robustness/fed_client_dropout_summary.csv`；`fed_communication_delay_summary.csv`；`fed_gradient_noise_summary.csv` | 已核验 | 是 | 主文只保留 `FedAvg` 行，且“梯度噪声”只能写为模拟扰动 |
| 梯度噪声属于正式差分隐私机制 | 联邦鲁棒性 | `results/simulation_experiments/fed_robustness/fed_gradient_noise_summary.csv`；`simulation_experiments/fed_robustness_experiments/fr_core.py` | 暂不进入主文 | 否 | 当前只是 simulated gradient perturbation，不能写成 DP |
| GCN 固定图与动态图比较可进入主文，但需限定为趋势性证据 | GCN 增强实验 | `results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic_summary.csv`；`simulation_experiments/gcn_fed_enhanced_experiments/gfe_core.py` | 已核验 | 是 | `SEEDS = [42]`，只能作趋势性表述 |
| 增强 GCN 已具备强统计结论条件 | GCN 增强实验 | `simulation_experiments/gcn_fed_enhanced_experiments/gfe_core.py` | 需作者确认 | 否 | 当前仅单种子，若需强统计结论需补多种子结果 |
| 含 `Proposed`、`Loss-weighted`、`Data-loss weighted` 的聚合对比可进入主文主结果 | CNN 增强实验；GCN 增强实验 | `results/simulation_experiments/cnn_fed_enhanced_experiments/cnn_enhanced_aggregation_summary.csv`；`results/simulation_experiments/gcn_fed_enhanced_experiments/gcn_enhanced_aggregation_summary.csv` | 仅历史探索 | 否 | 可作为历史探索材料记录，不进入主文 |
