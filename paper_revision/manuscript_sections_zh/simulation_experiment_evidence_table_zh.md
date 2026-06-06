# 仿真实验证据表

| 论文结论 | 使用实验组 | 结果文件 | 图表文件 | 是否进入主文 | 注意事项 |
|---------|-----------|---------|---------|------------|---------|
| CNN-FedAvg 在 4/5 客户端上 MSE 优于 Independent | CNN 基础联邦 | `cnn_fed_base/main_metrics.csv` | 待制作：柱状对比图 | 是 | 客户端 2 持平（均为 1.85×10⁻⁴） |
| CNN-FedAvg 平均 MSE=1.83×10⁻⁴，Independent=2.47×10⁻⁴ | CNN 基础联邦 | `cnn_fed_base/main_summary.csv` | — | 是 | 标准差见 CSV mse_std 字段 |
| CNN-FedAvg 在客户端 3 改善最显著（约 2.56 倍） | CNN 基础联邦 | `cnn_fed_base/main_metrics.csv` | — | 是 | 1.91×10⁻⁴ vs 4.88×10⁻⁴ |
| CNN-FedAvg 训练损失从第 1 轮 1.13×10⁻³ 降至第 15 轮约 7.02×10⁻⁵ | CNN 基础联邦 | `cnn_fed_base/convergence_history.csv` | 待制作：收敛曲线（双纵轴） | 是 | 建议仅放图，文本引用关键轮次 |
| CNN-FedAvg 验证 RMSE 从第 1 轮 9.47×10⁻² 降至第 15 轮 1.24×10⁻² | CNN 基础联邦 | `cnn_fed_base/convergence_history.csv` | 同上 | 是 | 前 5 轮为主要下降区间 |
| GCN-FedAvg 在全部汇总指标上优于 Independent | GCN 基础联邦 | `gcn_fed_base/main_summary.csv` | — | 是 | MSE=1.98×10⁻⁴ vs 2.08×10⁻⁴ |
| GCN-FedAvg MAE 标准差（1.11×10⁻³）低于 Independent（1.94×10⁻³） | GCN 基础联邦 | `gcn_fed_base/main_summary.csv` | — | 是 | mae_std 字段 |
| GCN-FedAvg 验证 RMSE 从第 1 轮 2.68×10⁻² 降至第 10 轮 1.38×10⁻² | GCN 基础联邦 | `gcn_fed_base/convergence_history.csv` | 待制作：收敛曲线 | 是 | 趋势与 CNN 一致 |
| 合成路网图：8 节点、10 边、密度 0.357、平均度 2.5 | GCN 基础联邦 | `gcn_fed_base/base_graph_summary.csv` | — | 可选 | 可在实验设定中作为脚注 |
| 低非 IID 下 FedAvg MSE=9.99、MAPE=3.31% | CNN 增强实验 | `cnn_fed_enhanced_experiments/cnn_enhanced_noniid_summary.csv` | 待制作：柱状图（三级对比） | 是 | 仅取 FedAvg 行 |
| 中等非 IID 下 FedAvg MSE=70.98 | CNN 增强实验 | 同上 | 同上 | 是 | — |
| 高非 IID 下 FedAvg MSE=235.61、MAPE=150.87% | CNN 增强实验 | 同上 | 同上 | 是 | 极差结果，反映高异质性挑战 |
| FedAvg 在 3/5/8 客户端下 MSE 分别为 44.37/70.98/63.06 | CNN 增强实验 | `cnn_fed_enhanced_experiments/cnn_enhanced_client_scale_summary.csv` | 可选 | 是 | 非线性趋势，值得讨论 |
| 特征消融：flow_region 最优（MSE=69.22），full 最差（MSE=102.35） | CNN 增强实验 | `cnn_fed_enhanced_experiments/cnn_enhanced_feature_ablation_summary.csv` | 待制作：柱状图 | 是 | FedAvg 策略行 |
| 动态图 MSE 略低于固定图（约 1% 改善） | GCN 增强实验 | `gcn_fed_enhanced_experiments/gcn_enhanced_fixed_vs_dynamic_summary.csv` | 可选 | 是 | FedAvg 策略行；改善有限但在真实数据中可能更大 |
| 客户端掉线 0%→40%，FedAvg RMSE 仅从 7.93 升至 7.98（+0.6%） | 联邦鲁棒性 | `fed_robustness/fed_client_dropout_summary.csv` | 待制作：分组柱状图 | 是 | 高度鲁棒 |
| 通信延迟 1 轮，FedAvg RMSE 升至 8.27（+4.3%），为最大变化 | 联邦鲁棒性 | `fed_robustness/fed_communication_delay_summary.csv` | 同上 | 是 | 最敏感的扰动类型 |
| 梯度噪声 0.05，FedAvg RMSE 升至 8.19（+3.3%） | 联邦鲁棒性 | `fed_robustness/fed_gradient_noise_summary.csv` | 同上 | 是 | 聚合平均对零均值噪声有平滑作用 |
| Loss-weighted、Data-loss weighted、Proposed 等聚合策略对比 | CNN 增强实验（历史探索） | `cnn_fed_enhanced_experiments/cnn_enhanced_aggregation_summary.csv` | 不使用 | 排除主文核心结论 | 历史探索策略，不纳入正文。如确需提及，仅在脚注或展望中写"曾尝试替代聚合策略"而不展开数值 |
