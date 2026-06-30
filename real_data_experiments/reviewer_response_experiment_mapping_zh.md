# 一审意见与实验 1–6 修改映射表

> 生成日期：2026-06-29
> 将一审审稿意见和导师要求映射到实验 1–6 的具体修改项

---

| 审稿/导师要求 | 对应实验/修改 | 当前状态 | 下一步 |
|---|---|---|---|
| **多区域交通数据定量支撑** | 实验 1/3/5 client 分布统计 | 实验 1 formal 已完成，提供了 5 个 grid cells 的 client-level 指标。实验 3/5 smoke 有 client 分布 summary。实验 5 formal 有 region_assignment 和 non_iid_summary。 | 从已有 `client_distribution_summary.csv` 和 `non_iid_summary.csv` 提取；补充 client 样本量分布表 |
| **收敛性和稳健性** | 每 round loss + RMSE 曲线 | 实验 1 有完整 `convergence_history.json` (20 rounds × 6 metrics)。实验 5 scaler 修复后 r3e1 诊断有 round-level 收敛记录（train_loss: 0.313→0.017）。实验 3 仅 smoke r1e1。 | Exp5 已有 r3e1 round-level 诊断。待 r20 formal 后补全收敛曲线 |
| **超参数表** | 所有实验 | 超参数表已在此次整理中更新，修正了输入通道描述（total flow + mean flow）、归一化状态、聚合策略描述 | ✅ 已修正 |
| **消融实验** | 实验 2/4/6 | Exp2: 代码存在（`sia_core.py`），历史成功运行但目录已删除，需恢复或重跑。Exp4: 未开发。Exp6: scaler 已修复，仅 full variant；待补全 without_attention/without_cnn/without_lstm | P1: 补跑 exp2 formal。P2: 开发 exp4 入口或弱化论文对应叙述。P0: Exp6 scaler 已修复 ✅，下一步补全 4 个 variant |
| **client 数量与异质性** | 实验 3/5 | Exp3: 5 clients (similarity_k5)，smoke 通过。Exp5: 3 clients (spatial_block/flow_kmeans)，formal pipeline 通过。exp5 有 `non_iid_summary.csv` 记录异质性。 | exp3 已产出 client-level smoke 指标，但仅为 r1e1；需要 r20 formal 才能形成 client 异质性结论。exp5 formal 指标异常（scaler 缺失），需修复后重新评估 |
| **对比基线不足** | 实验 1/3/5 | Exp1: 6 baselines (FedAvg, Independent, NaiveLastValue, CalendarProfileNaive, DailySeasonalNaive, WeeklySeasonalNaive)。Exp3: 3 baselines (FedAvg, Independent, NaiveLastValue)。Exp5: ✅ 已补 NaiveLastValue baseline（本轮修复）。 | ✅ Exp5 已补 NaiveLastValue。P2: exp3 formal 后补充周期性 baselines |
| **聚合策略讨论** | 所有联邦实验 | 当前仅使用标准 FedAvg（sample_count 加权），无 λ/β/ρ 超参数。`trainer.py` 和 `fedavg.py` 实现清晰。论文旧稿中出现过 loss-aware weighting、λ/β/ρ 和 smoothing 描述，但代码未实现。 | 论文中统一写为标准 FedAvg。若审稿人要求更多聚合策略对比，需补充 FedProx 或 FedAvgM |
| **通信开销/掉线/鲁棒性** | 写作补充 | 当前实验均在全量 client 参与 + 无掉线条件下运行。无通信开销模拟。 | 在论文 limitations 或 discussion 中说明：当前实验假设全量 client 参与，通信开销和掉线鲁棒性作为未来工作或仿真实验补充 |
| **GCN 真实数据** | 可选 | 当前未在真实数据上运行 GCN 实验。GCN 实验仅在 `results/simulation_experiments/` 中（仿真数据）。 | 在 limitations 中说明计算成本限制；或补一个轻量 GCN 真实数据测试 |
| **日历周期特征对 FedAvg 的影响** | 实验 1 Calendar 诊断 | Exp1 formal 中已输出 CalendarProfileNaive / DailySeasonalNaive / WeeklySeasonalNaive 与 FedAvg 的对比。但 CalendarFeature-FedAvg 和 SeasonalResidual-FedAvg 尚未实现 | 在论文中写：日历特征当前仅作为独立 baseline 评估，尚未接入 FedAvg 训练链路；作为未来工作 |
| **节假日和调休标注** | 实验 1 | 已实现：`calendar_features_15min_2017_04_01_to_2017_05_31.csv` 含清明节、劳动节、端午节及调休上班日 | 可直接写入论文数据描述 |
| **预测模式分析（weekday/weekend）** | 实验 1 | CalendarProfileNaive 按 is_effective_workday + slot_of_day 分离 profile | 可从 client_metrics 中提取 weekday/weekend 分组对比 |
| **论文 formal 结果可信度** | Exp1 ✅ / Exp2 ❌ / Exp3 ❌ / Exp4 ❌ / Exp5 ❌ / Exp6 ❌ | 仅实验 1 产出可在论文中直接使用的 formal 结果。其余均需修复、补跑或补指标 | 当前论文仅能依赖实验 1 作为 grid-cell-level 主结果 |
