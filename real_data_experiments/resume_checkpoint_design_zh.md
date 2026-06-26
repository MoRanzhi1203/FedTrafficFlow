# 真实数据实验断点续跑改造方案

## 1. 本阶段范围

本阶段只做静态审计和方案设计，不运行实验、不运行训练、不运行 profiling、不修改代码。

本阶段只基于当前真实数据实验代码结构，分析如何为四类正式实验增加断点续跑能力，并明确：

- 断点续跑应挂载在哪些训练循环边界；
- checkpoint 应保存哪些内容；
- resume 前需要做哪些一致性校验；
- 输出目录、Git 边界和后续分批实施顺序应如何冻结。

本阶段不改变标准样本量加权 FedAvg，不改变模型结构，不改变客户端划分，不改变 train/val/test split，不改变 tensor 数据入口，不改变指标计算。

## 2. 当前训练流程静态审计

### 2.1 grid_cell main

- 入口文件：`real_data_experiments/single_intersection_client/sic_core.py`
- FedAvg 聚合函数位置：`real_data_experiments/common/fedavg.py` 中的 `fedavg_aggregate()`
- federated round 循环位置：`sic_core.py` 中 `run_fedavg_experiment()` 的 `for round_idx in range(1, config.communication_rounds + 1):`
- client local training 调用位置：`run_fedavg_experiment()` 内部，每个 round 为每个 client 新建 `local_model`，再调用 `train_local_model()`
- 结果写出位置：`sic_core.py` 中 `export_results()`
- `run_config.json` / `split_summary.json` / `main_metrics.csv` / `main_summary.csv` / `client_metrics.csv` / `convergence_history.csv` / `prediction_samples.csv` 都在 `export_results()` 末尾统一写出
- 当前 `output_dir` 处理逻辑：`run_experiment()` 一开始调用 `prepare_output_dir(config.output_dir)`，而 `prepare_output_dir()` 只是确保目录存在，不会拒绝覆盖既有目录
- 当前没有 checkpoint / resume 实现

### 2.2 grid_cell ablation

- 入口文件：`real_data_experiments/single_intersection_ablation/sia_core.py`
- variant 外层循环位置：`run_experiment()` 中 `for variant in variant_names:`
- 每个 variant 内 FedAvg round 位置：`run_variant()` 中通过 `run_federated_rounds()` 进入 `common/trainer.py`
- `run_federated_rounds()` 的 round 循环位置：`real_data_experiments/common/trainer.py` 中 `for round_idx in range(1, communication_rounds + 1):`
- client local training 调用位置：`common/trainer.py` 中 `local_results = [client.train(global_state) for client in clients]`
- 结果写出位置：`sia_core.py` 中 `export_results()`
- `run_config.json` / `split_summary.json` / `ablation_metrics.csv` / `ablation_summary.csv` / `ablation_client_metrics.csv` / `convergence_history.csv` / `prediction_samples.csv` 在 `export_results()` 统一写出
- 当前 `output_dir` 同样只做 `prepare_output_dir()`，没有覆盖保护
- 当前没有 checkpoint / resume 实现

### 2.3 cluster main

- 入口文件：`real_data_experiments/region_client/rc_core.py`
- FedAvg round 位置：`rc_core.py` 中 `run_fedavg_experiment()` 调用 `run_federated_rounds()`
- `run_federated_rounds()` 内部 round 循环位置：`common/trainer.py`
- client local training 调用位置：`common/client.py` 中 `FedClient.train()`
- 结果写出位置：`rc_core.py` 中 `export_results()`
- `run_config.json` / `split_summary.json` / `region_assignment.csv` / `client_distribution_summary.csv` / `non_iid_summary.csv` / `main_metrics.csv` / `main_summary.csv` / `client_metrics.csv` / `convergence_history.csv` / `prediction_samples.csv` 都在 `export_results()` 统一写出
- 当前 `output_dir` 仍然只做 `prepare_output_dir()`，没有 overwrite 保护
- 当前没有 checkpoint / resume 实现

### 2.4 cluster ablation

- 入口文件：`real_data_experiments/region_ablation/ra_core.py`
- variant 外层循环位置：`run_experiment()` 中 `for variant in selected_variants:`
- 每个 variant 内 FedAvg round 位置：`run_variant()` 中调用 `run_federated_rounds()`
- client local training 调用位置：`common/client.py` 中 `FedClient.train()`
- 结果写出位置：`ra_core.py` 中 `export_results()`
- `run_config.json` / `split_summary.json` / `region_assignment.csv` / `client_distribution_summary.csv` / `non_iid_summary.csv` / `ablation_metrics.csv` / `ablation_summary.csv` / `ablation_client_metrics.csv` / `convergence_history.csv` 都在 `export_results()` 统一写出
- 当前 `output_dir` 同样只会自动创建，不会拒绝复用旧目录
- 当前没有 checkpoint / resume 实现

### 2.5 公共层补充结论

- 标准 FedAvg 聚合函数固定在 `common/fedavg.py` 的 `fedavg_aggregate()`
- 公共 round 循环固定在 `common/trainer.py` 的 `run_federated_rounds()`
- 当前 `common/client.py` 的 `FedClient.train()` 每次调用都会：
  - 新建 local model
  - 重新加载 global state
  - 新建 `Adam` optimizer
- 因此当前 optimizer 不是跨 round 持久复用，而是每次 client-local train 临时创建
- `common/result_writer.py` 的 `prepare_output_dir()` 只做 `mkdir(parents=True, exist_ok=True)`，没有 `output_dir 已存在但未 resume 就拒绝覆盖` 的保护逻辑

## 3. 为什么需要断点续跑

真实数据正式实验已经进入长时运行阶段，断点续跑的工程必要性主要来自以下风险：

- 正式实验比 smoke / profiling 更长，尤其是 `rounds=20`、`local_epochs=3`、多 client 与多 variant 串行时
- Windows 笔记本环境存在休眠、系统更新、显卡驱动波动、电源策略中断等现实风险
- GPU 侧还可能出现 CUDA OOM、显存碎片、后台进程争用等异常
- `cluster uncapped` 属于最高耗时、最高中断风险任务，一次中断后从头跑代价很高
- ablation 任务有 variant 外层串行循环，若中断时不能识别“哪些 variant 已完成”，就会造成重复计算
- 当前四类实验的结果文件都是在训练完成后统一导出，一旦中途失败，前面已经完成的 round 信息没有持久化保护

因此，断点续跑的目标不是改变算法，而是在保持正式实验可比性的前提下增加工程容错能力。

## 4. 断点续跑设计边界

断点续跑必须被定义为“工程容错层”，而不是“实验配置层”或“算法层”改造。边界必须保持如下冻结：

- 不改变标准样本量加权 FedAvg
- 不改变模型结构
- 不改变 client 划分
- 不改变 train / val / test split
- 不改变 seed
- 不改变 tensor 数据入口
- 不改变指标计算
- 不把 checkpoint 提交到 Git
- checkpoint 只写入 `results/` 下的输出目录

额外强调：

- resume 不是“允许换配置重跑”
- resume 不是“自动从更轻配置切回正式配置”
- resume 不是“自动覆盖旧目录”
- resume 只能在原实验配置一致的前提下恢复到未完成 round 继续执行

## 5. checkpoint 粒度设计

### 5.1 grid_cell main

建议粒度：

- 每个 federated round 结束后保存一次 checkpoint

原因：

- `grid_cell main` 没有 variant 外层循环，最自然的恢复边界就是 round
- round 结束时全局模型已经完成一次标准 FedAvg 聚合，恢复语义最清晰

### 5.2 grid_cell ablation

建议粒度：

- 每个 variant 内，每个 federated round 结束后保存 checkpoint
- 已完成 variant 在 resume 时可以直接跳过

原因：

- ablation 有两层进度：`variant` 和 `round`
- 只按整次实验保存 checkpoint 太粗，中途失败会重复整套 variant
- 需要能识别：
  - 当前正在进行哪个 variant
  - 当前 variant 已完成到第几轮
  - 哪些 variant 已全部完成

### 5.3 cluster main

建议粒度：

- 每个 federated round 结束后保存一次 checkpoint

原因：

- `cluster main` 虽然没有 variant 外层循环，但单个 round 的成本更高
- 尤其在 capped / uncapped 之间，uncapped 更需要 round 级恢复

### 5.4 cluster ablation

建议粒度：

- 每个 variant 内，每个 federated round 结束后保存 checkpoint
- 已完成 variant 在 resume 时可以直接跳过

原因：

- 这是最重的一类任务：`cluster` 样本更大，`ablation` 变体更多
- 必须支持“variant 级跳过 + round 级恢复”的双层恢复语义

### 5.5 推荐实施顺序

建议按以下顺序分批实现：

1. `single_intersection_client`
2. `single_intersection_ablation`
3. `region_client`
4. `region_ablation`

原因：

- `single_intersection_client` 结构最直，先把 round 级 checkpoint 机制走通
- `single_intersection_ablation` 再补 variant 外层状态机
- `region_client` 复用同一套 round 恢复机制，但要把 partition / split 信息一起纳入校验
- `region_ablation` 最复杂，留到最后

## 6. checkpoint 内容设计

建议 checkpoint 至少保存以下字段：

- `global_model_state_dict`
- `completed_round`
- `next_round`
- `metrics_so_far`
- `run_config`
- `config_hash`
- `selected_clients`
- `seed`
- `tensor_path`
- `train/val/test split` 信息
- `timestamp`
- `best_validation_metric`，如已有
- `python / torch / cuda` 环境摘要

进一步建议按实验类型增加：

- main 任务：
  - `convergence_history_so_far`
  - `completed_rounds`
- ablation 任务：
  - `variant_name`
  - `completed_variants`
  - `variant_histories_so_far`
  - `variant_metrics_so_far`

### 6.1 optimizer state 是否需要保存

根据当前静态审计：

- `common/client.py` 的 `FedClient.train()` 每次调用都会重新构建 `Adam`
- `sic_core.py` 中 `train_local_model()` 也是每次调用都新建 `Adam`
- 当前 optimizer 并不跨 federated round 复用

因此在“federated round 边界恢复”的设计下：

- 可以不保存 optimizer state
- 恢复时只需要恢复 `global_model_state_dict`、`completed_round` 和相关进度元数据即可

这也是当前设计最稳妥的原因：恢复语义直接对齐到“下一轮从全局模型继续开始”

如果未来某条训练线改成了“optimizer 跨 round 持久复用”，那时才需要把 optimizer state 一并纳入 checkpoint；但按当前代码结构，不需要。

## 7. resume 参数设计

本阶段只设计参数语义，不实现。

建议新增以下 CLI 参数：

- `--resume`
- `--checkpoint-dir`
- `--checkpoint-every-round`
- `--allow-overwrite`

### 7.1 `--resume`

建议语义：

- 从 `output_dir/checkpoints/latest.pt` 恢复
- 默认 `False`
- 当启用时，程序先读取 checkpoint，再做配置一致性校验

### 7.2 `--checkpoint-dir`

建议语义：

- 默认值为 `output_dir/checkpoints`
- 允许用户在不改变正式结果目录的前提下调整 checkpoint 子目录
- 一般不建议脱离 `output_dir` 单独放到别处，以免增加对齐风险

### 7.3 `--checkpoint-every-round`

建议语义：

- 默认启用
- 每轮结束后保存一次 round checkpoint
- 后续如果确实需要，可以再扩展成整数型间隔参数，但第一阶段不建议复杂化

### 7.4 `--allow-overwrite`

建议语义：

- 默认 `False`
- 当 `output_dir` 已存在且未指定 `--resume` 时，直接停止并报错
- 只有显式指定 `--allow-overwrite` 时，才允许覆盖既有目录

设计理由：

- 当前代码默认 `prepare_output_dir()` 会直接复用旧目录，这是正式实验的风险点
- 在引入 resume 之前，应先补齐“已有目录但未 resume 时拒绝覆盖”的安全边界

## 8. 配置一致性校验

resume 前必须做严格配置一致性校验。至少以下配置项必须一致：

1. `num_clients`
2. `selected_clients`
3. `rounds` 必须大于已完成 round
4. `local_epochs`
5. `batch_size`
6. `sequence_length`
7. `seed`
8. `tensor_path`
9. `train/val/test split`
10. `model variant`
11. `data_mode`

对于 cluster 任务，还建议补充：

- `partition_method`
- `max_samples_per_client_split`
- `use_active_regions_only`
- `regions_path`

对于 ablation 任务，还建议补充：

- `variants` 全列表
- 当前恢复的 `variant_name`
- 已完成 `completed_variants`

### 8.1 不一致时的行为

如果任一关键配置不一致，必须：

- 停止运行并报错
- 不自动覆盖
- 不自动从头跑
- 不自动切换配置

推荐错误信息应明确指出：

- 哪个字段不一致
- checkpoint 中的值是什么
- 当前 CLI / config 中的值是什么

## 9. 输出目录和 Git 边界

建议新增或使用以下文件：

- `output_dir/checkpoints/latest.pt`
- `output_dir/checkpoints/round_0001.pt`
- `output_dir/checkpoints/round_0002.pt`
- `output_dir/checkpoints/checkpoint_manifest.json`
- `output_dir/resume_log.jsonl`

对于 ablation，建议按 variant 分层：

- `output_dir/checkpoints/<variant_name>/latest.pt`
- `output_dir/checkpoints/<variant_name>/round_0001.pt`
- `output_dir/checkpoints/<variant_name>/checkpoint_manifest.json`

### 9.1 `checkpoint_manifest.json` 建议内容

- checkpoint 文件列表
- 已完成 round
- 当前 latest 指向
- config hash
- 最后更新时间
- 对 ablation 还要写已完成 variant 列表

### 9.2 `resume_log.jsonl` 建议内容

每次恢复时追加一行，记录：

- 恢复时间
- 恢复来源 checkpoint
- 上次完成 round
- 本次计划继续的 next round
- 配置一致性校验结果
- Python / torch / cuda 环境摘要

### 9.3 Git 边界

- checkpoint 只写入 `results/` 下
- checkpoint 不提交 Git
- `checkpoint_manifest.json` 与 `resume_log.jsonl` 也不提交 Git
- 后续若实现，需要同步确认 `.gitignore` 已覆盖这些结果目录

## 10. 分批实施建议

建议按以下四步分批落地：

1. 先实现 `single_intersection_client`
2. 再实现 `single_intersection_ablation`
3. 再实现 `region_client`
4. 最后实现 `region_ablation`

### 10.1 第一步：`single_intersection_client`

目标：

- 在 `run_fedavg_experiment()` 中加入每轮结束 checkpoint
- 在 `run_experiment()` 前加入 output_dir overwrite / resume 保护
- 首先把单层 round 恢复机制做通

### 10.2 第二步：`single_intersection_ablation`

目标：

- 在 variant 外层循环中加入 variant 级进度管理
- 支持跳过已完成 variant
- 支持在某个 variant 的某个 round 继续恢复

### 10.3 第三步：`region_client`

目标：

- 复用 main round 级 checkpoint 逻辑
- 补充 cluster 侧 partition / split / cap 参数的一致性校验

### 10.4 第四步：`region_ablation`

目标：

- 复用 region main 的 round 级恢复能力
- 再叠加 variant 外层状态机
- 这是最复杂的一步，放在最后

## 11. 风险与测试建议

断点续跑应分阶段落地，每一阶段先做 smoke，不直接跑正式 full。

建议至少覆盖以下测试：

- 正常从头跑
- 中断后 resume
- 配置不一致时拒绝 resume
- `output_dir` 已存在但未 `resume` 时拒绝覆盖
- resume 后指标文件可读

### 11.1 第一阶段 smoke 建议

先对 `single_intersection_client` 做一个极小配置 smoke：

- 少量 clients
- 少量 rounds
- 中途手动停止
- 再用 `--resume` 恢复

只验证：

- 能否正确读取 latest checkpoint
- 是否从下一轮继续
- 最终 `convergence_history.csv` 和 `main_metrics.csv` 是否仍可读

### 11.2 第二阶段 smoke 建议

对 `single_intersection_ablation` 做小规模 smoke，重点验证：

- variant 内 round 恢复
- 已完成 variant 跳过
- manifest 与 resume log 是否正确记录

### 11.3 cluster 线风险提示

- `cluster uncapped` 不应作为 resume 机制的第一条验证线
- 应先在 capped 或更轻配置下验证 checkpoint 逻辑
- 避免把“恢复机制 bug”和“uncapped 资源风险”混在一起排查

## 12. 本阶段结论

当前代码结构非常适合在“federated round 边界”引入 checkpoint：

- FedAvg 聚合边界清晰
- optimizer 不跨 round 复用
- ablation 的 variant 外层循环也已显式存在
- 结果写出统一集中，便于后续把中间状态和最终导出分层管理

但当前系统还缺少三类关键工程保护：

- checkpoint / resume 机制本身
- `output_dir` 已存在时的拒绝覆盖逻辑
- resume 前的配置一致性校验

因此，下一阶段最合理的做法不是一次性改四条线，而是先从 `single_intersection_client` 开始分批实现，再逐步推广到 `ablation` 与 `cluster`。
