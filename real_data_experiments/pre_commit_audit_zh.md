# 提交前审计与文件清理建议

## 1. 审计范围

- 本阶段只做提交前审计与提交清单规划。
- 本阶段未执行 `git commit`，未执行 `git push`。
- 本阶段未修改 LaTeX。
- 本阶段未修改 `simulation_experiments/`。
- 本阶段未改变标准 `FedAvg` 主线。

## 2. Git 状态审计

### 2.1 `git status --short`

- 审计时执行 `git status --short`，输出为空。
- 结论：审计开始时工作区是干净的，没有待提交文件。

### 2.2 按要求分类

#### A. 应提交的代码文件

- 当前 `git status --short` 为空，因此当前没有“待提交中的代码文件”。
- 若后续需要重新整理“区域客户端 notebook 到 py 迁移”这一阶段的提交范围，建议纳入以下代码文件：
  - `real_data_experiments/common/region_partition.py`
  - `real_data_experiments/common/region_tensor_dataset.py`
  - `real_data_experiments/region_client/rc_config.py`
  - `real_data_experiments/region_client/rc_core.py`
  - `real_data_experiments/region_client/rc_visualization.py`
  - `real_data_experiments/region_ablation/ra_config.py`
  - `real_data_experiments/region_ablation/ra_core.py`
  - `real_data_experiments/region_ablation/ra_visualization.py`

#### B. 应提交的文档文件

- 当前 `git status --short` 为空，因此当前没有“待提交中的文档文件”。
- 若后续需要重新整理本阶段文档提交范围，建议纳入：
  - `real_data_experiments/real_experiment_report.md`
  - `real_data_experiments/RUN_REAL_EXPERIMENTS_zh.md`
  - `real_data_experiments/tensor_only_experiment_plan_zh.md`
  - `real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md`
  - `real_data_experiments/region_client/README_zh.md`
  - `real_data_experiments/region_client/region_notebook_migration_zh.md`
  - `real_data_experiments/region_client/historical_notes_zh.md`
  - `real_data_experiments/region_ablation/README_zh.md`
  - `real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md`
  - `real_data_experiments/region_ablation/historical_notes_zh.md`
  - `real_data_experiments/pre_commit_audit_zh.md`

#### C. 可选提交的小型审计文件

- 当前 `git status --short` 为空，因此当前没有“待提交中的小型审计文件”。
- 若作者希望把固定 region 方案与审计链路一并保留，可选纳入：
  - `real_data_experiments/selected_regions_fixed_plan.csv`

#### D. 不建议提交的数据/结果大文件

- 当前 `git status --short` 为空，因此当前没有“待提交中的大文件”。
- 但仓库中已经存在大量已跟踪结果文件，不建议在后续同类提交中继续纳入：
  - `results/real_data_experiments/region_client_tensor_smoke/`
  - `results/real_data_experiments/region_ablation_tensor_smoke/`
  - `results/real_data_experiments/single_intersection_client_tensor/`
  - `results/real_data_experiments/single_intersection_ablation_tensor/`
  - `results/real_data_experiments/**/*.png`
  - `results/real_data_experiments/**/*.csv`
  - `results/real_data_experiments/**/*.json`
- 全仓库已跟踪的大型结果文件统计：
  - `*.pt = 0`
  - `*.npy = 0`
  - `*.parquet = 3`
  - `*.pkl = 0`
  - `*.joblib = 0`
  - `*.png = 264`
  - `*.pdf = 200`
  - `results/* = 7236`
  - `results/real_data_experiments/* = 112`
- 其中两组区域 smoke 目录已经被 Git 跟踪，这不利于后续保持提交简洁。

#### E. 建议加入 `.gitignore` 的文件或目录

- 当前 `.gitignore` 已包含 `data/processed/node_flow_grid/`，因此正式 tensor 数据目录目前未被 Git 跟踪。
- 当前已按治理建议在 `.gitignore` 中追加 `results/real_data_experiments/*_smoke/` 及相关真实数据实验结果目录规则。
- 新追加的忽略规则不会自动取消 Git 已跟踪文件，历史结果目录若已入库，仍需后续作者单独确认是否清理缓存索引。
- 本轮追加的典型规则包括：
  - `results/real_data_experiments/region_client_tensor_smoke/`
  - `results/real_data_experiments/region_ablation_tensor_smoke/`
  - `results/real_data_experiments/single_intersection_client_tensor/`
  - `results/real_data_experiments/single_intersection_ablation_tensor/`
- 若作者决定今后不再把真实数据实验运行产物入库，也可继续沿用：
  - `results/real_data_experiments/**/*.png`
  - `results/real_data_experiments/**/*.pdf`
  - `results/real_data_experiments/**/prediction_samples.csv`
  - `results/real_data_experiments/**/convergence_history.csv`
  - `results/real_data_experiments/**/run_config.json`
  - `results/real_data_experiments/**/split_summary.json`
- 本轮只追加忽略规则，不执行 `git rm --cached`。

## 3. 大文件与结果目录检查

### 3.1 正式数据目录

- `data/processed/node_flow_grid/` 当前未被 Git 跟踪。
- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt` 未被 Git 跟踪。
- 结论：正式 tensor 数据入口仍保持本地数据工件，不在版本库内。

### 3.2 结果目录

- `results/` 目录下已有大量已跟踪结果文件。
- `results/real_data_experiments/region_client_tensor_smoke/` 已被 Git 跟踪。
- `results/real_data_experiments/region_ablation_tensor_smoke/` 已被 Git 跟踪。
- 结论：从“后续提交整洁性”角度看，不建议继续提交 smoke test 结果目录。

### 3.3 文件类型检查结论

- 未发现 `.pt` 待提交或已跟踪文件。
- 未发现 `.npy` 待提交或已跟踪文件。
- 发现少量已跟踪 `.parquet` 文件，不属于本阶段真实数据 tensor 主线，后续不建议继续增加同类结果文件。
- 发现大量已跟踪 `.png` / `.pdf` 结果图，后续建议避免继续扩大此类提交范围。

## 4. 历史文件检查

- 已执行等价搜索检查 `*6.池化网格张量.pt*`。
- 未发现 `6.池化网格张量.pt`。
- 结论：当前未误生成该历史命名文件。

## 5. 标准 FedAvg 主线检查

### 5.1 关键词风险检查范围

- 已检查：
  - `real_data_experiments/region_client/rc_core.py`
  - `real_data_experiments/region_ablation/ra_core.py`
  - `real_data_experiments/single_intersection_client/sic_core.py`
  - `real_data_experiments/single_intersection_ablation/sia_core.py`

### 5.2 检查结论

- 未在上述四个默认主流程核心文件中发现以下关键词进入默认执行路径：
  - `FedProx`
  - `server damping`
  - `personalization`
  - `loss-weighted`
  - `data-loss weighted`
  - `Proposed aggregation`
  - `similarity-aware`
  - `quality-weighted`
- `rc_core.py` 与 `ra_core.py` 均复用公共 `FedClient` / `run_federated_rounds` / 标准 `FedAvg` 训练链路。
- 上述关键词目前主要出现在：
  - `historical_notes_zh.md`
  - notebook 迁移审计文档
  - 总体说明文档
- 结论：默认训练主流程仍保持标准样本量加权 `FedAvg`，未发现非主线聚合被混入默认执行路径。

## 6. 区域客户端定义检查

### 6.1 检查文件

- `real_data_experiments/region_client/rc_core.py`
- `real_data_experiments/region_ablation/ra_core.py`
- `real_data_experiments/common/region_partition.py`
- `real_data_experiments/common/region_tensor_dataset.py`

### 6.2 检查结论

- `region_client` / `region_ablation` 当前定义正确：
  - 每个 client = 一组 pooled grid regions。
- `single_intersection_client` / `single_intersection_ablation` 在说明文档中也已明确：
  - 每个 client = 1 个 pooled grid region。
- `RegionClientWindowDataset` 以 `region_ids` 列表构造一个 client 的样本窗口，语义与“簇级客户端设置中的多 grid-cell client”一致。
- `assign_region_clients()` 输出 `client_region_ids`，语义清晰。
- 本次检查未发现区域客户端定义错误。

### 6.3 文档状态说明

- 当前两类真实数据实验定义已统一：
  - `single_intersection_*`：网格单元级客户端联邦学习设置，每个 client = 1 个 pooled grid region。
  - `region_*`：簇级客户端联邦学习设置，每个 client = 一组 pooled grid regions。
- 本轮已同步修正文档状态描述，避免再把区域客户端实验写成“迁移中”。

## 7. Smoke Test 标记检查

### 7.1 已明确写明 smoke 不是论文正式结果

- `real_data_experiments/tensor_only_experiment_plan_zh.md`

### 7.2 已补充直接声明的文件

- `real_data_experiments/real_experiment_report.md`
- `real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md`
- `real_data_experiments/region_client/region_notebook_migration_zh.md`
- `real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md`

### 7.3 结论

- 相关文档已统一补充：
  “当前 smoke test 结果仅用于验证代码链路、输出文件和可视化流程，不作为论文正式结果。”

## 8. 提交建议

## 建议提交

### 代码

- `real_data_experiments/common/region_partition.py`
- `real_data_experiments/common/region_tensor_dataset.py`
- `real_data_experiments/region_client/rc_config.py`
- `real_data_experiments/region_client/rc_core.py`
- `real_data_experiments/region_client/rc_visualization.py`
- `real_data_experiments/region_ablation/ra_config.py`
- `real_data_experiments/region_ablation/ra_core.py`
- `real_data_experiments/region_ablation/ra_visualization.py`

### 文档

- `real_data_experiments/real_experiment_report.md`
- `real_data_experiments/RUN_REAL_EXPERIMENTS_zh.md`
- `real_data_experiments/tensor_only_experiment_plan_zh.md`
- `real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md`
- `real_data_experiments/region_client/README_zh.md`
- `real_data_experiments/region_client/region_notebook_migration_zh.md`
- `real_data_experiments/region_client/historical_notes_zh.md`
- `real_data_experiments/region_ablation/README_zh.md`
- `real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md`
- `real_data_experiments/region_ablation/historical_notes_zh.md`
- `real_data_experiments/pre_commit_audit_zh.md`

### 小型配置/CSV

- 可选：`real_data_experiments/selected_regions_fixed_plan.csv`
- 不建议把 `results/` 目录下的 `run_config.json`、`split_summary.json`、`figure_index.csv` 作为本阶段小型配置文件提交。

## 不建议提交

- `data/processed/node_flow_grid/**/*.pt`
- `data/processed/node_flow_grid/**/*.npy`
- `results/real_data_experiments/region_client_tensor_smoke/`
- `results/real_data_experiments/region_ablation_tensor_smoke/`
- `results/real_data_experiments/single_intersection_client_tensor/`
- `results/real_data_experiments/single_intersection_ablation_tensor/`
- `results/real_data_experiments/**/*.png`
- `results/real_data_experiments/**/*.pdf`
- `results/real_data_experiments/**/prediction_samples.csv`
- `results/real_data_experiments/**/convergence_history.csv`
- 任何仅用于 smoke test 连通性验证的结果目录与图表产物

## 建议加入 `.gitignore`

- `results/real_data_experiments/region_client_tensor_smoke/`
- `results/real_data_experiments/region_ablation_tensor_smoke/`
- `results/real_data_experiments/single_intersection_client_tensor/`
- `results/real_data_experiments/single_intersection_ablation_tensor/`
- 若作者确认“真实数据实验结果不再入库”，再考虑加入：
  - `results/real_data_experiments/**/*.png`
  - `results/real_data_experiments/**/*.pdf`
  - `results/real_data_experiments/**/prediction_samples.csv`
  - `results/real_data_experiments/**/convergence_history.csv`
  - `results/real_data_experiments/**/run_config.json`
  - `results/real_data_experiments/**/split_summary.json`

## 9. 审计结论摘要

- 当前待提交状态：无。
- 当前默认主流程风险：未发现 `FedProx` / `server damping` / `personalization` 混入默认执行路径。
- 当前区域客户端定义：正确。
- 当前最大提交清理风险：`results/` 已长期被跟踪，尤其是 `results/real_data_experiments/*_smoke/`。
- 当前文档风险：主要转为历史文本与已跟踪结果目录的治理问题，当前状态描述已基本统一。

## 10. 需要作者确认

- 是否在下一轮提交前，把 `results/real_data_experiments/region_client_tensor_smoke/` 与 `results/real_data_experiments/region_ablation_tensor_smoke/` 从后续提交范围中排除。
- 是否把 `results/real_data_experiments/single_intersection_client_tensor/` 与 `results/real_data_experiments/single_intersection_ablation_tensor/` 也视为本地结果目录，不再继续纳入提交。
- 是否接受后续修改 `.gitignore`，把真实数据实验 smoke / 结果目录统一忽略。
- 是否需要在下一轮文档整理中继续沿用统一的 smoke 非正式结果声明。
- 是否需要同步清理或重构已跟踪的 `results/` 历史产物策略。

## 11. 本阶段说明

- 本阶段只完成提交前审计和文件清理建议。
- 未执行 `git commit`。
- 未执行 `git push`。
- 未修改 LaTeX。
- 未修改 `simulation_experiments/`。
- 未改变标准 `FedAvg` 主线。

## 12. 修正后状态

- 已修正 `real_experiment_report.md` 中将区域客户端实验描述为“尚处于迁移阶段”的旧表述。
- 已在相关文档中补充统一声明：smoke test 结果仅用于连通性与输出链路验证，不作为论文正式结果。
- 已在 `.gitignore` 中追加真实数据实验本地结果目录与大图产物忽略规则，但未取消跟踪历史已入库文件。
- 已明确两类真实数据实验：
  - `single_intersection_*`：网格单元级客户端联邦学习设置，每个 client = 1 个 pooled grid region；
  - `region_*`：簇级客户端联邦学习设置，每个 client = 一组 pooled grid regions。
