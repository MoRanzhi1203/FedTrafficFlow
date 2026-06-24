# 提交前最终 Diff 核验报告

## 1. 本阶段范围

- 本阶段只做最终 diff 核验与提交前风险检查。
- 本阶段不运行训练，不运行正式实验。
- 本阶段不修改训练代码，不修改 LaTeX，不修改 `simulation_experiments/`。
- 本阶段不重新生成 `node_flow_grid_tensor.pt`，不生成 `6.池化网格张量.pt`。
- 本阶段不执行 `git commit`，不执行 `git push`，不执行 `git rm --cached`。

## 2. Git 状态摘要

### 2.1 `git status --short`

```text
 M .gitignore
 M real_data_experiments/RUN_REAL_EXPERIMENTS_zh.md
 M real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md
 M real_data_experiments/data_entry_audit_zh.md
 M real_data_experiments/real_experiment_report.md
 M real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md
 M real_data_experiments/region_client/region_notebook_migration_zh.md
 M real_data_experiments/tensor_only_experiment_plan_zh.md
?? real_data_experiments/pre_commit_audit_zh.md
```

### 2.2 `git diff --stat`

```text
 .gitignore                                         | 15 +++++++++
 real_data_experiments/RUN_REAL_EXPERIMENTS_zh.md   |  8 +++++
 .../RUN_TENSOR_ONLY_EXPERIMENTS_zh.md              | 38 ++++++++++++++++++++--
 real_data_experiments/data_entry_audit_zh.md       | 12 +++++--
 real_data_experiments/real_experiment_report.md    | 30 +++++++++++------
 .../region_ablation_notebook_migration_zh.md       | 19 ++++++++++-
 .../region_client/region_notebook_migration_zh.md  | 19 ++++++++++-
 .../tensor_only_experiment_plan_zh.md              | 30 +++++++++++++----
 8 files changed, 147 insertions(+), 24 deletions(-)
```

### 2.3 `git diff --name-status`

```text
M       .gitignore
M       real_data_experiments/RUN_REAL_EXPERIMENTS_zh.md
M       real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md
M       real_data_experiments/data_entry_audit_zh.md
M       real_data_experiments/real_experiment_report.md
M       real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md
M       real_data_experiments/region_client/region_notebook_migration_zh.md
M       real_data_experiments/tensor_only_experiment_plan_zh.md
```

### 2.4 摘要结论

- 当前已修改文件共 `8` 个，均为 `.md` 文档。
- 当前有 `1` 个未跟踪文件：`real_data_experiments/pre_commit_audit_zh.md`。
- 当前尚未出现 `.py`、`.tex`、`simulation_experiments/`、数据文件或结果目录的工作区修改。
- 当前修改范围集中在文档与 `.gitignore`，符合本阶段限制。

## 3. 文件分类

| 类别 | 文件 | 建议 |
|---|---|---|
| 文档修改 | `real_data_experiments/real_experiment_report.md` | 建议提交 |
| 文档修改 | `real_data_experiments/tensor_only_experiment_plan_zh.md` | 建议提交 |
| 文档修改 | `real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md` | 建议提交 |
| 文档修改 | `real_data_experiments/RUN_REAL_EXPERIMENTS_zh.md` | 建议提交 |
| 文档修改 | `real_data_experiments/data_entry_audit_zh.md` | 建议提交 |
| 文档修改 | `real_data_experiments/region_client/region_notebook_migration_zh.md` | 建议提交 |
| 文档修改 | `real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md` | 建议提交 |
| .gitignore 修改 | `.gitignore` | 建议提交 |
| 新增审计文件 | `real_data_experiments/pre_commit_audit_zh.md` | 建议纳入提交 |
| 新增审计文件 | `real_data_experiments/pre_commit_final_diff_check_zh.md` | 建议纳入提交 |
| 代码修改 | 无 | 当前无代码 diff |
| 数据或结果文件修改 | 无 | 当前工作区未发现 `.pt` / `.npy` / `results/` diff |
| 可疑文件 | 无 | 当前未发现额外未跟踪结果文件或数据文件 |

## 4. 关键 Diff 核验

### 4.1 已修改文件

- `.gitignore`：新增真实数据实验本地结果目录与大图产物忽略规则。
- `real_data_experiments/real_experiment_report.md`：把区域实验状态统一为“已完成迁移并通过 smoke test”，新增 smoke 非正式结果说明。
- `real_data_experiments/tensor_only_experiment_plan_zh.md`：新增“当前状态更新”，把旧阶段范围保留为历史语境，并补充两类客户端设置关系。
- `real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md`：新增 `region_client` / `region_ablation` smoke 命令与正式训练占位计划。
- `real_data_experiments/RUN_REAL_EXPERIMENTS_zh.md`：新增当前状态更新与 smoke 非正式结果说明。
- `real_data_experiments/data_entry_audit_zh.md`：新增当前状态更新，替换过时阶段范围表述。
- `real_data_experiments/region_client/region_notebook_migration_zh.md`：新增“当前迁移状态”，明确已迁移、默认输入、默认聚合和 smoke 非正式结果。
- `real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md`：新增“当前迁移状态”，明确已迁移、默认输入、默认聚合和 smoke 非正式结果。

### 4.2 未跟踪 / 未修改 / 不存在

- `real_data_experiments/pre_commit_audit_zh.md`：未跟踪，建议纳入提交。
- `real_data_experiments/pre_commit_final_diff_check_zh.md`：本轮新生成，未跟踪，建议纳入提交。
- 当前未发现其他 `real_data_experiments/*.md` 之外的未跟踪风险文件。

## 5. 未跟踪文件核验

### 5.1 `git ls-files --others --exclude-standard`

```text
real_data_experiments/pre_commit_audit_zh.md
```

### 5.2 结论

- 建议纳入提交：`real_data_experiments/pre_commit_audit_zh.md`
- 本轮生成报告后，还应一并纳入：
  `real_data_experiments/pre_commit_final_diff_check_zh.md`
- 未发现未跟踪的 `results/`、`data/processed/`、`*.pt`、`*.npy`、`*.parquet`、`*.png`、`*.pdf`。
- 结论：当前未跟踪文件风险较低，主要是审计文档本身。

## 6. .gitignore 核验

### 6.1 规则存在性

- 已确认 `.gitignore` 已包含：
  `data/processed/node_flow_grid/`
- 已追加以下规则：

```gitignore
# Real data experiment local results
results/real_data_experiments/*_smoke/
results/real_data_experiments/single_intersection_client_tensor/
results/real_data_experiments/single_intersection_ablation_tensor/
results/real_data_experiments/single_region_client_tensor_*/
results/real_data_experiments/single_region_ablation_tensor_*/
results/real_data_experiments/region_client_tensor_*/
results/real_data_experiments/region_ablation_tensor_*/

# Large generated experiment artifacts
results/real_data_experiments/**/*.png
results/real_data_experiments/**/*.pdf
results/real_data_experiments/**/prediction_samples.csv
results/real_data_experiments/**/convergence_history.csv
```

### 6.2 路径核验结论

- `data/processed/node_flow_grid/final_sum_mean_standard/node_flow_grid_tensor.pt`
  当前不在工作区 diff 中，且其父目录已被 `.gitignore` 覆盖，不建议提交。
- `results/real_data_experiments/region_client_tensor_smoke/`
  已有忽略规则，但该目录在历史上已被 Git 跟踪过。
- `results/real_data_experiments/region_ablation_tensor_smoke/`
  已有忽略规则，但该目录在历史上已被 Git 跟踪过。
- `results/real_data_experiments/single_intersection_client_tensor/`
  已有忽略规则，但历史结果目录是否全部取消跟踪仍需单独治理。
- `results/real_data_experiments/single_intersection_ablation_tensor/`
  已有忽略规则，但历史结果目录是否全部取消跟踪仍需单独治理。

### 6.3 风险说明

- 若路径已经被 Git 跟踪，`.gitignore` 不会自动取消跟踪。
- 如需清理历史已跟踪结果目录，后续需作者确认后再执行 `git rm --cached`。
- 本阶段未执行 `git rm --cached`。

## 7. 文档状态核验

### 7.1 旧阶段表述搜索

- 已搜索：
  `区域实验正在迁移`
  `区域客户端正在迁移`
  `本阶段未迁移区域实验`
  `当前不迁移区域实验`
  `未迁移 region_client`
  `未迁移 region_ablation`
  `不迁移 region_client`
  `不迁移 region_ablation`
  `尚未完成 region_client`
  `尚未完成 region_ablation`
  `区域 notebook 正在迁移到`
- 当前 `real_data_experiments/*.md` 中未检出上述旧阶段表述作为当前状态出现。
- 结论：当前没有发现“区域实验仍在迁移中”的现行文档风险。

### 7.2 smoke test 非正式结果声明

- 已确认以下关键文件均包含明确声明：
  - `real_data_experiments/real_experiment_report.md`
  - `real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md`
  - `real_data_experiments/region_client/region_notebook_migration_zh.md`
  - `real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md`
- 额外已覆盖：
  - `real_data_experiments/RUN_REAL_EXPERIMENTS_zh.md`
  - `real_data_experiments/data_entry_audit_zh.md`
  - `real_data_experiments/tensor_only_experiment_plan_zh.md`
- 结论：已统一写明 smoke test 结果不作为论文正式结果。

## 8. FedAvg 主线核验

- 已检查：
  - `real_data_experiments/region_client/rc_core.py`
  - `real_data_experiments/region_ablation/ra_core.py`
  - `real_data_experiments/single_intersection_client/sic_core.py`
  - `real_data_experiments/single_intersection_ablation/sia_core.py`
- 未发现以下关键词混入默认主流程：
  - `FedProx`
  - `server damping`
  - `personalization`
  - `loss-weighted`
  - `data-loss weighted`
  - `Proposed aggregation`
  - `similarity-aware`
  - `quality-weighted`
- 结论：未发现 `FedProx` / `server damping` / `personalization` 等混入默认主流程。

## 9. 历史文件核验

- 已检查 `6.池化网格张量.pt`。
- 未发现该文件存在。
- 结论：当前不存在 `6.池化网格张量.pt` 的误生成风险。

## 10. 是否可以提交

- 可以提交。
- 当前 diff 仅集中在文档与 `.gitignore`，未发现训练代码、实验代码、LaTeX、`simulation_experiments/`、正式 tensor 数据或结果目录的工作区修改。
- 提交前仅需作者确认是否一并纳入两份审计文档：
  - `real_data_experiments/pre_commit_audit_zh.md`
  - `real_data_experiments/pre_commit_final_diff_check_zh.md`

## 11. 推荐提交范围

- `.gitignore`
- `real_data_experiments/real_experiment_report.md`
- `real_data_experiments/tensor_only_experiment_plan_zh.md`
- `real_data_experiments/RUN_TENSOR_ONLY_EXPERIMENTS_zh.md`
- `real_data_experiments/RUN_REAL_EXPERIMENTS_zh.md`
- `real_data_experiments/data_entry_audit_zh.md`
- `real_data_experiments/region_client/region_notebook_migration_zh.md`
- `real_data_experiments/region_ablation/region_ablation_notebook_migration_zh.md`
- `real_data_experiments/pre_commit_audit_zh.md`
- `real_data_experiments/pre_commit_final_diff_check_zh.md`

## 12. 推荐 commit message

```text
docs: align tensor-only real-data experiment status
```

## 13. 仍需作者确认

- 是否将 `real_data_experiments/pre_commit_audit_zh.md` 纳入本次提交。
- 是否将 `real_data_experiments/pre_commit_final_diff_check_zh.md` 纳入本次提交。
- 是否后续单独治理已被 Git 跟踪的 `results/real_data_experiments/*` 历史结果目录。
- 是否在完成 `git add` / `git commit` 后，再单独确认是否执行 `git push`。
- 是否同步把两类真实数据实验的正式名称统一为“网格单元级客户端联邦学习设置 / 簇级客户端联邦学习设置”。
