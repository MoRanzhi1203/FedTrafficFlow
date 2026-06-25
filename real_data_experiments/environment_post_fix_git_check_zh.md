# 环境修复后 Git 提交前核验报告

## 1. 本阶段范围

本阶段只检查环境修复完成后的 Git 修改范围与报告文档状态，不运行训练、不修改训练代码、不执行 commit、不执行 push。

## 2. 当前环境状态

| 项目 | 状态 |
|---|---|
| 环境路径 | `E:\anaconda3\envs\FedTrafficFlow` |
| Python 解释器 | `E:\anaconda3\envs\FedTrafficFlow\python.exe` |
| PYTHONNOUSERSITE | `1` |
| ENABLE_USER_SITE | `False` |
| torch | `2.8.0+cpu` |
| cuda_available | `False` |
| Jupyter kernel | `Python (FedTrafficFlow)` |
| smoke test | `passed` |

## 3. Git 状态摘要

### `git status --short`

```text
?? env_backup/
?? real_data_experiments/compute_time_estimation_i7_3060_zh.md
?? real_data_experiments/environment_isolation_fix_report_zh.md
?? real_data_experiments/environment_path_check_report_zh.md
?? real_data_experiments/profile_tensor_experiments.py
?? results/real_data_experiments/compute_time_profile/
```

### `git diff --stat`

```text
(无输出；当前状态以未跟踪文件为主)
```

### `git diff --name-status`

```text
(无输出；当前状态以未跟踪文件为主)
```

说明：

- 当前 Git 变化不是“已跟踪文件的修改 diff”，而是“多个未跟踪文件/目录”。
- `git diff -- real_data_experiments/environment_isolation_fix_report_zh.md` 为空，是因为该文件当前为未跟踪文件，不是已跟踪文件上的增量修改。

## 4. 建议提交文件

建议本阶段仅考虑提交与环境修复记录直接相关的文档文件：

- `real_data_experiments/environment_isolation_fix_report_zh.md`
- `real_data_experiments/environment_post_fix_git_check_zh.md`
- `real_data_experiments/environment_path_check_report_zh.md`

说明：

- `environment_isolation_fix_report_zh.md` 已准确记录管理员权限修复、`PYTHONNOUSERSITE=1` 持久化、`ENABLE_USER_SITE=False`、`python/pip/site-packages` 指向目标环境、依赖安装、Jupyter kernel、smoke test 和 `torch` CPU-only 状态。
- `environment_path_check_report_zh.md` 作为前序问题定位证据，也可以与最终修复报告一并保留。

## 5. 不建议提交文件

以下文件或目录不建议在本阶段提交：

- `env_backup/`
- `results/`
- `results/real_data_experiments/pip_env_admin_fixed_smoke/`
- `results/real_data_experiments/compute_time_profile/`
- 本地环境快照、pip/conda 列表、ACL 检查输出等临时核验文件
- `real_data_experiments/profile_tensor_experiments.py`（属于训练/实验脚本文件，不属于本阶段环境文档提交范围）

额外说明：

- `real_data_experiments/compute_time_estimation_i7_3060_zh.md` 当前也是未跟踪文档，但与本阶段“环境修复提交前核验”不是同一批最小提交范围，建议单独判断是否需要提交。

## 6. 风险检查

本次检查结果：

- 训练代码修改：发现 `real_data_experiments/profile_tensor_experiments.py` 为未跟踪文件，属于风险项，本阶段不建议提交
- LaTeX 修改：未发现
- `simulation_experiments` 修改：未发现
- tensor 数据修改：未发现
- `results` 目录待提交：发现，属于本地产物，不建议提交
- `env_backup` 待提交：发现，属于本地核验产物，不建议提交
- `.pt / .npy / .parquet` 大文件待提交：当前 `git status --short` 未发现

## 7. ignore 规则检查

### `git check-ignore -v env_backup/FedTrafficFlow_isolated_pip_freeze.txt`

```text
(无输出)
```

结论：

- 当前 `.gitignore` 没有覆盖 `env_backup/`，因此 `env_backup/` 会作为未跟踪文件进入 Git 状态。

### `git check-ignore -v results/real_data_experiments/pip_env_admin_fixed_smoke/`

```text
.gitignore:255:results/real_data_experiments/*_smoke/   results/real_data_experiments/pip_env_admin_fixed_smoke/
```

结论：

- `results/real_data_experiments/*_smoke/` 已被 `.gitignore` 正确覆盖。
- 但 `results/real_data_experiments/compute_time_profile/` 当前仍在未跟踪列表中，说明它不在现有忽略规则覆盖范围内，本阶段只报告，不擅自修改 `.gitignore`。

## 8. 是否可以进入提交阶段

结论：

- 可以进入文档提交阶段。
- 但必须严格限制提交范围，只提交明确的环境修复文档，不要提交 `env_backup/`、`results/` 或训练脚本文件。

## 9. 推荐 commit message

```text
docs: record FedTrafficFlow environment isolation fix
```

## 10. 后续建议

1. 提交前不要使用 `git add .`
2. 只 `git add` 明确的文档文件
3. CUDA 版 PyTorch 后续单独处理
4. 当前环境可作为 CPU 运行环境
