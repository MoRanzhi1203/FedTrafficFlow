# 推送前最终核验报告

## 1. 本阶段范围

本阶段只做推送前核验，不运行训练、不运行 profiling、不修改代码、不 push。

## 2. 当前工作区状态

- `git status --short`

```text
(empty)
```

- `git status`

```text
On branch main
Your branch is ahead of 'origin/main' by 4 commits.
  (use "git push" to publish your local commits)

nothing to commit, working tree clean
```

结论：

- 当前工作区干净；
- 未发现未跟踪文件、未暂存修改或已暂存但未提交修改。

## 3. 最近提交链

- `git log -5 --oneline`

```text
41742f2 (HEAD -> main) tools: add tensor experiment profiling wrapper
ca25756 docs: update compute time estimate environment details
4dcbfaf chore: ignore local environment and profiling artifacts
d4a3f7d docs: record FedTrafficFlow environment isolation fix
6784e40 (origin/main, origin/HEAD) docs: update experiment documentation and audit reports, update .gitignore
```

核验结果：

- 最近 4 个本地 commit 与预期一致；
- 当前本地 `main` 位于远端 `origin/main` 之前 4 个 commit。

## 4. 最近 4 个 commit 文件范围

- `git diff --name-status HEAD~4..HEAD`

```text
M       .gitignore
A       real_data_experiments/compute_time_estimation_i7_3060_zh.md
A       real_data_experiments/environment_isolation_fix_report_zh.md
A       real_data_experiments/environment_path_check_report_zh.md
A       real_data_experiments/environment_post_fix_git_check_zh.md
A       real_data_experiments/profile_tensor_experiments.py
A       real_data_experiments/profile_tensor_experiments_review_zh.md
A       real_data_experiments/untracked_artifact_governance_zh.md
```

- `git diff --stat HEAD~4..HEAD`

```text
 .gitignore                                         |   4 +
 .../compute_time_estimation_i7_3060_zh.md          | 407 +++++++++
 .../environment_isolation_fix_report_zh.md         | 150 ++++
 .../environment_path_check_report_zh.md            | 134 +++
 .../environment_post_fix_git_check_zh.md           | 105 +++
 .../profile_tensor_experiments.py                  | 900 +++++++++++++++++++++
 .../profile_tensor_experiments_review_zh.md        | 108 +++
 .../untracked_artifact_governance_zh.md            | 107 +++
 8 files changed, 1915 insertions(+)
```

说明：

- 最近 4 个 commit 只包含环境修复文档、治理文档、耗时估算文档、profiling 工具脚本、审查报告以及 `.gitignore`；
- 未发现 `env_backup/`、`results/`、`data/processed/`、大文件、LaTeX 或 `simulation_experiments/` 文件进入这 4 个 commit。

## 5. 风险文件检查

`git diff --name-only HEAD~4..HEAD` 输出为：

```text
.gitignore
real_data_experiments/compute_time_estimation_i7_3060_zh.md
real_data_experiments/environment_isolation_fix_report_zh.md
real_data_experiments/environment_path_check_report_zh.md
real_data_experiments/environment_post_fix_git_check_zh.md
real_data_experiments/profile_tensor_experiments.py
real_data_experiments/profile_tensor_experiments_review_zh.md
real_data_experiments/untracked_artifact_governance_zh.md
```

辅助检查：

- `git diff --name-only HEAD~4..HEAD | findstr /I "\.pt$ \.npy$ \.parquet$ \.png$ \.pdf$ \.zip$ \.7z$ \.tar$"` 无输出

| 风险项 | 是否出现 | 说明 |
|---|---|---|
| `env_backup/` | 否 | 未进入最近 4 个 commit |
| `results/` | 否 | 未进入最近 4 个 commit |
| `data/processed/` | 否 | 未进入最近 4 个 commit |
| `.pt / .npy` | 否 | 扩展名检查无输出 |
| `.png / .pdf` | 否 | 扩展名检查无输出 |
| LaTeX 修改 | 否 | 最近 4 个 commit 中未出现相关文件 |
| `simulation_experiments` 修改 | 否 | 最近 4 个 commit 中未出现相关文件 |
| 正式训练产物 | 否 | 未出现训练输出目录或大结果文件 |

## 6. .gitignore 生效情况

- `git check-ignore -v env_backup/FedTrafficFlow_isolated_pip_freeze.txt`

```text
.gitignore:232:env_backup/      env_backup/FedTrafficFlow_isolated_pip_freeze.txt
```

- `git check-ignore -v results/real_data_experiments/compute_time_profile/`

```text
.gitignore:256:results/real_data_experiments/compute_time_profile/      results/real_data_experiments/compute_time_profile/
```

- `git check-ignore -v results/real_data_experiments/compute_time_profile/hardware_summary.json`

```text
.gitignore:256:results/real_data_experiments/compute_time_profile/      results/real_data_experiments/compute_time_profile/hardware_summary.json
```

结论：

- `env_backup/` 已被 `.gitignore` 正确忽略；
- `results/real_data_experiments/compute_time_profile/` 及其内部 profiling 输出已被 `.gitignore` 正确忽略。

## 7. 远端与分支状态

- `git remote -v`

```text
origin  git@github.com:MoRanzhi1203/FedTrafficFlow.git (fetch)
origin  git@github.com:MoRanzhi1203/FedTrafficFlow.git (push)
```

- `git branch --show-current`

```text
main
```

- `git status -sb`

```text
## main...origin/main [ahead 4]
```

记录：

- 当前分支：`main`
- 远端地址：`git@github.com:MoRanzhi1203/FedTrafficFlow.git`
- 本地状态：ahead of `origin/main`
- ahead commit 数：`4`

## 8. 是否可以 push

结论：

- **可以进入 push 阶段**

原因：

- 工作区干净；
- 最近 4 个 commit 范围正确；
- 未发现 `env_backup/`、`results/`、`data/processed/`、大文件、LaTeX、`simulation_experiments/` 或正式训练产物进入提交；
- `.gitignore` 对本地产物的忽略规则仍然生效；
- 当前只是本地比远端领先 4 个 commit，适合在作者确认后统一推送。

## 9. 推荐 push 命令

如果作者确认可以推送，建议命令为：

```bash
git push
```

如果后续发现当前分支未绑定 upstream，再考虑：

```bash
git push -u origin main
```

但本阶段未执行 push。
