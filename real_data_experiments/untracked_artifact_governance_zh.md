# 未跟踪文件治理报告

## 1. 本阶段范围

本阶段只治理未跟踪文件与 `.gitignore`，不运行训练、不处理 CUDA、不修改训练逻辑。

## 2. 初始未跟踪项

治理前的主要未跟踪项为：

- `env_backup/`
- `real_data_experiments/compute_time_estimation_i7_3060_zh.md`
- `real_data_experiments/profile_tensor_experiments.py`
- `results/real_data_experiments/compute_time_profile/`

上一条提交已确认是：

`d4a3f7d docs: record FedTrafficFlow environment isolation fix`

## 3. .gitignore 修正

本阶段在 `.gitignore` 中补充了以下规则：

```gitignore
env_backup/
results/real_data_experiments/compute_time_profile/
results/real_data_experiments/profile_*/
results/real_data_experiments/**/*_smoke/
```

说明：

- 未新增对 `real_data_experiments/*.md` 的忽略规则；
- 未新增对 `real_data_experiments/*.py` 的忽略规则；
- 这样可以继续单独审查 `compute_time_estimation_i7_3060_zh.md` 与 `profile_tensor_experiments.py`，而不会误伤文档和脚本本身。

## 4. 本地产物处理结论

| 路径 | 类型 | 处理建议 |
|---|---|---|
| `env_backup/` | 本地环境备份 | 不提交，加入 `.gitignore` |
| `results/real_data_experiments/compute_time_profile/` | profiling 输出 | 不提交，加入 `.gitignore` |

核验结果：

- `git check-ignore -v env_backup/FedTrafficFlow_isolated_pip_freeze.txt` 已命中 `.gitignore:232:env_backup/`
- `git check-ignore -v results/real_data_experiments/compute_time_profile/hardware_summary.json` 已命中 `.gitignore:256:results/real_data_experiments/compute_time_profile/`
- `git ls-files env_backup` 无输出
- `git ls-files results/real_data_experiments/compute_time_profile` 无输出

结论：

- `env_backup/` 与 `results/real_data_experiments/compute_time_profile/` 当前均未被 Git 跟踪；
- 本阶段无需执行 `git rm --cached`；
- 这些目录现在已成功从候选提交范围中剔除。

## 5. 文档审查结论

审查对象：`real_data_experiments/compute_time_estimation_i7_3060_zh.md`

审查结论：

- 该文档整体上确实是在描述计算量、CPU profiling、GPU 理论建议和耗时估算；
- 文档明确写明了 `CUDA available = False`，也区分了“真实 CPU profiling”与“GPU 理论建议”；
- 文档没有把 smoke/profiling 结果表述为论文正式结果；
- 文档没有修改实验主线或 FedAvg 叙述。

但当前存在一个重要问题：

- 文档中的环境信息仍写为 `Python 3.12.3`、`PyTorch 2.12.0+cpu`；
- 而当前修复完成后的正式环境为 `Python 3.9`、`torch 2.8.0+cpu`。

因此本阶段结论是：

- **暂不建议直接提交 `compute_time_estimation_i7_3060_zh.md`**
- 建议先单独校正文档中的环境版本说明，再决定是否作为文档提交

## 6. Profiling 脚本审查结论

审查对象：`real_data_experiments/profile_tensor_experiments.py`

静态审查结论：

- 脚本定位是 profiling 外层包装脚本，不是训练核心逻辑文件；
- 它通过已有 `single_intersection_*` / `region_*` 实验入口调度运行，不直接改写 FedAvg 或 model core；
- 默认输出目录是：
  `results/real_data_experiments/compute_time_profile/`
- 脚本主要记录：
  - 硬件环境
  - 数据规模
  - wall time
  - split/样本数摘要
  - nvidia-smi / 进程内存等 profiling 信息
- 脚本中未发现个人绝对路径写死；
- 未发现重新生成正式 tensor 数据的逻辑；
- 未发现依赖 C 盘用户目录的逻辑。

因此本阶段结论是：

- `profile_tensor_experiments.py` 可以视为独立 profiling 工具脚本；
- 但它属于 `.py` 代码文件，不应与 `.gitignore` / 治理文档混在同一次提交里；
- **建议后续单独审查后再单独提交**

## 7. 当前 Git 状态

### `git status --short`

```text
?? real_data_experiments/compute_time_estimation_i7_3060_zh.md
?? real_data_experiments/profile_tensor_experiments.py
```

补充说明：

- `.gitignore` 当前有未提交修改；
- `env_backup/` 与 `results/real_data_experiments/compute_time_profile/` 已不再出现在未跟踪列表中。

### `git diff --stat`

```text
.gitignore | 4 ++++
1 file changed, 4 insertions(+)
```

### `git diff --name-status`

```text
M       .gitignore
```

## 8. 建议提交范围

如果本轮只提交 `.gitignore` 和治理报告，建议：

```bash
git add .gitignore
git add real_data_experiments/untracked_artifact_governance_zh.md
```

当前**不建议**额外添加：

```bash
real_data_experiments/compute_time_estimation_i7_3060_zh.md
real_data_experiments/profile_tensor_experiments.py
env_backup/
results/
```

如果后续先校正 `compute_time_estimation_i7_3060_zh.md` 的环境版本说明，再考虑追加：

```bash
git add real_data_experiments/compute_time_estimation_i7_3060_zh.md
```

## 9. 推荐 commit message

如果只提交 `.gitignore` 与治理报告，建议：

```text
chore: ignore local environment and profiling artifacts
```

如果后续连同校正后的耗时估算文档一起提交，可考虑：

```text
docs: add compute time estimation and artifact governance
```

## 10. 后续建议

1. 后续单独审查并提交 `profile_tensor_experiments.py`
2. 后续单独处理 CUDA 版 PyTorch
3. 当前 CPU 环境已可用
