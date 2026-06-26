# Batch 1 正式输出目录 .gitignore 边界修复报告

## 1. 本阶段范围

本阶段只修复 `.gitignore` 对正式真实数据实验输出目录的覆盖，不运行 Batch 1、不运行训练、不运行 profiling、不修改训练代码、不修改 LaTeX、不修改 `simulation_experiments`、不修改 conda 环境。

本阶段也不生成新的 `results`，不创建 `results/real_data_experiments/formal/grid_cell_main_full_cuda`，只处理忽略规则边界与对应文档说明。

## 2. 问题来源

上一阶段在 Batch 1 运行前检查中发现：

- 目标输出目录：`results/real_data_experiments/formal/grid_cell_main_full_cuda`
- `Test-Path` 返回 `False`
- `git check-ignore` 未命中
- 因此按中止规则停止，未执行 Batch 1

也就是说，虽然 Batch 1 的正式命令与参数已经静态确认完毕，但正式输出目录所在的 `formal/` 路径当时未被 `.gitignore` 覆盖，若直接运行，`results/` 有进入 Git 状态的风险。

## 3. 修复内容

本次只在 `.gitignore` 中新增：

```gitignore
# Real-data formal experiment outputs
results/real_data_experiments/formal/
```

新增位置放在现有 `results/real_data_experiments/` 相关忽略规则附近，未删除任何已有规则，也未扩大到不必要的目录范围。

## 4. 验证结果

本次修复后执行了以下检查：

- `git check-ignore -v --no-index results/real_data_experiments/formal/grid_cell_main_full_cuda`
- `git check-ignore -v --no-index results/real_data_experiments/formal/grid_cell_main_full_cuda/dummy.txt`

命中结果为：

```text
.gitignore:263:results/real_data_experiments/formal/    results/real_data_experiments/formal/grid_cell_main_full_cuda
.gitignore:263:results/real_data_experiments/formal/    results/real_data_experiments/formal/grid_cell_main_full_cuda/dummy.txt
```

说明新增规则已经覆盖 Batch 1 正式输出目录及其目录下文件路径，后续如果按该路径运行 Batch 1，`results/real_data_experiments/formal/` 将按预期被 Git 忽略。

## 5. 提交边界

本阶段只允许提交以下两项：

- `.gitignore`
- 本报告 `real_data_experiments/batch1_output_gitignore_fix_zh.md`

本阶段不提交以下内容：

- `results/`
- `.pt` / `.npy` / `parquet` / `png` / `pdf`
- 训练代码
- LaTeX 文件
- `simulation_experiments`
- 环境文件

## 6. 后续建议

本次忽略边界修复提交后，可以重新生成并执行 Batch 1：`grid_cell main full`，`device=cuda` 的正式运行指令。

下一次进入 Batch 1 执行阶段时，建议继续沿用以下边界：

- 先做 Git clean 检查；
- 再做 CUDA 与 `nvidia-smi` 核验；
- 继续使用 `results/real_data_experiments/formal/grid_cell_main_full_cuda` 作为正式输出目录；
- 运行后只提交报告文档，不提交 `results/`。
