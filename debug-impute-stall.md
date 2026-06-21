# [OPEN] Debug Session: impute-stall / [进行中] 调试会话：impute-stall

## Context / 背景

**中文：**

- 任务：继续执行 `snh_mix` 插补流程，参数为 `--stage impute --resume true`
- 现象：进程仍在运行，但 `imp_data` 与 `progress` 计数停止变化
- 范围：`results\rdm_exp\scenarios\snh_mix`

**English:**

- Task: continue `snh_mix` imputation with `--stage impute --resume true`
- Symptom: the process stays alive, but `imp_data` and `progress` counts stop changing
- Scope: `results\rdm_exp\scenarios\snh_mix`

## Current Evidence / 当前证据

**中文：**

- `5%` 缺失率下，8 种方法均已完成：`61/61`
- `10%` 缺失率下，8 种方法均处于部分完成：`17/61`
- `20%` 与 `30%` 目录下尚未创建方法子目录
- 后台命令保持存活，但终端没有新增输出
- 观察 60 秒后，没有新增 `.parquet` 或 `.done.json`

**English:**

- `5%` is complete for all 8 methods: `61/61`
- `10%` is partial for all 8 methods: `17/61`
- `20%` and `30%` do not have method directories yet
- The background command stays alive without new terminal output
- A 60-second observation window shows no new `.parquet` or `.done.json`

## Hypotheses / 假设

**中文：**

- H1：某个特定 chunk 卡住，原因可能是某种方法进入极慢分支
- H2：断点续跑扫描或文件存在性检查被损坏/半写入结果阻塞
- H3：进程卡在 parquet 读写 I/O，而不是 CPU 侧计算
- H4：进程卡在 `10%` 缺失率、`017+` chunk 的高成本相关性或拓扑计算
- H5：后台进程虽未退出，但因为未处理的等待状态而没有实际前进

**English:**

- H1: the process is blocked on a specific chunk because one method enters an extremely slow branch
- H2: resume scanning or file existence checks are hanging on a corrupted or partially written output
- H3: the process is waiting on I/O for a parquet read/write operation instead of CPU-side computation
- H4: the process is stuck in a high-cost correlation/topology computation for `10%` chunk `017+`
- H5: the background process is alive but not making forward progress because of an unhandled runtime wait state

## Plan / 调试计划

**中文：**

- 仅增加插桩，不先改逻辑
- 在 chunk 和 method 边界增加定向日志后重现问题
- 基于运行时证据确认精确卡点
- 收集到证据后再做最小修复

**English:**

- Add instrumentation only
- Reproduce with targeted logs around chunk and method boundaries
- Confirm the exact stall point from runtime evidence
- Apply a minimal fix only after evidence is collected
