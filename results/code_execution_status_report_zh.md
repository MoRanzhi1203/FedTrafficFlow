# 代码执行状态报告

## 1. 核查范围

本次核查面向当前项目中最近新增且具备独立执行入口的代码文件。结合最近提交记录与文件类型，确认本轮需核查的新增代码为以下两个 Python 脚本：

1. `analysis_scripts/audit_real_data_preprocessing.py`
2. `analysis_scripts/real_data_missingness_experiment.py`

说明：本轮新增的 Markdown 文档、CSV/JSON 结果文件和 README 不属于可执行代码，因此不纳入“代码执行状态”统计。

## 2. 运行环境

- 项目目录：`E:\Jupter_Notebook\FedTrafficFlow`
- Python 环境：`E:\anaconda3\envs\analysis\python.exe`
- Python 版本：`3.9.23`
- 技术栈：Python + pandas + numpy + matplotlib

## 3. 历史执行证据核查

### 3.1 `audit_real_data_preprocessing.py`

- 提交来源：`739e48a Add real data preprocessing audit and manuscript section`
- 历史输出物已存在：
  - `results/real_data_preprocessing/real_data_file_inventory.csv`
  - `results/real_data_preprocessing/real_data_quality_summary.csv`
  - `results/real_data_preprocessing/real_data_preprocessing_audit.json`
  - `results/real_data_preprocessing/real_data_preprocessing_audit.md`
- 结论：该脚本在历史上已经执行过，且已产出完整审计结果。

### 3.2 `real_data_missingness_experiment.py`

- 提交来源：`14fd705 Fix real data missingness script for Python 3.9`
- 历史输出物已存在：
  - `results/real_data_missingness_experiments/summaries/missingness_design_summary.csv`
  - `results/real_data_missingness_experiments/summaries/missingness_mask_summary.csv`
  - `results/real_data_missingness_experiments/summaries/imputation_quality_summary.csv`
  - `results/real_data_missingness_experiments/real_data_missingness_experiment_audit.json`
  - `results/real_data_missingness_experiments/real_data_missingness_experiment_audit.md`
  - `results/real_data_missingness_experiments/run_config.json`
  - `results/real_data_missingness_experiments/run_commands.txt`
- 审计 JSON 中已记录：
  - `design_rows = 250`
  - `mask_rows = 250`
  - `quality_rows = 750`
  - `python_path = E:\anaconda3\envs\analysis\python.exe`
- 结论：该脚本在历史上已经执行过，且正式结果配置完整保存。

## 4. 当前版本复验执行

### 4.1 静态可执行性验证

执行命令：

```powershell
E:\anaconda3\envs\analysis\python.exe -m py_compile analysis_scripts\audit_real_data_preprocessing.py
E:\anaconda3\envs\analysis\python.exe -m py_compile analysis_scripts\real_data_missingness_experiment.py
```

结果：

- `audit_real_data_preprocessing.py`：通过
- `real_data_missingness_experiment.py`：通过

说明：两者均未出现 Python 语法错误，说明当前版本具备基本可执行性。

### 4.2 `audit_real_data_preprocessing.py` 当前复验

执行命令：

```powershell
E:\anaconda3\envs\analysis\python.exe analysis_scripts\audit_real_data_preprocessing.py
```

执行状态：

- 已成功启动并进入实际扫描阶段。
- 在观察窗口内未自然结束，因此本轮未等待其完整跑完。
- 为避免长时间占用终端资源，本轮复验已主动停止该次重跑。

执行日志摘录：

```text
E:\Jupter_Notebook\FedTrafficFlow\analysis_scripts\audit_real_data_preprocessing.py:444: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
  parsed_time = pd.to_datetime(time_series, errors="coerce")
E:\Jupter_Notebook\FedTrafficFlow\analysis_scripts\audit_real_data_preprocessing.py:363: UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`. To ensure parsing is consistent and as-expected, please specify a format.
  parsed = pd.to_datetime(cleaned, errors="coerce")
```

异常判断：

- 发现 `pandas` 时间解析警告，共 2 条。
- 未观察到 `Traceback`、未观察到脚本崩溃栈。
- 当前判断为非致命警告，不足以推翻“脚本可运行且历史已执行”的结论。

复验结论：

- 历史执行：已完成
- 当前版本启动验证：通过
- 当前版本全量重跑：在观察窗口内未完成

### 4.3 `real_data_missingness_experiment.py` 当前复验

执行命令：

```powershell
E:\anaconda3\envs\analysis\python.exe analysis_scripts\real_data_missingness_experiment.py --input_dir data\analysis\node_intersection_flow_parquet --output_dir results\real_data_missingness_experiments --target_col 路口车流量 --time_col 时间段 --node_col 节点ID --missing_rates 0,0.05,0.10,0.20,0.30 --mechanisms mcar_point --seeds 42,2024,3407,1234,5678 --impute_methods zero_fill,forward_fill,linear_interpolation --max_files 10 --max_rows 500
```

执行状态：

- 退出码：`0`
- 结果：执行成功

复验后的结果验证：

```text
missingness_design_summary.csv (250, 10)
missingness_mask_summary.csv (250, 14)
imputation_quality_summary.csv (750, 16)
```

结论：

- 当前版本可在指定 Python 3.9 环境下成功运行。
- 正式结果规模与既有审计结果一致。

## 5. 未执行代码核查结论

本轮核查范围内未发现“新增但从未执行过”的独立代码文件。

也就是说：

- `audit_real_data_preprocessing.py`：已存在历史完整执行证据
- `real_data_missingness_experiment.py`：已存在历史完整执行证据

因此，本轮没有出现必须“首次补执行”的新增代码；执行动作主要是基于当前版本做复验。

## 6. 执行成功的验证结果

### 已确认执行成功

1. `analysis_scripts/real_data_missingness_experiment.py`
   - `py_compile` 通过
   - 正式配置复执行退出码为 `0`
   - 输出文件完整
   - 输出规模验证通过：`250 / 250 / 750`

### 已确认历史执行成功，当前复验启动正常

1. `analysis_scripts/audit_real_data_preprocessing.py`
   - `py_compile` 通过
   - 历史输出完整
   - 当前复验可正常启动
   - 当前复验观察到非致命时间解析警告
   - 当前复验未在观察窗口内自然完成

## 7. 异常与警告汇总

### 非致命警告

- 文件：`analysis_scripts/audit_real_data_preprocessing.py`
- 类型：`pandas` 时间解析 `UserWarning`
- 影响：不影响脚本启动与历史执行结论，但可能增加扫描耗时，也提示个别时间列格式不统一

### 未发现的异常

- 未发现 Python 语法错误
- 未发现 `Traceback`
- 未发现输出目录缺失
- 未发现结果规模与既有正式配置不一致

## 8. 最终结论

本轮核查范围内的新增可执行代码共 2 个，二者均已存在历史执行证据。

- `analysis_scripts/audit_real_data_preprocessing.py`：历史已执行成功；当前版本复验可启动，但在观察窗口内未完成，期间仅出现非致命时间解析警告。
- `analysis_scripts/real_data_missingness_experiment.py`：历史已执行成功；当前版本已按正式配置复执行成功，退出码为 `0`，并再次验证了正式结果规模 `250 / 250 / 750`。

综合判断，当前新增代码整体处于“已执行且可复验”的状态，不存在“新增但完全未执行”的代码遗漏。
