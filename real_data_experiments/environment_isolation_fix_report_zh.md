# FedTrafficFlow 环境隔离修复报告

## 1. 本阶段目标

本阶段聚焦于修复 `FedTrafficFlow` 环境中的 `pip` / user-site 污染，并进一步确认：

- 临时隔离是否成功；
- `PYTHONNOUSERSITE=1` 是否能够持久化到 conda 环境；
- `E:\anaconda3\envs\FedTrafficFlow` 是否具备正常写权限；
- `requirements.txt` 是否能够安装到环境内部；
- 关键库、Jupyter kernel 和 smoke test 是否可以在隔离环境下继续完成。

## 2. 修复前问题
- conda 激活路径正确：`E:\anaconda3\envs\FedTrafficFlow`
- Python 指向 FedTrafficFlow：`E:\anaconda3\envs\FedTrafficFlow\python.exe`
- 但 pip 和关键库来自：`C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages`
- 环境隔离被破坏，存在用户目录污染风险

## 3. 修复操作

- 以管理员身份打开 `Anaconda PowerShell Prompt`；
- 使用 `cmd /c icacls "E:\anaconda3\envs\FedTrafficFlow" /grant "%USERNAME%:(OI)(CI)M" /T` 为当前用户递归授予 `Modify` 权限；
- 对 `E:\anaconda3\envs\FedTrafficFlow`、`conda-meta`、`Lib\site-packages` 执行写入测试；
- 成功执行 `conda env config vars set -p E:\anaconda3\envs\FedTrafficFlow PYTHONNOUSERSITE=1`；
- 重新激活环境后，验证 `PYTHONNOUSERSITE=1` 与 `ENABLE_USER_SITE=False`；
- 再次验证 `python`、`pip` 和 `site-packages` 已全部回到目标环境内部。

## 4. 路径核验

| 项目 | 修复后结果 | 是否正确 |
| --- | --- | --- |
| python executable | `E:\anaconda3\envs\FedTrafficFlow\python.exe` | yes |
| sys.prefix | `E:\anaconda3\envs\FedTrafficFlow` | yes |
| python -m pip --version | `pip 25.2 from E:\anaconda3\envs\FedTrafficFlow\lib\site-packages\pip (python 3.9)` | yes |
| site.getsitepackages | `["E:\\anaconda3\\envs\\FedTrafficFlow", "E:\\anaconda3\\envs\\FedTrafficFlow\\lib\\site-packages"]` | yes |
| ENABLE_USER_SITE | `False` | yes |

说明：

- 当前 `python`、`python -m pip` 和 `site-packages` 都已回到 `FedTrafficFlow` 环境内部。
- 这次结果来自管理员终端重新激活后的实测，不再依赖仅限单会话的临时变量。

## 5. 隔离持久化状态

| 项目 | 结果 | 是否完成 |
|---|---|---|
| 临时设置 `PYTHONNOUSERSITE=1` | 成功 | yes |
| `conda env config vars set -p E:\anaconda3\envs\FedTrafficFlow PYTHONNOUSERSITE=1` | 成功 | yes |
| 重新激活后自动生效 | 已确认成功 | yes |
| 重新激活后核验结果 | `PYTHONNOUSERSITE=1`，`ENABLE_USER_SITE=False` | yes |

说明：

- 当前已经不是“仅临时隔离成功”，而是“隔离已成功持久化”。
- `usersite` 路径仍会被打印出来，但由于 `ENABLE_USER_SITE=False`，该用户目录不会参与当前解释器的包解析。

## 6. 环境目录写权限

| 路径 | ACL 结果 | 写权限判断 |
|---|---|---|
| `E:\anaconda3\envs\FedTrafficFlow` | 已递归授予当前用户 `Modify` 权限 | 可写 |
| `E:\anaconda3\envs\FedTrafficFlow\conda-meta` | 写入测试成功 | 可写 |
| `E:\anaconda3\envs\FedTrafficFlow\Lib\site-packages` | 写入测试成功 | 可写 |

说明：

- 管理员终端中执行的 `icacls` 已成功处理 `6675` 个文件，未出现失败项。
- 三处写测试均成功创建 `__write_test.txt`，说明原环境的核心写入阻塞已经解除。

## 7. 关键库来源路径

| 库 | 状态 | 来源路径 | 是否属于 FedTrafficFlow |
| --- | --- | --- | --- |
| numpy | installed | `E:\anaconda3\envs\FedTrafficFlow\lib\site-packages\numpy\__init__.py` | yes |
| pandas | installed | `E:\anaconda3\envs\FedTrafficFlow\lib\site-packages\pandas\__init__.py` | yes |
| scipy | installed | `E:\anaconda3\envs\FedTrafficFlow\lib\site-packages\scipy\__init__.py` | yes |
| sklearn | installed | `E:\anaconda3\envs\FedTrafficFlow\lib\site-packages\sklearn\__init__.py` | yes |
| matplotlib | installed | `E:\anaconda3\envs\FedTrafficFlow\lib\site-packages\matplotlib\__init__.py` | yes |
| torch | installed | `E:\anaconda3\envs\FedTrafficFlow\lib\site-packages\torch\__init__.py` | yes |
| tqdm | installed | `E:\anaconda3\envs\FedTrafficFlow\lib\site-packages\tqdm\__init__.py` | yes |

**说明**：

- 上述关键库均来自 `E:\anaconda3\envs\FedTrafficFlow\lib\site-packages`，未再引用用户目录或其他环境路径。
- `torch` 当前版本为 `2.8.0+cpu`，`torch.version.cuda = None`，`cuda_available = False`，说明目前安装的是 CPU-only 版本。

## 8. requirements.txt 安装结果

- 早期曾因权限不足而失败一次。
- 在管理员权限修复和隔离持久化完成后，`requirements.txt` 中列出的核心依赖已能够在目标环境中正确导入。
- 另外，考虑到项目代码实际依赖，后续又单独补装了 `tqdm`、`torch` 和 `ipykernel`。
- 当前未见库路径回落到 `C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages` 的情况。

## 9. Jupyter kernel

`ipykernel` 已安装成功，且自定义 kernel 已按目标名称注册完成。

当前状态：

- 已执行：
  `python -m ipykernel install --user --name FedTrafficFlow --display-name "Python (FedTrafficFlow)"`
- 输出：
  `Installed kernelspec FedTrafficFlow in C:\Users\MSIPC\AppData\Roaming\jupyter\kernels\fedtrafficflow`
- `jupyter kernelspec list` 显示：
  - `python3`
  - `analysis`
  - `blender_4.0.2`
  - `fedtrafficflow`
  - `reptile`

结论：

- `python -m pip install --no-user ipykernel` 已成功；
- `python -m ipykernel install --user --name FedTrafficFlow --display-name "Python (FedTrafficFlow)"` 已成功；
- `jupyter kernelspec list` 中的目录名显示为小写 `fedtrafficflow`，这在 Windows/Jupyter 环境下是正常的，不影响 display name 为 `Python (FedTrafficFlow)`。

## 10. smoke test

已执行并通过。

执行命令：

`python real_data_experiments/profile_tensor_experiments.py --setting grid_cell --task main --device cpu --num-clients 3 --rounds 1 --local-epochs 1 --batch-size 32 --sequence-length 12 --output-dir results/real_data_experiments/pip_env_admin_fixed_smoke`

执行结果：

- 输出日志：`[profiling] completed -> results\real_data_experiments\pip_env_admin_fixed_smoke`
- 输出目录：`results/real_data_experiments/pip_env_admin_fixed_smoke`

## 11. CUDA 状态说明

本阶段未主动修复 CUDA；当前 `torch` 为 `2.8.0+cpu`，若后续需要 GPU profiling，需要单独处理 CUDA 版 PyTorch。

## 12. 结论

当前结论更新为：

- 原环境 `E:\anaconda3\envs\FedTrafficFlow` 的目录权限已通过管理员终端修复；
- `PYTHONNOUSERSITE=1` 已成功持久化；
- `python`、`pip` 和 `site-packages` 已全部指向目标环境内部；
- `requirements.txt` 相关核心依赖与补充缺失依赖已安装并完成路径核验；
- `Jupyter kernel` 已注册完成；
- `smoke test` 已通过；
- 当前环境已可作为项目默认 CPU 运行环境使用。

建议的后续路径：

- 在 `Trae` / `VS Code` / `Jupyter` 中将解释器切换为：
  `E:\anaconda3\envs\FedTrafficFlow\python.exe`
- 若后续需要 GPU 能力，再单独处理 CUDA 版 PyTorch。
