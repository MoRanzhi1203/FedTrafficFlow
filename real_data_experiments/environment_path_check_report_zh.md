# FedTrafficFlow 环境安装路径核验报告

## 1. 本阶段目标

本阶段只检查 `FedTrafficFlow` conda 环境下的 Python / pip / site-packages 路径是否属于目标环境，不安装依赖，不修改训练代码，不运行正式训练。

## 2. 当前激活环境

| 项目 | 检查结果 |
|---|---|
| conda active environment | `FedTrafficFlow` |
| active env location | `E:\anaconda3\envs\FedTrafficFlow` |
| CONDA_PREFIX | `E:\anaconda3\envs\FedTrafficFlow` |
| CONDA_DEFAULT_ENV | `FedTrafficFlow` |

说明：

- `conda info --envs` 显示当前激活环境前缀为 `*`，对应路径为 `E:\anaconda3\envs\FedTrafficFlow`。
- `conda info` 中 `active env location` 也指向 `E:\anaconda3\envs\FedTrafficFlow`。

## 3. Python 路径检查

| 项目 | 检查结果 | 是否正确 |
|---|---|---|
| where python | `E:\anaconda3\envs\FedTrafficFlow\python.exe`；`C:\Users\MSIPC\AppData\Local\Microsoft\WindowsApps\python.exe` | yes |
| sys.executable | `E:\anaconda3\envs\FedTrafficFlow\python.exe` | yes |
| sys.prefix | `E:\anaconda3\envs\FedTrafficFlow` | yes |
| Python version | `3.9.23` | yes |

说明：

- 当前 `python` 解释器本身来自目标环境。
- `WindowsApps` 中的 `python.exe` 仍在系统 `PATH` 中，但没有排在当前激活环境解释器之前。

## 4. pip 路径检查

| 项目 | 检查结果 | 是否正确 |
|---|---|---|
| where pip | `E:\anaconda3\envs\FedTrafficFlow\Scripts\pip.exe`；`E:\anaconda3\Scripts\pip.exe` | no |
| python -m pip --version | `pip 26.0.1 from C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages\pip (python 3.9)` | no |
| pip --version | `pip 26.0.1 from C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages\pip (python 3.9)` | no |

说明：

- 虽然 `where pip` 能看到目标环境内的 `pip.exe`，但也同时暴露了 `base` 下的 `pip.exe`，路径存在混杂风险。
- 更关键的是，`python -m pip --version` 和 `pip --version` 都指向 `C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages\pip`，这不属于 `FedTrafficFlow` 环境自己的 `Lib\site-packages`。

## 5. site-packages 检查

| 项目 | 路径 | 是否正确 |
|---|---|---|
| site.getsitepackages | `E:\anaconda3\envs\FedTrafficFlow`；`E:\anaconda3\envs\FedTrafficFlow\lib\site-packages` | yes |
| usersite | `C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages` | no |
| ENABLE_USER_SITE | `True` | no |

说明：

- 目标环境自己的 `site-packages` 路径是存在的。
- 但用户站点包目录已启用，且当前 `usersite` 指向 `C:` 用户目录，这会破坏项目环境隔离。

## 6. pip user install 配置

`python -m pip config list` 未返回显式配置项，未直接发现 `global.user = true`、`user.user = true` 或 `install.user = true`。

但本次核验中仍观察到以下隔离风险：

- `python -m pip --version` 指向用户站点包目录中的 `pip` 模块。
- 多个关键库来源路径位于 `C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages`。

因此，即使没有显式 `user=true` 配置，当前环境依然存在明显的 user-site 污染风险。

## 7. 关键库来源路径

| 库 | 状态 | 来源路径 | 是否属于 FedTrafficFlow |
|---|---|---|---|
| numpy | installed | `C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages\numpy\__init__.py` | no |
| pandas | installed | `C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages\pandas\__init__.py` | no |
| torch | not installed | `NOT INSTALLED` | no |
| sklearn | installed | `C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages\sklearn\__init__.py` | no |
| matplotlib | installed | `C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages\matplotlib\__init__.py` | no |
| scipy | installed | `C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages\scipy\__init__.py` | no |
| tqdm | installed | `C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages\tqdm\__init__.py` | no |

说明：

- 当前核验到的关键库均未落在 `E:\anaconda3\envs\FedTrafficFlow\Lib\site-packages`。
- `torch` 当前未安装。
- 未发现这些关键库来自 `base` 或其他 conda 环境；主要风险来自 `C:` 用户目录的用户站点包。

## 8. 风险判断

结论：

- 路径存在风险，不建议继续安装，需要先修正环境激活或 pip user/site-packages 污染问题。

具体风险包括：

- `python` 来自目标环境，但 `python -m pip` 的模块来源不在目标环境内。
- 关键库来源路径集中在 `C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages`。
- `ENABLE_USER_SITE = True`，当前环境未实现项目级隔离。

在该状态下继续安装依赖，容易把库继续装到用户目录，而不是 `FedTrafficFlow` 环境自己的 `Lib\site-packages`。

## 9. 是否需要隔离

项目依赖必须隔离安装在：

`E:\anaconda3\envs\FedTrafficFlow\Lib\site-packages`

不得安装到以下位置：

- `E:\anaconda3\Lib\site-packages`
- `C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages`
- 其他 conda 环境路径

当前环境不满足上述隔离要求，需要先修正后再继续安装。

## 10. 后续建议

当前不建议继续执行依赖安装。建议先修正以下问题：

- 修正 `python -m pip` 的模块来源，使其回到 `E:\anaconda3\envs\FedTrafficFlow\Lib\site-packages\pip`。
- 排查为什么当前环境会优先加载 `C:\Users\MSIPC\AppData\Roaming\Python\Python39\site-packages` 中的 `pip` 与关键库。
- 在确认 `python -m pip --version` 与关键库来源都回到目标环境后，再执行依赖安装。

只有在路径修正完成后，才建议使用以下命令继续安装：

`python -m pip install --no-user -r requirements.txt`

补充信息：

- `PYTHONPATH`、`PYTHONHOME`、`PIP_USER` 当前未见显式环境变量值。
- `CONDA_PREFIX` 与 `CONDA_DEFAULT_ENV` 指向正确的目标环境。
- `conda config --show envs_dirs` 中同时包含 `E:\anaconda3\envs` 和用户目录下的 conda 环境路径，但当前 active env location 仍是正确的 `E:` 盘环境。
