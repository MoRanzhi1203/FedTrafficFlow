# CUDA 版 PyTorch 环境方案设计与风险评估

## 1. 本阶段范围

本阶段只进行 CUDA 版 PyTorch 环境方案设计与风险评估，不安装 CUDA 版 PyTorch，不卸载当前 `torch`，不修改当前 `FedTrafficFlow` conda 环境，不创建新环境，不运行训练，不运行 profiling，不修改项目代码，不执行 `git add`、`git commit` 或 `git push`。

## 2. 当前 CPU 环境状态

检查口径基于以下流程：

- 在项目目录 `E:\Jupter_Notebook\FedTrafficFlow` 下执行 `git status --short` 与 `git log -3 --oneline`。
- 通过 `conda activate E:\anaconda3\envs\FedTrafficFlow` 激活当前 CPU 环境后，执行 Python / pip / torch 状态检查。

当前结果如下：

| 项目 | 检查结果 |
|---|---|
| Git 工作区 | clean，`git status --short` 无输出 |
| 最近 3 个 commit | `462adf7 docs: add pre-push final check report`；`41742f2 tools: add tensor experiment profiling wrapper`；`ca25756 docs: update compute time estimate environment details` |
| Conda 环境路径 | `E:\anaconda3\envs\FedTrafficFlow` |
| Python 解释器 | `E:\anaconda3\envs\FedTrafficFlow\python.exe` |
| Python 版本 | `3.9.23` |
| pip 路径 | `E:\anaconda3\envs\FedTrafficFlow\lib\site-packages\pip` |
| torch 版本 | `2.8.0+cpu` |
| torch 文件路径 | `E:\anaconda3\envs\FedTrafficFlow\lib\site-packages\torch\__init__.py` |
| torch CUDA | `None` |
| cuda_available | `False` |
| `PYTHONNOUSERSITE` | `1` |
| `ENABLE_USER_SITE` | `False` |

说明：

- 当前激活后的 `FedTrafficFlow` 环境仍保持 CPU-only 状态，`cuda_available=False` 是预期结果。
- 当前激活后的环境隔离状态是正常的，`pip` 与 `torch` 都来自目标环境内部路径。
- 补充观察：如果不通过 `conda activate`，而是直接调用环境内 `python.exe`，当前 shell 不一定自动继承 `PYTHONNOUSERSITE=1`。因此后续 GPU 环境创建后，仍应坚持“激活环境后再核验”的流程，避免 user-site 污染误判。

## 3. GPU / 驱动状态

`nvidia-smi` 摘要如下：

| 项目 | 检查结果 |
|---|---|
| GPU 型号 | `NVIDIA GeForce RTX 3060 Laptop GPU` |
| Driver Version | `560.70` |
| CUDA Version | `12.6` |
| 显存 | `6144 MiB` |
| 当前显存占用 | `1188 MiB / 6144 MiB` |
| 当前 GPU 利用率 | `8%` |
| 当前进程占用情况 | 存在浏览器、QQ、微信、Trae 等图形界面进程，占用类型主要为 `C+G` |

说明：

- `nvidia-smi` 显示的 `CUDA Version = 12.6` 表示当前驱动支持的最高 CUDA runtime 版本，不等于当前 PyTorch 已具备 CUDA 能力。
- 当前 `torch=2.8.0+cpu`，因此即使机器有 NVIDIA GPU，`torch.cuda.is_available()` 仍然返回 `False`，这不是异常。
- `RTX 3060 Laptop GPU` 对后续 GPU smoke test、轻量 profiling 和中小规模训练是适合的，但显存只有 `6 GB`，后续需要控制 batch size、序列长度和并发占用，不能默认按高显存桌面卡的参数运行。

## 4. Conda 环境现状

当前 `conda env list` 与 `conda info` 显示：

| 项目 | 检查结果 |
|---|---|
| 当前 `FedTrafficFlow` 环境路径 | `E:\anaconda3\envs\FedTrafficFlow` |
| 当前 active environment | `base`，路径 `E:\anaconda3` |
| base 路径 | `E:\anaconda3` |
| envs_dirs | `E:\anaconda3\envs`；`C:\Users\MSIPC\.conda\envs`；`C:\Users\MSIPC\AppData\Local\conda\conda\envs` |
| 现有环境 | `FedTrafficFlow`、`analysis`、`blender-env`、`pyqt`、`reptile`、`shopsync`、`uav_platform`、`yx_hcr` 等 |
| 是否已有明显的 GPU / CUDA 专用环境 | 未发现命名上明确对应 `FedTrafficFlowGPU` 或专用 CUDA 训练环境 |

说明：

- 目前最自然的新环境落点是 `E:\anaconda3\envs\FedTrafficFlowGPU`，它与现有 `envs_dirs` 完全一致，后续管理成本最低。
- 如果出于磁盘配额或权限考虑，也可以放在 `E:\conda_envs\FedTrafficFlowGPU`，但这会引入额外路径管理与解释器选择成本。

## 5. 方案 A：新建单独 GPU 环境

候选路径示例：

- `E:\anaconda3\envs\FedTrafficFlowGPU`
- `E:\conda_envs\FedTrafficFlowGPU`

### 优点

- 不破坏当前已经验证通过的 CPU 环境，`FedTrafficFlow` 可以继续作为稳定保底环境。
- 即使 CUDA 版 PyTorch 安装失败，也不会影响当前项目的 CPU 可运行性与已修复的环境隔离状态。
- 可以为 GPU 环境单独注册 Jupyter kernel，例如 `Python (FedTrafficFlowGPU)`，避免 notebook 混用解释器。
- 可以将 GPU profiling、GPU smoke test、正式 GPU 训练与当前 CPU 方案明确分离，减少结果解释歧义。
- 更容易做 A/B 对照，例如 CPU 与 GPU 环境分别验证依赖、速度与显存行为。

### 风险与成本

- 需要重复安装项目依赖，并重新做一次 `PYTHONNOUSERSITE=1`、`pip` 路径、`torch` 路径和 kernel 路径核验。
- VS Code / Trae / Jupyter 需要显式切换到新的解释器或 kernel，否则可能仍然落回 CPU 环境。
- 需要维护一套额外环境，后续升级包版本时要注意 CPU/GPU 环境的差异。

### 对关键问题的影响判断

| 关注点 | 影响判断 |
|---|---|
| 是否破坏当前 CPU 环境 | 否，只要不在 `FedTrafficFlow` 中安装 CUDA 包，当前环境保持不动 |
| 是否影响 Jupyter kernel | 可以做到不影响；建议为 GPU 环境注册独立 kernel，而不是覆写现有 `fedtrafficflow` kernel |
| 是否影响 `requirements.txt` | 默认不影响；环境安装行为不会自动修改仓库中的 `requirements.txt` |
| 是否影响已提交项目代码 | 不影响；环境切换不修改代码文件 |
| 是否适合 RTX 3060 Laptop GPU | 适合；更利于按 `6 GB` 显存约束做独立调试和 profiling |
| 后续 GPU profiling 在哪里执行 | 建议只在新建 GPU 环境中执行 |

### 预计步骤

只做草案，不执行：

1. 新建 `FedTrafficFlowGPU` 环境。
2. 在新环境中持久化 `PYTHONNOUSERSITE=1`。
3. 安装项目基础依赖。
4. 安装匹配驱动与官方支持矩阵的 CUDA 版 PyTorch。
5. 注册独立 Jupyter kernel。
6. 验证 `python -m pip --version`、`torch.__file__` 与 `torch.cuda.is_available()`。
7. 仅在验证通过后，再做轻量 GPU smoke / profiling。

### 初始风险评估结论

`从风险最小化角度看，方案 A 更优。`

### 方案 B：替换当前 CPU 环境 torch

目标环境：

- `E:\anaconda3\envs\FedTrafficFlow`

即在当前环境中移除 `torch=2.8.0+cpu`，再安装 CUDA 版 PyTorch。

### 潜在优点

- 无需新建环境，表面上看磁盘空间与环境管理数量更少。
- 现有项目命令、解释器路径与 notebook 入口不需要额外切换。

### 主要风险

- 直接影响当前已经修复好的 CPU 环境，一旦 CUDA 版安装失败，当前环境可能进入不可用或半可用状态。
- 当前 `fedtrafficflow` Jupyter kernel 的 `kernel.json` 明确指向 `E:\anaconda3\envs\FedTrafficFlow\python.exe`，因此一旦替换当前环境中的 `torch`，该 kernel 会被直接连带改变。
- 若 pip / conda 混用或 CUDA 依赖解析不一致，容易出现版本冲突、DLL 问题、`torch` 可导入但 CUDA 不可用等难以快速回滚的情形。
- 如果后续发现 CUDA 版与驱动、依赖或项目其他包不兼容，还需要额外回滚到 CPU-only 版本，恢复成本高于新建环境。
- 当前 `requirements.txt` 仅列出基础依赖，并未锁定 `torch`。如果直接在当前环境替换 `torch`，仓库文件虽然不会自动变化，但“仓库声明”与“环境事实”会进一步分离，不利于复现管理。

### 对关键问题的影响判断

| 关注点 | 影响判断 |
|---|---|
| 是否破坏当前 CPU 环境 | 有较高概率影响，至少会改变当前已验证环境的核心深度学习栈 |
| 是否影响 Jupyter kernel | 会直接影响当前 `fedtrafficflow` kernel，因为 kernel 指向的就是该环境 |
| 是否影响 `requirements.txt` | 文件本身默认不变，但会加剧环境与声明不一致的问题 |
| 是否影响已提交项目代码 | 不直接修改代码，但会改变代码运行所依赖的唯一主环境 |
| 是否适合 RTX 3060 Laptop GPU | 从硬件角度可运行，但不构成替换当前 CPU 环境的理由 |
| 后续 GPU profiling 在哪里执行 | 若采用本方案，只能在被替换后的主环境中执行，风险集中 |

### 回滚难度

`中到高。`

原因：

- 回滚不仅是重新安装 `torch=2.8.0+cpu`，还要重新确认 `pip` 路径、`torch` 路径、kernel 行为与依赖一致性。
- 一旦安装过程中混入其他版本包，回滚后的环境也未必等同于当前已验证状态。

### 初始风险评估结论

`从纯风险控制角度看，方案 B 不是默认首选方案，但在作者明确要求不创建新环境的前提下，可作为后续执行方案。`

## 6. 最终执行决策

在初始风险评估中，方案 A（新建独立 GPU 环境）风险最低；但作者已明确要求“不创建新的 conda 环境，就修改现有的这个环境”。因此，本项目后续 CUDA PyTorch 处理将按照方案 B 执行：

- 不创建新的 conda 环境。
- 继续使用现有环境 `E:\anaconda3\envs\FedTrafficFlow`。
- 在该环境中将 `torch 2.8.0+cpu` 原地替换为 CUDA 版 PyTorch。
- 替换前必须确认 Git 工作区干净。
- 替换前必须备份 `pip freeze`。
- 替换后必须验证 `torch.cuda.is_available()`。
- 替换后只允许先做极轻量 CUDA 张量测试，不直接运行正式训练或正式 profiling。

该决策的风险高于新建独立 GPU 环境，因为当前稳定 CPU 环境会被直接改变。因此后续执行必须严格保留安装前记录和回滚依据。

## 7. 后续执行草案

以下仅为草案，不在本阶段执行：

1. 保持使用现有环境 `E:\anaconda3\envs\FedTrafficFlow`。
2. 激活环境后再次核验 `PYTHONNOUSERSITE=1`、`ENABLE_USER_SITE=False` 与 `python -m pip --version` 的来源路径。
3. 备份 `pip freeze` 与替换前 `torch` 状态，作为原地替换前基线。
4. 在现有环境中安装与 Windows、Python 3.9、当前 NVIDIA 驱动兼容的 CUDA 版 PyTorch。
5. 验证 `python -m pip --version`、`torch.__version__`、`torch.version.cuda`、`torch.__file__`、`torch.cuda.is_available()`。
6. 仅在 CUDA 可用时，先运行极轻量 GPU 张量测试。
7. GPU 可用性核验通过后，再决定是否进入轻量 GPU smoke / profiling 阶段。

## 8. 风险控制

- 本次最终执行口径遵循作者决策，不创建新环境，但必须承认方案 B 的环境风险高于方案 A。
- 原地替换前必须确认当前 CPU 环境 `E:\anaconda3\envs\FedTrafficFlow` 的 Git 工作区、`pip` 来源与 user-site 隔离状态均正常。
- 必须保留替换前的 `pip freeze`、`torch` 版本和安装路径记录，确保出现兼容性问题时可回滚。
- 当前 `fedtrafficflow` Jupyter kernel 绑定的是 `E:\anaconda3\envs\FedTrafficFlow\python.exe`，因此原地替换后需要再次核验 kernel 指向是否保持一致。
- `requirements.txt` 当前未锁定 `torch`，后续如需提高复现性，应补充记录原地替换时的实际安装命令和结果。
- 安装失败或 CUDA 不可用时，不得继续运行训练、正式 profiling 或论文正式实验。
- 不使用 `force push`，也不将 CUDA 可用性测试或轻量 GPU 张量测试直接视为论文正式结果。

## 9. 是否可以进入 CUDA PyTorch 原地替换阶段

可以进入原地替换阶段，但前提是：

1. Git 工作区必须干净。
2. `PYTHONNOUSERSITE=1` 与 `ENABLE_USER_SITE=False` 必须正常。
3. `python -m pip` 必须指向 `E:\anaconda3\envs\FedTrafficFlow`。
4. 必须先备份 `pip freeze`。
5. 安装失败时不得继续运行训练或 profiling。
6. 安装成功后只做 CUDA 可用性核验和极轻量 GPU 张量测试。
