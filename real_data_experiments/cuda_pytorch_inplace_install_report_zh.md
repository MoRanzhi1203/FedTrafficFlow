# 现有 FedTrafficFlow 环境原地替换 CUDA 版 PyTorch 报告

## 1. 本阶段范围

本阶段未创建新 conda 环境，而是在现有 `E:\anaconda3\envs\FedTrafficFlow` 中原地替换 CPU-only PyTorch。

本阶段未运行训练、未运行正式 profiling、未修改训练代码、未修改 LaTeX、未修改 `simulation_experiments`。

## 2. 替换前状态

- Python 版本：`3.9.23`
- Python 解释器：`E:\anaconda3\envs\FedTrafficFlow\python.exe`
- `PYTHONNOUSERSITE`：`1`
- `ENABLE_USER_SITE`：`False`
- pip 路径：`E:\anaconda3\envs\FedTrafficFlow\lib\site-packages\pip`
- torch 版本：`2.8.0+cpu`
- torch CUDA：`None`
- `cuda_available`：`False`
- pip freeze 备份路径：`env_backup\FedTrafficFlow_before_cuda_torch_pip_freeze.txt`

补充记录：

- `torchvision` 替换前未安装。
- `torchaudio` 替换前未安装。

## 3. GPU / 驱动状态

- GPU 型号：`NVIDIA GeForce RTX 3060 Laptop GPU`
- Driver Version：`560.70`
- CUDA Version：`12.6`
- 显存：`6144 MiB`
- `nvidia-smi` 是否可用：`是`

## 4. 安装命令

实际执行命令如下：

`python -m pip install --no-user --force-reinstall torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126`

说明：

- 按作者要求，未使用裸 `pip`。
- 按作者要求，保留了 `--no-user`。
- 按作者要求，未使用 `conda install torch`。

## 5. 替换后状态

- torch 版本：`2.8.0+cu126`
- torch 文件路径：`E:\anaconda3\envs\FedTrafficFlow\lib\site-packages\torch\__init__.py`
- `torch.version.cuda`：`12.6`
- `torch.cuda.is_available()`：`True`
- `torch.cuda.device_count()`：`1`
- `torch.cuda.get_device_name(0)`：`NVIDIA GeForce RTX 3060 Laptop GPU`
- torchvision 版本：`0.23.0+cu126`
- torchaudio 版本：`2.8.0+cu126`

## 6. 极轻量 GPU 张量测试

结果：`通过`

实际测试口径：

- `x=torch.randn(256,256,device='cuda')`
- `y=x @ x`
- `torch.cuda.synchronize()`

输出摘要：

- `ok= torch.Size([256, 256])`
- `mean= -0.0007351022213697433`

说明该测试仅用于 CUDA 可用性核验，不是训练，不是 profiling，不是论文正式结果。

## 7. Jupyter kernel 状态

`Python (FedTrafficFlow)` kernel 仍指向：

`E:\anaconda3\envs\FedTrafficFlow\python.exe`

核验结果：

- `jupyter kernelspec list` 中仍存在 `fedtrafficflow`
- `C:\Users\MSIPC\AppData\Roaming\jupyter\kernels\fedtrafficflow\kernel.json` 中 `argv[0]` 仍为 `E:\anaconda3\envs\FedTrafficFlow\python.exe`

## 8. 风险说明

- 本方案按作者要求，不创建新环境。
- 当前 CPU-only torch 已被 CUDA 版 torch 替换。
- 如果后续出现 CUDA 兼容性问题，需要回滚 torch。
- 不应把 CUDA 张量测试写成论文正式实验结果。
- 当前环境已经从 CPU-only 栈切换为 CUDA 栈，后续若要恢复到 CPU-only 状态，应基于本次 `pip freeze` 备份和安装前记录执行回滚。

## 9. 后续建议

如果 CUDA 可用，下一阶段再单独运行轻量 GPU smoke / profiling。

如果 CUDA 不可用，下一阶段先分析错误，不要直接运行训练。
