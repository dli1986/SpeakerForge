# CosyVoice WSL2 Environment Setup

**System:** Ubuntu 24.04 LTS on WSL2, NVIDIA RTX 2000 Ada (8GB VRAM), driver 595.71, CUDA 12.6 runtime from Windows.

---

## 一、Python 3.10 安装

Ubuntu 24.04 默认源只有 Python 3.12，CosyVoice 需要 3.10，通过 deadsnakes PPA 安装：

```bash
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.10 python3.10-venv python3.10-dev
python3.10 -m venv ~/cosyvoice_env
source ~/cosyvoice_env/bin/activate
```

---

## 二、依赖安装（完整命令，顺序不可变）

```bash
source ~/cosyvoice_env/bin/activate
cd ~/CosyVoice

# 1. 固定 setuptools（82+ 移除了 pkg_resources，openai-whisper 的 setup.py 需要它）
pip install "setuptools==68.2.2"

# 2. 先单独装 whisper，加 --no-build-isolation（绕过隔离构建环境重新拉取新版 setuptools）
#    副作用：会把 torch 降到 2.3.1，torchaudio 产生冲突
pip install --no-build-isolation openai-whisper==20231117

# 3. 对齐 torchaudio 版本
pip install "torchaudio==2.3.1" --extra-index-url https://download.pytorch.org/whl/cu121

# 4. 安装全部依赖（DS_BUILD_OPS=0 跳过 deepspeed CUDA ops 预编译，WSL2 下无 nvcc 时必须）
DS_BUILD_OPS=0 pip install -r requirements.txt

# 5. 安装 Matcha-TTS（必须加 --no-deps！否则会把 torch 升到最新版破坏整个环境）
pip install --no-deps -e third_party/Matcha-TTS
```

---

## 三、Troubleshooting

### 问题 1：`pkg_resources` 缺失

```
ModuleNotFoundError: No module named 'pkg_resources'
```

**原因：** setuptools 79+/82+ 已移除该模块，`openai-whisper` 的旧版 setup.py 依赖它。  
**修复：** `pip install "setuptools==68.2.2"`

---

### 问题 2：隔离构建环境仍拉取新版 setuptools

即使降级了 setuptools，pip 的隔离构建（build isolation）会重新从 PyPI 拉取最新版，导致问题复现。  
**修复：** `pip install --no-build-isolation openai-whisper==20231117`

---

### 问题 3：deepspeed 构建失败（`CUDA_HOME` 不存在）

```
MissingCUDAException: CUDA_HOME does not exist, unable to compile CUDA op(s)
```

**原因：** WSL2 只有 CUDA 运行时库，没有 CUDA Toolkit（nvcc）。

**快速修复（推理/普通训练）：**
```bash
DS_BUILD_OPS=0 pip install -r requirements.txt
```

**完整修复（需要训练或 TensorRT）：**
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update && sudo apt install -y cuda-toolkit-12-6
echo 'export CUDA_HOME=/usr/local/cuda-12.6' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

### 问题 4：安装 Matcha-TTS 后 torch 被升到 2.11.0，环境崩溃

**原因：** `pip install -e third_party/Matcha-TTS` 读取其 requirements.txt，其中 `torch>=2.0.0` 无上限，pip 自动升到最新版，顺带拉来 `torchcodec+cu130` 等不兼容包。

**修复：**
```bash
# 强制降回 torch 2.3.1（--no-deps 避免再次触发依赖升级）
pip install --no-deps "torch==2.3.1+cu121" "torchaudio==2.3.1+cu121" \
    --extra-index-url https://download.pytorch.org/whl/cu121

pip uninstall torchcodec torchvision -y

# 重装 Matcha-TTS（--no-deps 是关键）
pip install --no-deps -e third_party/Matcha-TTS
```

---

### 问题 5：`diffusers` 导入失败

```
ImportError: cannot import name 'cached_download' from 'huggingface_hub'
```

**原因：** `huggingface_hub 0.27+` 移除了 `cached_download`，而 `diffusers==0.29` 依赖它；`transformers 4.51.3` 又要求 `huggingface_hub>=0.30`，两者冲突。  
**修复：**
```bash
pip install "diffusers>=0.30.0,<0.32.0"
```

---

## 四、验证

```bash
python -c "
import torch, torchaudio, deepspeed
from matcha.models.matcha_tts import MatchaTTS
import transformers, diffusers
print('torch:', torch.__version__)           # 2.3.1+cu121
print('torchaudio:', torchaudio.__version__) # 2.3.1+cu121
print('deepspeed:', deepspeed.__version__)   # 0.15.1
print('matcha-tts: OK')
print('transformers:', transformers.__version__)  # 4.51.3
print('diffusers:', diffusers.__version__)        # 0.31.0
print('GPU:', torch.cuda.is_available())          # True
"
```

---

## 五、经验总结

环境出问题时首先检查 `pip list | grep torch`。两个最常见的破坏源：

| 操作 | 副作用 |
|------|--------|
| `pip install openai-whisper` | 把 torch 降到 2.3.1 |
| `pip install -e third_party/Matcha-TTS` | 把 torch 升到最新版（如 2.11.0） |

任何一步完成后都应验证 torch 版本未被意外改变。
