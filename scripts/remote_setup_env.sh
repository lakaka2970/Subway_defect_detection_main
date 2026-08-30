#!/bin/bash
# 地铁接触网项目 —— RTX 5090 环境构建
# Python 3.12 + torch 2.7.1+cu128（SAM 3 / ultralytics 8.4.60 / timm 均可满足）
set -euo pipefail

export PATH=/root/miniconda3/bin:$PATH
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"
PY=/root/miniconda3/envs/subway/bin/python
PIP=/root/miniconda3/envs/subway/bin/pip

echo "[1/5] conda env python=3.12  $(date)"
conda create -y -n subway python=3.12 -q 2>&1 | tail -3

echo "[2/5] torch 2.7.1+cu128  $(date)"
$PIP install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128/ --no-cache-dir 2>&1 | tail -4

echo "[3/5] 基础视觉库  $(date)"
$PIP install -i "$MIRROR" --no-cache-dir -q \
  opencv-python-headless numpy pandas scikit-learn matplotlib tqdm pyyaml 2>&1 | tail -3

echo "[4/5] ultralytics + timm + transformers  $(date)"
$PIP install -i "$MIRROR" --no-cache-dir -q \
  ultralytics==8.4.60 timm transformers accelerate 2>&1 | tail -3

echo "[5/5] 校验  $(date)"
$PY -c "
import torch, cv2, numpy, timm
print('python', __import__('sys').version.split()[0])
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.version.cuda)
print('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
print('cv2', cv2.__version__, 'numpy', numpy.__version__, 'timm', timm.__version__)
import ultralytics; print('ultralytics', ultralytics.__version__)
print('ENV_OK')
"
echo "SETUP_COMPLETE $(date)"
