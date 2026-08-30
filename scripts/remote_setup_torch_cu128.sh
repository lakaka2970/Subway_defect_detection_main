#!/bin/bash
# 用 AutoDL 学术加速代理从 pytorch.org 装 cu128 版 torch（RTX 5090 sm_120 必需）
# 实测：不走代理 8 KB/s（87 小时），走代理 6.9 MB/s（约 2 分钟）
set -euo pipefail

source /etc/network_turbo >/dev/null 2>&1 || true

PY=/root/miniconda3/envs/subway/bin/python
PIP=/root/miniconda3/envs/subway/bin/pip

echo "[1/3] 卸载现有 cu126 版 torch  $(date)"
$PIP uninstall -y torch torchvision triton >/dev/null 2>&1 || true

echo "[2/3] 安装 torch 2.7.1+cu128  $(date)"
$PIP install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128/ --no-cache-dir 2>&1 | tail -6

echo "[3/3] 校验  $(date)"
$PY /root/autodl-tmp/subway/scripts/verify_gpu.py
echo "TORCH_CU128_SETUP_COMPLETE $(date)"
