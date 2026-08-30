#!/bin/bash
# 在远端 subway 环境安装 AI 数据工厂所需依赖
set -x
# 注意：不要 source /etc/network_turbo —— 它会设置 http_proxy 覆盖掉可用的
# 阿里云 PyPI 镜像，导致 "No matching distribution found"。
# 只有 download.pytorch.org / github / huggingface 才需要那个代理。
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
source /root/miniconda3/etc/profile.d/conda.sh
conda activate subway
PIP=/root/miniconda3/envs/subway/bin/pip
PY=/root/miniconda3/envs/subway/bin/python

echo "=== [1/3] 基础科学计算 ==="
$PIP install --no-cache-dir opencv-python-headless pillow scikit-learn pandas matplotlib tqdm scipy pyyaml

echo "=== [2/3] timm (DINOv2 特征) ==="
$PIP install --no-cache-dir timm

echo "=== [3/3] ultralytics (SAM + YOLO 训练) ==="
$PIP install --no-cache-dir ultralytics==8.4.60

echo "=== 校验 ==="
$PY -c "
import importlib
for m in ['numpy','cv2','sklearn','pandas','matplotlib','tqdm','yaml','scipy','PIL']:
    try:
        mod=importlib.import_module(m)
        print('OK  %-12s %s'%(m, getattr(mod,'__version__','?')))
    except Exception as e:
        print('FAIL %-12s %s'%(m,e))
for m in ['timm','torch','ultralytics']:
    try:
        mod=importlib.import_module(m)
        print('OK  %-12s %s'%(m, getattr(mod,'__version__','?')))
    except Exception as e:
        print('FAIL %-12s %s'%(m,e))
"
echo "REMOTE_DEPS_DONE $(date)"
