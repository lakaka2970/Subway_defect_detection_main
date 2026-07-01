#!/bin/bash
# AutoDL platform environment setup for Subway Defect Detection
# Run once after creating an AutoDL instance

set -e

echo "=== AutoDL Environment Setup ==="

# AutoDL typically has conda pre-installed
# Create or activate environment
if conda env list | grep -q "subway"; then
    echo "Using existing 'subway' conda environment"
    source activate subway
else
    echo "Creating 'subway' conda environment (Python 3.10)..."
    conda create -n subway python=3.10 -y
    source activate subway
fi

# Install PyTorch (CUDA 12.1 -- matches AutoDL default)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the project in editable mode (pulls in all dependencies)
pip install -e .

# ── Install Kaggle tools for multi-source dataset downloads ──────────
# kagglehub: official Python library (auto-installed as core dependency)
# kaggle: CLI tool (optional, needs API key set up first)
pip install -e ".[download]" 2>/dev/null || pip install kaggle
echo ""
echo "=== Kaggle API Setup ==="
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "[OK] Kaggle API key found at ~/.kaggle/kaggle.json"
else
    echo "[INFO] Kaggle CLI needs an API key to download datasets."
    echo "  To set up:"
    echo "    1. Go to https://www.kaggle.com/settings"
    echo "    2. Click 'Create New Token' → downloads kaggle.json"
    echo "    3. mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/"
    echo "    4. chmod 600 ~/.kaggle/kaggle.json"
    echo ""
    echo "  Or use kagglehub (no auth needed for public datasets):"
    echo "    python -c \"import kagglehub; kagglehub.dataset_download('kaustubhdikshit/neu-surface-defect-database')\""
fi

# Verify GPU
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

# Create data directories
AUTODL_DATA="${1:-/root/autodl-tmp}"
mkdir -p "$AUTODL_DATA/datasets"
mkdir -p "$AUTODL_DATA/models"
mkdir -p "$AUTODL_DATA/runs"

echo ""
echo "=== Setup complete ==="
echo "Data directory: $AUTODL_DATA/datasets"
echo "Model directory: $AUTODL_DATA/models"
echo ""
echo "Next steps:"
echo "  1. Upload datasets to $AUTODL_DATA/datasets/"
echo "  2. Download COCO pretrained weights:"
echo "     wget https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt"
echo "  3. Train: train-defect --data $AUTODL_DATA/datasets/defects/defect_data.yaml --coco_pretrain"
