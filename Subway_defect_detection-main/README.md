# Subway Catenary Defect Detection

Two-stage AI pipeline for detecting defects on subway catenary infrastructure
from ultra-high-resolution (127 MP) imagery.

## Architecture

- **Stage 1**: YOLO11n ROI proposer — detects structural regions on downsampled images
- **Stage 2**: YOLO11s/m with EMA + SimAM attention — detects defects on full-resolution ROI tiles
- **Ensemble**: Weighted Boxes Fusion (WBF) for dual-GPU ground-side deployment

## Quick Start

```bash
# Install in editable mode
pip install -e .

# Verify installation
python -c "from subway_defect.modules.EMA import EMA; print('OK')"
```

## Usage

### Training

```bash
# Stage B: Train ROI proposer
train-roi --data datasets/roi/roi_data.yaml --device 0

# Stage C: Train defect detector (3 sub-stages)
train-defect --data datasets/defects/defect_data.yaml --coco_pretrain --device 0
```

### Inference Server

```bash
# Vehicle-side (single model)
subway-server --port 8001 --model runs/train/weights/best.pt --mode vehicle

# Ground-side (dual GPU ensemble)
subway-server --port 8001 --model runs/train/weights/best.pt \
    --model_b runs/train_p2/weights/best.pt --mode ground
```

### TensorRT Export

```bash
# FP16 export (recommended for vehicle-side)
export-tensorrt --model runs/train/weights/best.pt --fp16

# INT8 export with calibration
export-tensorrt --model runs/train/weights/best.pt --int8 --calibration_data datasets/calibration/
```

## Package Structure

```
subway_defect/           # Custom modules (augmentations, models, pipeline, train, deployment)
ultralytics/             # Vendored Ultralytics framework (with Extramodule bridge)
tests/                   # Test suite (50 tests)
scripts/                 # Utility scripts (AutoDL setup)
```

## Requirements

- Python >= 3.10
- PyTorch >= 2.0 (CUDA 12.1 recommended)
- NVIDIA GPU with >= 8 GB VRAM (RTX 4090 recommended)
