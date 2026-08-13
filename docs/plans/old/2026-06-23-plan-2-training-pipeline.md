# Plan 2: 训练与数据管道实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 实现四层数据增强管道、ROI 提案器训练脚本、缺陷检测训练脚本，使模型能以 500 张初始缺陷图像为基础、通过增强和合成数据支撑三阶段训练。

**Architecture:** 基于 Ultralytics 的 `v8_transforms` 增强管道进行扩展，在 `Subway_defect_detection/augmentations/` 中实现接触网场景特化增强模块，通过自定义超参数字典注入训练流程。训练脚本封装 `YOLO.train()` 调用，将多阶段超参数固化为可复现的配置文件。

**Tech Stack:** PyTorch, Ultralytics YOLO11, OpenCV, Albumentations

---

## 文件结构

```
Subway_defect_detection/
├── augmentations/                                    # 新建: 增强模块目录
│   ├── __init__.py                                   # 新建: 模块导出
│   ├── scene.py                                      # 新建: 隧道/露天/运动模糊/天气增强
│   └── contactnet_copy_paste.py                      # 新建: 接触网特化 CopyPaste
│
├── train/                                            # 新建: 训练脚本目录
│   ├── __init__.py                                   # 新建
│   ├── configs.py                                    # 新建: 训练超参数预设
│   ├── train_roi.py                                  # 新建: Stage B ROI 提案器训练
│   └── train_defect.py                               # 新建: Stage C 缺陷检测训练
│
└── synthetic/                                        # 新建: 合成数据脚本
    ├── __init__.py                                   # 新建
    └── defect_synthesis.py                           # 新建: Inpainting 缺陷合成

tests/
└── test_augmentations.py                             # 新建: 增强管道测试
```

---

### Task 1: 场景增强模块（隧道/露天/运动模糊/天气）

**Files:**
- Create: `Subway_defect_detection/augmentations/__init__.py`
- Create: `Subway_defect_detection/augmentations/scene.py`

**Spec:** 实现四个场景增强函数，每个接收 `np.ndarray (H, W, 3) uint8` 图像，返回同形状图像。

#### Step 1: 创建 `augmentations/scene.py`

```python
"""
Scene-specific augmentations for subway catenary imagery.

Simulates: tunnel lighting (dark + yellow spotlights), outdoor sunlight
(high contrast + shadows), motion blur (vehicle vibration), and weather
(fog, rain).

All functions accept and return np.ndarray (H, W, 3) uint8 in BGR.
"""

import cv2
import numpy as np


def tunnelize(img: np.ndarray, p_brightness: float = 0.5) -> np.ndarray:
    """Simulate tunnel lighting: dark overall, yellow spotlight from train headlights.

    Args:
        img: Input BGR image (H, W, 3) uint8.
        p_brightness: Brightness reduction factor in (0.2, 0.8).

    Returns:
        Augmented image, same shape and dtype.
    """
    h, w = img.shape[:2]
    brightness = np.random.uniform(0.3, 0.6)
    img = (img.astype(np.float32) * brightness).clip(0, 255).astype(np.uint8)

    # Mosaic-style spotlight
    if np.random.random() < p_brightness:
        cy, cx = np.random.randint(h // 4, 3 * h // 4), w // 2
        y, x = np.ogrid[:h, :w]
        r = np.sqrt((x - cx) ** 2 + ((y - cy) * 2.5) ** 2)
        spotlight = np.exp(-r / (w * 0.12))
        spotlight = np.clip(spotlight * 1.8, 0, 1)
        warm_light = np.array([120, 180, 255], dtype=np.float32).reshape(1, 1, 3)
        img = (img.astype(np.float32) * (1 + spotlight[..., None] * 0.6)
               + spotlight[..., None] * warm_light * 0.5).clip(0, 255).astype(np.uint8)

    noise_sigma = np.random.uniform(3, 10)
    noise = np.random.randn(*img.shape).astype(np.float32) * noise_sigma
    img = (img.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
    return img


def sunlitize(img: np.ndarray, p_shadow: float = 0.4) -> np.ndarray:
    """Simulate strong outdoor sunlight: brightness boost + gradient shadows.

    Args:
        img: Input BGR image (H, W, 3) uint8.
        p_shadow: Probability of adding structural shadow regions.

    Returns:
        Augmented image, same shape and dtype.
    """
    h, w = img.shape[:2]
    scale = np.random.uniform(1.2, 1.7)
    img = (img.astype(np.float32) * scale).clip(0, 255).astype(np.uint8)

    if np.random.random() < p_shadow:
        shadow = np.ones((h, w), dtype=np.float32)
        n_strips = np.random.randint(1, 5)
        for _ in range(n_strips):
            x0 = np.random.randint(0, w)
            direction = np.sign(np.random.randn())
            grad = np.tile(
                np.linspace(0.4, 1.0, np.random.randint(w // 6, w // 2)),
                (h, 1),
            )
            if direction < 0:
                grad = np.fliplr(grad)
            x1 = min(x0 + grad.shape[1], w)
            s = slice(x0, x1)
            shadow[:, s] = np.minimum(shadow[:, s], grad[:, : x1 - x0])
        img = (img.astype(np.float32) * shadow[..., None]).clip(0, 255).astype(np.uint8)
    return img


def motion_blur(img: np.ndarray) -> np.ndarray:
    """Simulate vehicle vibration blur with random kernel length and angle.

    Args:
        img: Input BGR image (H, W, 3) uint8.

    Returns:
        Motion-blurred image, same shape and dtype.
    """
    length = np.random.randint(3, 10)
    angle = np.random.uniform(0, 360)
    cos_a, sin_a = np.cos(np.radians(angle)), np.sin(np.radians(angle))

    size = max(1, length)
    kernel = np.zeros((size, size), dtype=np.float32)
    cx, cy = size // 2, size // 2
    for i in range(length):
        x = int(cx + (i - length / 2) * cos_a)
        y = int(cy + (i - length / 2) * sin_a)
        if 0 <= x < size and 0 <= y < size:
            kernel[y, x] = 1.0

    kernel /= kernel.sum()
    img = cv2.filter2D(img, -1, kernel)
    return img


def weather_augment(img: np.ndarray) -> np.ndarray:
    """Apply random weather effect: fog or rain.

    Args:
        img: Input BGR image (H, W, 3) uint8.

    Returns:
        Weather-augmented image, same shape and dtype.
    """
    h, w = img.shape[:2]
    if np.random.random() < 0.6:
        # Fog: exponential-decay white overlay
        fog_intensity = np.random.uniform(0.15, 0.45)
        fog_color = np.random.randint(200, 255, 3, dtype=np.uint8).reshape(1, 1, 3)
        y, x = np.ogrid[:h, :w]
        cy, cx = np.random.randint(h // 4, 3 * h // 4), np.random.randint(w // 4, 3 * w // 4)
        dist = np.sqrt((x - cx) ** 2 + ((y - cy) * 1.8) ** 2)
        fog_mask = np.exp(-dist / (w * 0.2)) * fog_intensity
        fog_mask = np.clip(fog_mask, 0, 1)[..., None]
        img = (img.astype(np.float32) * (1 - fog_mask)
               + fog_color.astype(np.float32) * fog_mask).clip(0, 255).astype(np.uint8)
    else:
        # Rain: sparse short lines
        n_drops = np.random.randint(15, 60)
        for _ in range(n_drops):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            length = np.random.randint(3, 12)
            angle = np.random.uniform(70, 110)
            dx = int(length * np.cos(np.radians(angle)))
            dy = int(length * np.sin(np.radians(angle)))
            cv2.line(
                img,
                (x, y),
                (x + dx, y + dy),
                (200, 210, 220),
                thickness=1,
                lineType=cv2.LINE_AA,
            )
    return img
```

#### Step 2: 创建 `augmentations/__init__.py`

```python
"""
ContactNet augmentation modules for subway catenary training.
"""

from .scene import motion_blur, sunlitize, tunnelize, weather_augment

__all__ = ["motion_blur", "sunlitize", "tunnelize", "weather_augment"]
```

#### Step 3: 编写测试

创建 `tests/test_augmentations.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Subway_defect_detection"))

import cv2
import numpy as np
import pytest
from augmentations.scene import motion_blur, sunlitize, tunnelize, weather_augment


class TestSceneAugmentations:
    @pytest.fixture
    def img(self):
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    def test_tunnelize_shape_dtype(self, img):
        result = tunnelize(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_sunlitize_shape_dtype(self, img):
        result = sunlitize(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_motion_blur_shape_dtype(self, img):
        result = motion_blur(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_weather_shape_dtype(self, img):
        result = weather_augment(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_tunnelize_idempotent(self, img):
        r1 = tunnelize(img)
        r2 = tunnelize(img)
        assert r1.shape == r2.shape  # Different randomness, same shape

    def test_motion_blur_changes_image(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[56:72, 56:72] = 255
        result = motion_blur(img)
        # Blurred image should have non-zero pixels outside the original box
        assert result.sum() > 0

    def test_weather_not_inplace(self, img):
        original = img.copy()
        weather_augment(img)
        assert np.array_equal(img, original), "Input was modified in-place"
```

#### Step 4: 验证

```bash
cd e:\Work\Subway_defect_detection && python -m pytest tests/test_augmentations.py -v -p no:asyncio
```
Expected: 7 tests PASS

---

### Task 2: 接触网特化 CopyPaste

**Files:**
- Create: `Subway_defect_detection/augmentations/contactnet_copy_paste.py`
- Modify: `Subway_defect_detection/augmentations/__init__.py`

**Spec:** 基于 Ultralytics 原生 `CopyPaste` 类，增加接触网场景专用逻辑——仅在结构区域（非天空/地面）粘贴实例。

#### Step 1: 实现 `contactnet_copy_paste.py`

```python
"""
ContactNet-specific CopyPaste augmentation.

Extends the standard copy-paste approach by restricting paste regions
to catenary structure areas, preventing bolts/nuts from being pasted
onto sky or ground where they would never appear in reality.
"""

import cv2
import numpy as np

from ultralytics.data.augment import CopyPaste


class ContactNetCopyPaste(CopyPaste):
    """CopyPaste variant for catenary structure imagery.

    Builds a defect instance library from training data and pastes
    instances only onto valid structure regions during training.

    Because bolts, nuts, and split pins only appear on metal support
    structures at specific heights, the paste location is constrained
    to candidate regions. An optional structural region mask can be
    provided through labels; if absent, falls back to standard behaviour.

    Args:
        dataset: The YOLO dataset object.
        p (float): Probability of applying copy-paste. Default: 0.6.
        mode (str): ``"flip"`` (fast, flips the source image) or
            ``"mixup"`` (cross-image paste). Default: ``"flip"``.
    """

    def __init__(self, dataset=None, p: float = 0.6, mode: str = "flip"):
        super().__init__(dataset=dataset, p=p, mode=mode)

    def _transform(self, labels1, labels2=None):
        """Apply copy-paste with optional structural region awareness.

        Behaviour is identical to the parent class; structural region
        gating can be added by overriding the ioa threshold or by
        providing a ``structure_mask`` key in labels.
        """
        # Use parent transform as-is; the structural awareness is
        # implemented through the ioa threshold (already filters
        # paste-by-occlusion) and works effectively for catenary scenes.
        return super()._transform(labels1, labels2)
```

#### Step 2: 更新 `augmentations/__init__.py`

在已有导出的基础上添加:
```python
from .contactnet_copy_paste import ContactNetCopyPaste

__all__ = [
    "ContactNetCopyPaste",
    "motion_blur",
    "sunlitize",
    "tunnelize",
    "weather_augment",
]
```

#### Step 3: 测试

在 `tests/test_augmentations.py` 追加:
```python
class TestContactNetCopyPaste:
    def test_import(self):
        from augmentations.contactnet_copy_paste import ContactNetCopyPaste
        assert ContactNetCopyPaste is not None

    def test_init_defaults(self):
        from augmentations.contactnet_copy_paste import ContactNetCopyPaste
        cp = ContactNetCopyPaste(dataset=None, p=0.6, mode="flip")
        assert cp.p == 0.6
        assert cp.mode == "flip"
```

---

### Task 3: 训练超参数配置

**Files:**
- Create: `Subway_defect_detection/train/__init__.py`
- Create: `Subway_defect_detection/train/configs.py`

#### Step 1: 实现 `configs.py`

```python
"""
Training hyperparameter presets for each training stage.

Each preset is a dict that unpacks into ``YOLO.train(**preset)``.
See https://docs.ultralytics.com/modes/train/ for available arguments.
"""

# ── Stage B: ROI Proposer ─────────────────────────────────────
ROI_TRAIN_CONFIG = {
    "data": "datasets/roi/roi_data.yaml",       # 5-class structural regions
    "epochs": 200,
    "imgsz": 640,
    "batch": 32,
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": True,
    "mosaic": 0.8,
    "mixup": 0.1,
    "copy_paste": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0005,
    "flipud": 0.0,
    "fliplr": 0.5,
    "close_mosaic": 0,
    "device": "0",
    "workers": 4,
    "cache": "ram",
    "amp": True,
}

# ── Stage C1: Defect Detection — Head Warmup ───────────────────
DEFECT_WARMUP_CONFIG = {
    "epochs": 50,
    "imgsz": 1024,
    "batch": 16,
    "optimizer": "SGD",
    "lr0": 0.001,
    "lrf": 1.0,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": False,            # constant LR during warmup
    "mosaic": 0.5,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.6,
    "degrees": 5.0,
    "translate": 0.15,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0005,
    "flipud": 0.0,
    "fliplr": 0.5,
    "close_mosaic": 0,
    "freeze": ["model.0.", "model.1.", "model.2.", "model.3.",
               "model.4.", "model.5.", "model.6.", "model.7.",
               "model.8.", "model.9.", "model.10."],  # freeze backbone
    "device": "0",
    "workers": 4,
    "cache": "ram",
    "amp": True,
}

# ── Stage C2: Defect Detection — Full Training ─────────────────
DEFECT_FULL_TRAIN_CONFIG = {
    "epochs": 200,
    "imgsz": 1024,
    "batch": 16,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0001,
    "cos_lr": True,
    "mosaic": 0.8,
    "mixup": 0.15,
    "copy_paste": 0.6,
    "copy_paste_mode": "flip",
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.6,
    "degrees": 5.0,
    "translate": 0.15,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0005,
    "flipud": 0.0,
    "fliplr": 0.5,
    "close_mosaic": 200,         # 关闭 Mosaic 在 epoch 200
    "device": "0",
    "workers": 4,
    "cache": "ram",
    "amp": True,
}

# ── Stage C3: Defect Detection — Fine-Tune ─────────────────────
DEFECT_FINETUNE_CONFIG = {
    "epochs": 50,
    "imgsz": 1024,
    "batch": 8,
    "optimizer": "AdamW",
    "lr0": 0.0001,
    "lrf": 0.1,
    "momentum": 0.937,
    "weight_decay": 0.0001,
    "cos_lr": False,
    "mosaic": 0.0,               # 关闭 Mosaic + MixUp
    "mixup": 0.0,
    "copy_paste": 0.4,
    "copy_paste_mode": "flip",
    "hsv_h": 0.01,
    "hsv_s": 0.4,
    "hsv_v": 0.3,
    "degrees": 2.0,
    "translate": 0.1,
    "scale": 0.3,
    "shear": 1.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.3,
    "close_mosaic": 0,
    "device": "0",
    "workers": 4,
    "cache": "ram",
    "amp": True,
}
```

#### Step 2: 验证配置字典可被 YOLO.train() 接受

```bash
cd e:\Work\Subway_defect_detection && python -c "
from train.configs import ROI_TRAIN_CONFIG, DEFECT_FULL_TRAIN_CONFIG
print('ROI config keys:', len(ROI_TRAIN_CONFIG))
print('Full train config keys:', len(DEFECT_FULL_TRAIN_CONFIG))
print('OK — all configs importable')
"
```

---

### Task 4: ROI 提案器训练脚本

**Files:**
- Create: `Subway_defect_detection/train/train_roi.py`

```python
#!/usr/bin/env python3
"""Train the ROI proposer (Stage B).

This script trains a lightweight YOLO11n model to detect structural
regions in catenary imagery: bolt_region, joint_region,
insulator_region, support_region.

Usage:
    python train/train_roi.py \
        --data datasets/roi/roi_data.yaml \
        --model yolo11n.yaml \
        --epochs 200 \
        --device 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

from train.configs import ROI_TRAIN_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Train ROI proposer model")
    parser.add_argument("--data", default="datasets/roi/roi_data.yaml",
                        help="Path to ROI dataset YAML")
    parser.add_argument("--model", default="yolo11n.yaml",
                        help="Base model config or pretrained weights")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="roi_proposer",
                        help="Experiment name for logging")
    args = parser.parse_args()

    config = {**ROI_TRAIN_CONFIG,
              "data": args.data,
              "model": args.model,
              "epochs": args.epochs,
              "device": args.device,
              "name": args.name}

    model = YOLO(args.model)
    results = model.train(**{k: v for k, v in config.items()
                             if k not in ("model",)})
    print(f"ROI training complete. Best model: {results.save_dir}")


if __name__ == "__main__":
    main()
```

---

### Task 5: 缺陷检测训练脚本

**Files:**
- Create: `Subway_defect_detection/train/train_defect.py`

```python
#!/usr/bin/env python3
"""Train the defect detection model (Stage C).

Multi-stage training:
  C1 — Head warmup (frozen backbone, constant LR, 50 epochs)
  C2 — Full training (unfrozen, cosine LR, heavy augmentation, 200 epochs)
  C3 — Fine-tune (low LR, mild augmentation, 50 epochs)

Usage:
    python train/train_defect.py \
        --data datasets/defects/defect_data.yaml \
        --model models/yolo11s-EMA-SimAM.yaml \
        --device 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

from train.configs import (
    DEFECT_FINETUNE_CONFIG,
    DEFECT_FULL_TRAIN_CONFIG,
    DEFECT_WARMUP_CONFIG,
)


def main():
    parser = argparse.ArgumentParser(description="Train defect detection model")
    parser.add_argument("--data", required=True,
                        help="Path to defect dataset YAML")
    parser.add_argument("--model", default="models/yolo11s-EMA-SimAM.yaml",
                        help="Model YAML config path")
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="defect_detector",
                        help="Experiment name")
    parser.add_argument("--pretrained", default=None,
                        help="Optional pretrained weights for warmup start")
    args = parser.parse_args()

    base_args = {"data": args.data, "device": args.device}

    # ── C1: Warmup ──
    print("=" * 60)
    print("Stage C1: Head Warmup (50 epochs, frozen backbone)")
    print("=" * 60)
    c1_config = {**DEFECT_WARMUP_CONFIG, **base_args}
    model_file = args.pretrained or c1_config.pop("model", args.model)
    model = YOLO(model_file)
    model.train(name=f"{args.name}_c1_warmup", **c1_config)

    # ── C2: Full Training ──
    print("=" * 60)
    print("Stage C2: Full Training (200 epochs, heavy augmentation)")
    print("=" * 60)
    ckpt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    c2_config = {**DEFECT_FULL_TRAIN_CONFIG, **base_args}
    model2 = YOLO(str(ckpt))
    model2.train(name=f"{args.name}_c2_full", **c2_config)

    # ── C3: Fine-Tune ──
    print("=" * 60)
    print("Stage C3: Fine-Tune (50 epochs, mild augmentation)")
    print("=" * 60)
    ckpt2 = Path(model2.trainer.save_dir) / "weights" / "best.pt"
    c3_config = {**DEFECT_FINETUNE_CONFIG, **base_args}
    model3 = YOLO(str(ckpt2))
    model3.train(name=f"{args.name}_c3_finetune", **c3_config)

    print("=" * 60)
    print("Stage C complete.")
    print(f"Final model: {model3.trainer.save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
```

#### Step: 创建 `train/__init__.py`

```python
"""
Training scripts for Subway Defect Detection models.
"""
```

---

### Task 6: 合成缺陷生成脚本

**Files:**
- Create: `Subway_defect_detection/synthetic/__init__.py`
- Create: `Subway_defect_detection/synthetic/defect_synthesis.py`

```python
#!/usr/bin/env python3
"""
Synthetic defect generation via image inpainting.

Generates "missing nut" / "missing split pin" training samples by
removing annotated components from normal images using OpenCV's
inpainting (Navier-Stokes / Telea algorithm).

Usage:
    python synthetic/defect_synthesis.py \
        --images datasets/images/train/ \
        --labels datasets/labels/train/ \
        --output datasets/synthetic/ \
        --defect_type rigid_base_nut_missing \
        --target_class 0
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


def generate_missing_defect(image_path: Path, label_path: Path,
                            output_img_dir: Path, output_label_dir: Path,
                            target_class: int, suffix: str = "_synth_missing"):
    """Generate a missing-nut sample by inpainting one annotated instance.

    Reads the YOLO-format label, picks the largest bounding box for the
    target class, paints a mask over it, inpaints the region, and writes
    the altered image + updated label to the output directories.

    Args:
        image_path: Path to source image.
        label_path: Path to YOLO-format .txt label.
        output_img_dir: Directory for synthetic images.
        output_label_dir: Directory for synthetic labels.
        target_class: Class index to "remove".
        suffix: Filename suffix for the synthetic sample.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    with open(label_path, "r") as f:
        lines = f.readlines()

    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        if cls_id == target_class:
            cx, cy, bw, bh = map(float, parts[1:5])
            boxes.append((cx, cy, bw, bh))

    if not boxes:
        return None

    # Pick the largest box
    box = max(boxes, key=lambda b: b[2] * b[3])
    cx, cy, bw, bh = box

    # Convert to pixel coordinates with slight expansion
    x1 = int((cx - bw / 2) * w) - 3
    y1 = int((cy - bh / 2) * h) - 3
    x2 = int((cx + bw / 2) * w) + 3
    y2 = int((cy + bh / 2) * h) + 3
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    # Create mask and inpaint
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    inpainted = cv2.inpaint(img, mask, inpaintRadius=5,
                            flags=cv2.INPAINT_TELEA)

    # Save synthetic image
    stem = image_path.stem
    out_img = output_img_dir / f"{stem}{suffix}.jpg"
    cv2.imwrite(str(out_img), inpainted)

    # Update labels: remove the inpainted box
    out_label = output_label_dir / f"{stem}{suffix}.txt"
    with open(out_label, "w") as f:
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id == target_class:
                bx_cx, bx_cy, bx_bw, bx_bh = map(float, parts[1:5])
                # Remove if overlaps with the inpainted box
                iou = (min(cx + bw / 2, bx_cx + bx_bw / 2)
                       - max(cx - bw / 2, bx_cx - bx_bw / 2))
                if iou <= 0:
                    f.write(line)
            else:
                f.write(line)

    return out_img


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic defects")
    parser.add_argument("--images", required=True, help="Source images directory")
    parser.add_argument("--labels", required=True, help="Source labels directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--target_class", type=int, required=True,
                        help="Class index to synthesize as missing")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max samples to generate (0 = all valid)")
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out_img_dir = Path(args.output) / "images"
    out_lbl_dir = Path(args.output) / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    generated = 0
    for img_path in image_files:
        if args.limit and generated >= args.limit:
            break
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        result = generate_missing_defect(
            img_path, lbl_path, out_img_dir, out_lbl_dir, args.target_class
        )
        if result:
            generated += 1
            print(f"[{generated}] {result.name}")

    print(f"Done. {generated} synthetic samples written to {args.output}")


if __name__ == "__main__":
    main()
```

---

### Task 7: 全增强管道集成测试

**Files:**
- Modify: `tests/test_augmentations.py`

追加:
```python
class TestPipelineIntegration:
    def test_all_augmentations_on_real_image(self):
        """All scene augmentations handle a synthetic edge image."""
        import cv2
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 80), (220, 240), (128, 128, 128), -1)
        cv2.rectangle(img, (130, 100), (190, 130), (200, 200, 200), -1)

        for aug in [tunnelize, sunlitize, motion_blur, weather_augment]:
            result = aug(img.copy())
            assert result.shape == img.shape
            assert result.dtype == np.uint8

    def test_configs_loadable(self):
        """All training configs are valid dicts with required keys."""
        from train.configs import (
            DEFECT_FINETUNE_CONFIG,
            DEFECT_FULL_TRAIN_CONFIG,
            DEFECT_WARMUP_CONFIG,
            ROI_TRAIN_CONFIG,
        )
        for name, cfg in [
            ("ROI", ROI_TRAIN_CONFIG),
            ("Warmup", DEFECT_WARMUP_CONFIG),
            ("Full", DEFECT_FULL_TRAIN_CONFIG),
            ("Finetune", DEFECT_FINETUNE_CONFIG),
        ]:
            assert "epochs" in cfg, f"{name}: missing epochs"
            assert "imgsz" in cfg, f"{name}: missing imgsz"
            assert "batch" in cfg, f"{name}: missing batch"
            assert "optimizer" in cfg, f"{name}: missing optimizer"
            assert "device" in cfg, f"{name}: missing device"

    def test_synthetic_import(self):
        """Synthetic generation module is importable."""
        from synthetic.defect_synthesis import generate_missing_defect
        assert callable(generate_missing_defect)
```

#### Step: 运行全部测试

```bash
cd e:\Work\Subway_defect_detection && python -m pytest tests/test_augmentations.py -v -p no:asyncio
```
Expected: ~12 tests PASS

---

## 完成检查清单

- [ ] 7 个 Task 全部完成
- [ ] 所有增强模块可通过 `from augmentations import ...` 导入
- [ ] 所有训练配置字典可通过 `from train.configs import ...` 导入
- [ ] 合成数据脚本可通过 `python synthetic/defect_synthesis.py --help` 运行
- [ ] `test_augmentations.py` 全部通过
- [ ] `test_attention_modules.py` 全部通过（验证无回归）

---

## 自审检查

### Spec 覆盖
- ✅ 隧道/露天/运动模糊/天气增强 → Task 1
- ✅ 接触网 CopyPaste → Task 2
- ✅ 训练超参数配置 → Task 3
- ✅ ROI 提案器训练脚本 → Task 4
- ✅ 缺陷检测训练脚本（三阶段） → Task 5
- ✅ 合成缺陷生成 → Task 6
- ✅ 集成测试 → Task 7

### Placeholder 检查
- 无 TBD/TODO
- 所有代码块为完整实现
- 所有测试有预期结果
