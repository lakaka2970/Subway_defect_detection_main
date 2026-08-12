#!/usr/bin/env python3
"""
Collect hard negative crops from false-positive detections (Stage 5 pre-step).

Runs inference on a validation set using a trained model, identifies false
positive (FP) detection boxes, crops them from the source images, and saves
them organised by predicted class for use in hard-negative-mining training.

Workflow (as documented in Stage 5 YAML):
  1. python scripts/collect_hard_negatives.py --model weights/stage4_best_finetune.pt
  2. Manually spot-check FP crops (optional but recommended)
  3. Stage 5 training with hard negative crops mixed into the dataset

Output::

    data/hard_negatives/
    ├── VHBNM/            (FPs predicted as VHBNM)
    ├── VHBNL/
    ├── SVHBNM/           (FPs predicted as SVHBNM — typically most numerous)
    ├── SVHBNL/
    ├── SVHTNL/
    ├── CBHPM/
    ├── CBVPM/
    └── summary.json      (per-class FP counts, avg confidence, patterns)

Usage::

    # Basic: run with Stage 4 best model on subway_crops val
    python scripts/collect_hard_negatives.py \\
        --model weights/stage4_best_finetune.pt \\
        --data data/subway_crops/subway_crops.yaml

    # Full-image mode (Defect_dataset val, 5120×5120)
    python scripts/collect_hard_negatives.py \\
        --model weights/stage4_best_finetune.pt \\
        --data data/Defect_dataset/defect_data.yaml \\
        --conf 0.3 --iou-thresh 0.1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

SEED = 42
DEFAULT_CONF = 0.30        # minimum confidence for an FP to be collected
DEFAULT_IOU_THRESH = 0.10  # max IoU with any GT to count as FP
CONTEXT_MARGIN = 0.20      # extra margin around bbox when cropping (20%)
MIN_CROP_SIZE = 32          # minimum crop size in pixels
DEFAULT_OUTPUT = Path("data/hard_negatives")

# Class names are read from the data YAML at runtime via _load_class_names().
# The hardcoded fallback below matches the 16-class train_data_2 order (2026-08).
_FALLBACK_CLASS_NAMES_16 = [
    "VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
    "RHTBNM", "RHTBNL", "GWCSBNM", "GWCSBNL", "GWCNM", "GWCNL",
    "BSBM", "INSD", "DRPS",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _load_class_names(data_yaml: Path) -> List[str]:
    """Read class names from data YAML config."""
    import yaml as _yaml
    with open(data_yaml, encoding="utf-8") as f:
        cfg = _yaml.safe_load(f)
    names = cfg.get("names", [])
    if isinstance(names, dict):
        names = [str(names[i]) for i in range(len(names))]
    if not names:
        return _FALLBACK_CLASS_NAMES_16
    return [str(n) for n in names]


def _load_yolo_labels(label_dir: Path) -> Dict[str, List[Tuple[int, float, float, float, float]]]:
    """Load YOLO-format ground-truth labels keyed by image stem.

    Returns:
        ``{stem: [(cls_id, xc_norm, yc_norm, w_norm, h_norm), ...]}``
    """
    labels: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
    for lbl_file in sorted(label_dir.glob("*.txt")):
        stem = lbl_file.stem
        entries: List[Tuple[int, float, float, float, float]] = []
        for line in lbl_file.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                vals = [float(p) for p in parts[1:5]]
                entries.append((cls_id, vals[0], vals[1], vals[2], vals[3]))
            except (ValueError, IndexError):
                continue
        if entries:
            labels[stem] = entries
    return labels


def _box_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Compute IoU between two YOLO-format boxes (xc, yc, w, h) in normalised coords."""
    # Convert to (x1, y1, x2, y2)
    def to_corners(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        xc, yc, w, h = box
        return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)

    x1a, y1a, x2a, y2a = to_corners(box1)
    x1b, y1b, x2b, y2b = to_corners(box2)

    inter_x1 = max(x1a, x1b)
    inter_y1 = max(y1a, y1b)
    inter_x2 = min(x2a, x2b)
    inter_y2 = min(y2a, y2b)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _crop_fp_region(
    img: np.ndarray,
    xc_norm: float, yc_norm: float, w_norm: float, h_norm: float,
    margin: float = CONTEXT_MARGIN,
    min_size: int = MIN_CROP_SIZE,
) -> Optional[np.ndarray]:
    """Crop a false-positive region from the image with context margin.

    Args:
        img: Source image (H, W, 3) BGR.
        xc_norm, yc_norm, w_norm, h_norm: Normalised bbox.
        margin: Fractional margin to add on each side.
        min_size: Minimum crop dimension in pixels.

    Returns:
        Cropped image array, or None if crop is invalid.
    """
    h, w = img.shape[:2]
    bx = xc_norm * w
    by = yc_norm * h
    bw = w_norm * w
    bh = h_norm * h

    # Add margin
    bw_m = bw * (1 + 2 * margin)
    bh_m = bh * (1 + 2 * margin)

    # Ensure minimum size
    bw_m = max(bw_m, min_size)
    bh_m = max(bh_m, min_size)

    x1 = int(bx - bw_m / 2)
    y1 = int(by - bh_m / 2)
    x2 = int(bx + bw_m / 2)
    y2 = int(by + bh_m / 2)

    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return crop


# ── Main logic ───────────────────────────────────────────────────────────────

def collect_hard_negatives(
    model_weights: Path,
    data_yaml: Path,
    output_dir: Path = DEFAULT_OUTPUT,
    conf_threshold: float = DEFAULT_CONF,
    iou_threshold: float = DEFAULT_IOU_THRESH,
    device: str = "0",
    imgsz: int = 1280,
    max_fp_per_class: int = 0,
    split: str = "val",
) -> Dict:
    """Run inference and collect false-positive crops.

    Args:
        model_weights: Path to trained YOLO model (.pt).
        data_yaml: Path to dataset YAML config.
        output_dir: Root output directory for hard negative crops.
        conf_threshold: Minimum confidence to qualify as FP.
        iou_threshold: Maximum IoU with any ground truth to qualify as FP.
        device: CUDA device string (e.g. "0", "cpu").
        imgsz: Inference image size.
        max_fp_per_class: If > 0, cap FP crops per class (0 = unlimited).
        split: Dataset split to run inference on (default: "val").

    Returns:
        Summary statistics dict.
    """
    # ── Lazy import (avoid import-time CUDA init) ──────────────────────
    # Prefer subway_yolo (vendored Ultralytics with custom modules) for
    # loading EMA/SimAM/ECA checkpoints.  Fall back to stock ultralytics.
    try:
        from subway_yolo import YOLO
    except ImportError:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("ERROR: Neither subway_yolo nor ultralytics is installed.")
            sys.exit(1)

    print("=" * 60)
    print("  Hard Negative Collector")
    print("=" * 60)
    print(f"  Model        : {model_weights}")
    print(f"  Data         : {data_yaml}")
    print(f"  Conf threshold: {conf_threshold}")
    print(f"  IoU threshold : {iou_threshold}")
    print(f"  Output       : {output_dir}")
    print(f"  Device       : {device}")
    print()

    print(f"  Split        : {split}")
    print()

    # ── Load model ─────────────────────────────────────────────────────
    print("Loading model...")
    model = YOLO(str(model_weights))
    print(f"  Model loaded: {model_weights}")

    # ── Read data.yaml to find split image/label paths & class names ───
    import yaml as _yaml
    with open(data_yaml, encoding="utf-8") as f:
        ds_cfg = _yaml.safe_load(f)

    # Load class names from data YAML (authoritative for this model)
    CLASS_NAMES = _load_class_names(data_yaml)
    NC = len(CLASS_NAMES)
    print(f"  Classes ({NC}): {CLASS_NAMES}")

    ds_path = Path(ds_cfg.get("path", "."))
    split_img_rel = ds_cfg.get(split, f"images/{split}")
    split_img_dir = ds_path / split_img_rel
    if not split_img_dir.is_dir():
        # Try relative to data_yaml parent
        split_img_dir = data_yaml.parent / split_img_rel
    if not split_img_dir.is_dir():
        print(f"ERROR: {split} image directory not found: {split_img_dir}")
        sys.exit(1)

    # Resolve label directory — try multiple common YOLO layouts:
    #   Layout A (subway_crops):  {split}/images/  +  {split}/labels/
    #   Layout B (Defect_dataset): images/{split}/  +  labels/{split}/
    #   Layout C (flat):          images/{split}/  +  labels/
    candidates = [
        split_img_dir.parent / "labels",                                      # A
        split_img_dir.parent.parent / "labels" / split_img_dir.name,          # B
        split_img_dir.parent.parent / "labels",                               # C
    ]
    split_lbl_dir = None
    for cand in candidates:
        if cand.is_dir():
            split_lbl_dir = cand
            break
    if split_lbl_dir is None:
        print(f"WARNING: Label directory not found. Tried: {candidates}. "
              f"Will run inference without GT matching.")
        gt_labels = {}
    else:
        gt_labels = _load_yolo_labels(split_lbl_dir)
        print(f"  GT labels loaded: {len(gt_labels)} images from {split_lbl_dir}")

    # ── Run inference ──────────────────────────────────────────────────
    print(f"\nRunning inference on {split_img_dir}...")
    results = model.predict(
        source=str(split_img_dir),
        imgsz=imgsz,
        conf=conf_threshold,
        iou=0.45,  # NMS IoU
        device=device,
        verbose=False,
        stream=True,
    )

    # ── Collect false positives ────────────────────────────────────────
    fp_stats: Dict[str, Dict] = {name: {"count": 0, "conf_sum": 0.0, "conf_max": 0.0}
                                  for name in CLASS_NAMES}
    fp_crops: Dict[str, List[Tuple[np.ndarray, float]]] = {name: [] for name in CLASS_NAMES}
    total_detections = 0
    total_fps = 0

    for result in tqdm(results, desc="Processing predictions", unit="img"):
        if result is None or result.boxes is None:
            continue

        img_path = Path(result.path)
        stem = img_path.stem

        # Get ground truths for this image
        gts = gt_labels.get(stem, [])

        # Get predictions
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        # Load image once for cropping (only if we have potential FPs)
        img = None

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            conf = float(boxes.conf[i].item())
            xywhn = boxes.xywhn[i].tolist()  # normalised [xc, yc, w, h]

            total_detections += 1

            # Check IoU against all GTs
            is_fp = True
            for gt_cls, gt_xc, gt_yc, gt_w, gt_h in gts:
                iou = _box_iou(
                    (xywhn[0], xywhn[1], xywhn[2], xywhn[3]),
                    (gt_xc, gt_yc, gt_w, gt_h),
                )
                if iou > iou_threshold:
                    is_fp = False
                    break

            if not is_fp:
                continue

            # It's a false positive — crop it
            cls_name = CLASS_NAMES[cls_id] if cls_id < NC else f"class_{cls_id}"
            if cls_name not in fp_crops:
                fp_crops[cls_name] = []
                fp_stats[cls_name] = {"count": 0, "conf_sum": 0.0, "conf_max": 0.0}

            total_fps += 1
            fp_stats[cls_name]["count"] += 1
            fp_stats[cls_name]["conf_sum"] += conf
            fp_stats[cls_name]["conf_max"] = max(fp_stats[cls_name]["conf_max"], conf)

            # Lazy-load image
            if img is None:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

            crop = _crop_fp_region(img, xywhn[0], xywhn[1], xywhn[2], xywhn[3])
            if crop is not None:
                fp_crops[cls_name].append((crop, conf))

    # ── Save crops to disk ─────────────────────────────────────────────
    print(f"\nSaving hard negative crops to {output_dir} ...")
    saved_total = 0
    per_class_saved: Dict[str, int] = {}

    for cls_name in CLASS_NAMES:
        crops = fp_crops.get(cls_name, [])
        if not crops:
            continue

        # Apply per-class cap
        if max_fp_per_class > 0 and len(crops) > max_fp_per_class:
            # Keep highest-confidence FPs
            crops.sort(key=lambda x: x[1], reverse=True)
            crops = crops[:max_fp_per_class]

        cls_out_dir = output_dir / cls_name
        cls_out_dir.mkdir(parents=True, exist_ok=True)

        for idx, (crop, conf) in enumerate(crops):
            out_path = cls_out_dir / f"hn_{cls_name}_{idx:04d}_c{conf:.2f}.jpg"
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        per_class_saved[cls_name] = len(crops)
        saved_total += len(crops)
        print(f"  {cls_name:12s}: {len(crops):>4d} FP crops saved")

    # ── Summary statistics ─────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Collection Summary")
    print(f"{'=' * 60}")
    print(f"  Total detections : {total_detections}")
    print(f"  Total FPs        : {total_fps}")
    print(f"  FP rate          : {total_fps / max(1, total_detections) * 100:.1f}%")
    print(f"  Crops saved      : {saved_total}")
    print()

    for cls_name in sorted(fp_stats, key=lambda n: fp_stats[n]["count"], reverse=True):
        s = fp_stats[cls_name]
        if s["count"] == 0:
            continue
        avg_conf = s["conf_sum"] / s["count"]
        saved = per_class_saved.get(cls_name, 0)
        bar = "█" * min(40, int(s["count"] / max(1, total_fps) * 40))
        print(f"  {cls_name:12s}: {s['count']:>4d} FPs  avg_conf={avg_conf:.3f}  "
              f"max_conf={s['conf_max']:.3f}  saved={saved}  {bar}")

    # ── Write summary JSON ─────────────────────────────────────────────
    summary = {
        "model": str(model_weights),
        "data": str(data_yaml),
        "conf_threshold": conf_threshold,
        "iou_threshold": iou_threshold,
        "total_detections": total_detections,
        "total_fps": total_fps,
        "fp_rate": total_fps / max(1, total_detections),
        "crops_saved": saved_total,
        "per_class": {
            cls_name: {
                "fp_count": fp_stats[cls_name]["count"],
                "avg_confidence": fp_stats[cls_name]["conf_sum"] / max(1, fp_stats[cls_name]["count"]),
                "max_confidence": fp_stats[cls_name]["conf_max"],
                "crops_saved": per_class_saved.get(cls_name, 0),
            }
            for cls_name in CLASS_NAMES
            if fp_stats[cls_name]["count"] > 0
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )
    print(f"\n  Summary saved: {summary_path}")

    return summary


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect hard negative crops from false-positive detections",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/collect_hard_negatives.py --model weights/stage4_best_finetune.pt
  python scripts/collect_hard_negatives.py --model weights/stage4_best_finetune.pt --conf 0.25
  python scripts/collect_hard_negatives.py --model weights/stage4_best_finetune.pt --max-fp 200
""",
    )
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to trained YOLO model (.pt)",
    )
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/train_data_2/data.yaml"),
        help="Path to dataset YAML config (default: data/train_data_2/data.yaml)",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"Minimum confidence for FP collection (default: {DEFAULT_CONF})",
    )
    parser.add_argument(
        "--iou-thresh", type=float, default=DEFAULT_IOU_THRESH,
        help=f"Max IoU with GT to count as FP (default: {DEFAULT_IOU_THRESH})",
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="CUDA device (default: 0, use 'cpu' for CPU)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="Inference image size (default: 1280)",
    )
    parser.add_argument(
        "--max-fp", type=int, default=0,
        help="Max FP crops per class (0 = unlimited, default: 0)",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split to run on (default: 'train' for hard negative mining)",
    )
    args = parser.parse_args()

    collect_hard_negatives(
        model_weights=args.model,
        data_yaml=args.data,
        output_dir=args.output,
        conf_threshold=args.conf,
        iou_threshold=args.iou_thresh,
        device=args.device,
        imgsz=args.imgsz,
        max_fp_per_class=args.max_fp,
        split=args.split,
    )


if __name__ == "__main__":
    main()
