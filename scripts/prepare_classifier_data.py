#!/usr/bin/env python3
"""
Prepare classifier training data from YOLO proposals.

Runs the Stage 3 detector on subway_crops, extracts proposals with
1.5-2.0x context crops, and generates state labels for classifier training.

Output structure::

    data/classifier/
    ├── cbhpm/
    │   ├── train/
    │   │   ├── normal/     # CBHPM-class proposals with no GT match
    │   │   └── missing/    # CBHPM-class proposals matching GT
    │   ├── val/
    │   │   ├── normal/
    │   │   └── missing/
    │   └── test/
    │       ├── normal/
    │       └── missing/
    └── vhbnm_vhbnl/
        ├── train/
        │   ├── normal/
        │   ├── missing/
        │   ├── loose/
        │   └── ambiguous/
        ...

Usage::

    python scripts/prepare_classifier_data.py
    python scripts/prepare_classifier_data.py --model weights/stage3_main.pt --context-scale 1.75
    python scripts/prepare_classifier_data.py --dry-run
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import cv2
    import numpy as np
except ImportError:
    print("ERROR: opencv-python and numpy required")
    sys.exit(1)


SEED = 42
CONTEXT_SCALE = 1.75
MIN_CROP_SIZE = 32

# Class indices in the 7-class training set
CLASS_NAMES = ["VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM"]

# State mapping: which classes map to which classifier task
# CBHPM (idx 5): binary — missing (has GT) vs normal (FP, no GT)
# VHBNM (idx 0) + VHBNL (idx 1): 4-class — missing/loose/normal/ambiguous
CBHPM_IDX = 5
VHBNM_IDX = 0
VHBNL_IDX = 1


def extract_context_crop(
    img: np.ndarray, box_xyxy: Tuple[int, int, int, int], context_scale: float
) -> np.ndarray | None:
    """Extract a context crop around a detection box.

    Args:
        img: Source image (H, W, 3).
        box_xyxy: (x1, y1, x2, y2) pixel coordinates.
        context_scale: Expansion factor (1.5-2.0).

    Returns:
        Cropped image or None if too small.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    bw, bh = x2 - x1, y2 - y1

    # Expand by context_scale
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ctx_w = bw * context_scale
    ctx_h = bh * context_scale

    # Ensure minimum size
    ctx_w = max(ctx_w, MIN_CROP_SIZE)
    ctx_h = max(ctx_h, MIN_CROP_SIZE)

    # Compute crop bounds
    cx1 = int(max(0, cx - ctx_w / 2))
    cy1 = int(max(0, cy - ctx_h / 2))
    cx2 = int(min(w, cx + ctx_w / 2))
    cy2 = int(min(h, cy + ctx_h / 2))

    if cx2 - cx1 < MIN_CROP_SIZE or cy2 - cy1 < MIN_CROP_SIZE:
        return None

    return img[cy1:cy2, cx1:cx2]


def box_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def yolo_to_xyxy(cx, cy, w, h, img_w, img_h):
    """Convert YOLO normalized (cx, cy, w, h) to pixel (x1, y1, x2, y2)."""
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return (max(0, x1), max(0, y1), min(img_w, x2), min(img_h, y2))


def prepare_classifier_data(
    model_path: str,
    data_root: Path,
    output_root: Path,
    context_scale: float = 1.75,
    conf_threshold: float = 0.10,
    iou_match_threshold: float = 0.30,
    device: str = "0",
    imgsz: int = 1280,
    dry_run: bool = False,
) -> Dict:
    """Extract classifier training data from YOLO proposals.

    Args:
        model_path: Path to trained YOLO weights.
        data_root: Path to subway_crops dataset root.
        output_root: Output directory for classifier data.
        context_scale: Context expansion factor for crops.
        conf_threshold: Minimum detection confidence to consider.
        iou_match_threshold: IoU threshold to match proposal to GT.
        device: CUDA device.
        imgsz: Inference image size.
        dry_run: Print stats without writing files.

    Returns:
        Statistics dict.
    """
    from subway_yolo import YOLO

    random.seed(SEED)

    model = YOLO(model_path)
    stats = defaultdict(int)

    # Process both train and val splits
    for split in ["train", "val"]:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"
        if not img_dir.is_dir():
            print(f"  [SKIP] {img_dir} not found")
            continue

        images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        print(f"\n  Processing {split}: {len(images)} images")

        for idx, img_path in enumerate(images):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            # Load GT labels
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            gt_boxes: List[Tuple[int, Tuple[int, int, int, int]]] = []
            if lbl_path.exists():
                for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        xyxy = yolo_to_xyxy(
                            float(parts[1]), float(parts[2]),
                            float(parts[3]), float(parts[4]),
                            img_w, img_h,
                        )
                        gt_boxes.append((cls_id, xyxy))

            # Run YOLO inference
            results = model(img, conf=conf_threshold, imgsz=imgsz, verbose=False, device=device)
            if len(results) == 0 or results[0].boxes is None:
                continue

            boxes = results[0].boxes
            for i in range(len(boxes.cls)):
                det_cls = int(boxes.cls[i])
                det_conf = float(boxes.conf[i])
                det_xywh = boxes.xywh[i].cpu().numpy()
                det_xyxy = (
                    int(det_xywh[0] - det_xywh[2] / 2),
                    int(det_xywh[1] - det_xywh[3] / 2),
                    int(det_xywh[0] + det_xywh[2] / 2),
                    int(det_xywh[1] + det_xywh[3] / 2),
                )

                # Determine state based on GT match
                state = _determine_state(det_cls, det_xyxy, gt_boxes, iou_match_threshold)
                if state is None:
                    continue

                # Determine which classifier task this belongs to
                task = _get_classifier_task(det_cls)
                if task is None:
                    continue

                stats[f"{task}_{state}"] += 1

                if dry_run:
                    continue

                # Extract context crop
                crop = extract_context_crop(img, det_xyxy, context_scale)
                if crop is None:
                    continue

                # Save crop
                out_dir = output_root / task / split / state
                out_dir.mkdir(parents=True, exist_ok=True)
                crop_name = f"{img_path.stem}_det{i:03d}_c{det_conf:.2f}.jpg"
                cv2.imwrite(str(out_dir / crop_name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

            if (idx + 1) % 200 == 0:
                print(f"    [{idx+1}/{len(images)}] processed", flush=True)

    # Print statistics
    print(f"\n  {'='*50}")
    print(f"  Classifier Data Statistics")
    print(f"  {'='*50}")
    for key in sorted(stats.keys()):
        print(f"    {key}: {stats[key]}")

    return dict(stats)


def _determine_state(
    det_cls: int,
    det_xyxy: Tuple[int, int, int, int],
    gt_boxes: List[Tuple[int, Tuple[int, int, int, int]]],
    iou_threshold: float,
) -> str | None:
    """Determine the state label for a detection.

    Returns:
        State string or None if this detection should be skipped.
    """
    # Check if detection matches any GT box of the same class
    best_iou = 0.0
    matched_gt_cls = None
    for gt_cls, gt_xyxy in gt_boxes:
        iou = box_iou(det_xyxy, gt_xyxy)
        if iou > best_iou:
            best_iou = iou
            matched_gt_cls = gt_cls

    if best_iou >= iou_threshold and matched_gt_cls == det_cls:
        # True positive — this is a real defect
        if det_cls == CBHPM_IDX:
            return "missing"
        elif det_cls == VHBNM_IDX:
            return "missing"
        elif det_cls == VHBNL_IDX:
            return "loose"
        else:
            return None  # Other classes not in scope for now
    elif best_iou < iou_threshold:
        # False positive — no matching GT
        if det_cls == CBHPM_IDX:
            return "normal"
        elif det_cls in (VHBNM_IDX, VHBNL_IDX):
            return "normal"
        else:
            return None
    else:
        # Matched different class — ambiguous
        if det_cls in (VHBNM_IDX, VHBNL_IDX):
            return "ambiguous"
        return None


def _get_classifier_task(det_cls: int) -> str | None:
    """Map detection class to classifier task name."""
    if det_cls == CBHPM_IDX:
        return "cbhpm"
    elif det_cls in (VHBNM_IDX, VHBNL_IDX):
        return "vhbnm_vhbnl"
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare classifier training data from YOLO proposals"
    )
    parser.add_argument(
        "--model", type=str, default="weights/stage3_main.pt",
        help="Path to trained YOLO detector weights",
    )
    parser.add_argument(
        "--data", type=Path, default=Path("data/subway_crops"),
        help="Path to subway_crops dataset root",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/classifier"),
        help="Output directory for classifier data",
    )
    parser.add_argument(
        "--context-scale", type=float, default=1.75,
        help="Context expansion factor around bbox (default: 1.75)",
    )
    parser.add_argument(
        "--conf", type=float, default=0.10,
        help="Minimum detection confidence (default: 0.10)",
    )
    parser.add_argument(
        "--iou-thresh", type=float, default=0.30,
        help="IoU threshold for GT matching (default: 0.30)",
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA device")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference image size")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only")
    args = parser.parse_args()

    print("=" * 60)
    print("  Classifier Data Preparation")
    print("=" * 60)
    print(f"  Model:         {args.model}")
    print(f"  Data:          {args.data}")
    print(f"  Output:        {args.output}")
    print(f"  Context scale: {args.context_scale}")
    print(f"  Conf thresh:   {args.conf}")
    print(f"  IoU thresh:    {args.iou_thresh}")
    print(f"  Dry run:       {args.dry_run}")
    print()

    stats = prepare_classifier_data(
        model_path=args.model,
        data_root=args.data,
        output_root=args.output,
        context_scale=args.context_scale,
        conf_threshold=args.conf,
        iou_match_threshold=args.iou_thresh,
        device=args.device,
        imgsz=args.imgsz,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print(f"\n  Classifier data saved to: {args.output}")
        print(f"  Next: python scripts/train_state_classifier.py --task cbhpm")


if __name__ == "__main__":
    main()
