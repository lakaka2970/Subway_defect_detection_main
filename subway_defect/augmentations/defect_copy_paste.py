"""Offline defect-aware Copy-Paste augmentation for small-object detection.

Extracts small-defect patches from training images and pastes them onto
other images at non-overlapping locations, multiplying small-object
training instances.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def _yolo_to_xyxy(box: List[float], w: int, h: int) -> Tuple[int, int, int, int]:
    """Convert YOLO normalized [cx, cy, bw, bh] to pixel [x1, y1, x2, y2]."""
    cx, cy, bw, bh = box[1], box[2], box[3], box[4]
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> List[float]:
    """Convert pixel [x1, y1, x2, y2] to YOLO normalized [cx, cy, bw, bh]."""
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return [cx, cy, bw, bh]


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def copy_paste_defects(
    img_dir: Path,
    label_dir: Path,
    output_img_dir: Path,
    output_label_dir: Path,
    paste_prob: float = 0.30,
    max_pastes: int = 3,
    min_bbox_size: int = 8,
    max_bbox_size: int = 32,
    edge_margin: int = 50,
    iou_threshold: float = 0.10,
    alpha_blend: float = 0.85,
    seed: int = 42,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Run offline copy-paste augmentation.

    Returns stats dict with counts.
    """
    random.seed(seed)
    np.random.seed(seed)

    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    output_img_dir = Path(output_img_dir)
    output_label_dir = Path(output_label_dir)

    # Collect all labelled images with small defects
    defect_bank: List[Tuple[np.ndarray, int, int, int]] = []  # (patch, cls_id, pw, ph)
    image_files = sorted(
        [f for f in img_dir.glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )

    print(f"  Scanning {len(image_files)} images for small defects ({min_bbox_size}-{max_bbox_size}px)...")

    for img_path in image_files:
        lbl_path = label_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        lines = [l.strip() for l in lbl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            continue

        img = None
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            # Estimate pixel size (assume 1280x1280 or read actual)
            bw_norm, bh_norm = float(parts[3]), float(parts[4])
            # Use 1280 as reference (subway_crops are 1280x1280)
            bw_px = bw_norm * 1280
            bh_px = bh_norm * 1280
            side = max(bw_px, bh_px)
            if side < min_bbox_size or side > max_bbox_size:
                continue

            if img is None:
                img = cv2.imread(str(img_path))
                if img is None:
                    break

            h_img, w_img = img.shape[:2]
            x1, y1, x2, y2 = _yolo_to_xyxy([cls_id] + [float(x) for x in parts[1:5]], w_img, h_img)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            patch = img[y1:y2, x1:x2].copy()
            defect_bank.append((patch, cls_id, x2 - x1, y2 - y1))

    print(f"  Defect bank: {len(defect_bank)} small patches extracted")

    if not defect_bank:
        print("  WARNING: No small defects found in size range. Nothing to paste.")
        return {"pasted_images": 0, "total_pastes": 0, "defects_in_bank": 0}

    # Class distribution in bank
    cls_counts: Dict[int, int] = {}
    for _, cls_id, _, _ in defect_bank:
        cls_counts[cls_id] = cls_counts.get(cls_id, 0) + 1
    print(f"  Bank class distribution: {dict(sorted(cls_counts.items()))}")

    if dry_run:
        n_targets = int(len(image_files) * paste_prob)
        print(f"\n  [DRY-RUN] Would paste onto ~{n_targets} images")
        print(f"  [DRY-RUN] Max {max_pastes} pastes per image")
        print(f"  [DRY-RUN] Estimated new instances: ~{n_targets * max_pastes}")
        return {"pasted_images": n_targets, "total_pastes": n_targets * max_pastes, "defects_in_bank": len(defect_bank)}

    # Create output directories
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    pasted_images = 0
    total_pastes = 0

    print(f"\n  Generating copy-paste augmented images...")

    for idx, img_path in enumerate(image_files):
        lbl_path = label_dir / (img_path.stem + ".txt")

        # Read existing labels
        existing_lines: List[str] = []
        if lbl_path.exists():
            existing_lines = [l.strip() for l in lbl_path.read_text(encoding="utf-8").splitlines() if l.strip()]

        # Decide whether to paste onto this image
        if random.random() > paste_prob:
            # Just copy original
            shutil.copy2(img_path, output_img_dir / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, output_label_dir / lbl_path.name)
            else:
                (output_label_dir / (img_path.stem + ".txt")).write_text("", encoding="utf-8")
            continue

        # Load image for pasting
        img = cv2.imread(str(img_path))
        if img is None:
            shutil.copy2(img_path, output_img_dir / img_path.name)
            if lbl_path.exists():
                shutil.copy2(lbl_path, output_label_dir / lbl_path.name)
            continue

        h_img, w_img = img.shape[:2]

        # Parse existing boxes for IoU check
        existing_boxes: List[Tuple[int, int, int, int]] = []
        for line in existing_lines:
            parts = line.split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                box = _yolo_to_xyxy([cls_id] + [float(x) for x in parts[1:5]], w_img, h_img)
                existing_boxes.append(box)

        # Paste defects
        n_pastes = random.randint(1, max_pastes)
        new_lines: List[str] = []
        pasted_this_img = 0

        for _ in range(n_pastes):
            patch, cls_id, pw, ph = random.choice(defect_bank)

            # Find valid paste location
            attempts = 0
            placed = False
            while attempts < 20 and not placed:
                attempts += 1
                px = random.randint(edge_margin, max(edge_margin, w_img - pw - edge_margin))
                py = random.randint(edge_margin, max(edge_margin, h_img - ph - edge_margin))
                paste_box = (px, py, px + pw, py + ph)

                # Check IoU with existing + already pasted boxes
                overlap = False
                for eb in existing_boxes:
                    if _iou(paste_box, eb) > iou_threshold:
                        overlap = True
                        break
                if overlap:
                    continue

                # Alpha blend paste
                roi = img[py:py + ph, px:px + pw]
                if roi.shape[0] != ph or roi.shape[1] != pw:
                    continue
                img[py:py + ph, px:px + pw] = cv2.addWeighted(
                    patch, alpha_blend, roi, 1 - alpha_blend, 0
                )

                # Record new label
                yolo_box = _xyxy_to_yolo(px, py, px + pw, py + ph, w_img, h_img)
                new_lines.append(f"{cls_id} {yolo_box[0]:.6f} {yolo_box[1]:.6f} {yolo_box[2]:.6f} {yolo_box[3]:.6f}")
                existing_boxes.append(paste_box)
                pasted_this_img += 1
                placed = True

        if pasted_this_img > 0:
            pasted_images += 1
            total_pastes += pasted_this_img
            # Save augmented image with _cp suffix
            out_name = img_path.stem + "_cp" + img_path.suffix
            cv2.imwrite(str(output_img_dir / out_name), img)
            all_lines = existing_lines + new_lines
            (output_label_dir / (img_path.stem + "_cp.txt")).write_text(
                "\n".join(all_lines) + "\n", encoding="utf-8"
            )

        # Also copy original
        shutil.copy2(img_path, output_img_dir / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, output_label_dir / lbl_path.name)
        else:
            (output_label_dir / (img_path.stem + ".txt")).write_text("", encoding="utf-8")

        if (idx + 1) % 500 == 0:
            print(f"    [{idx+1}/{len(image_files)}] pasted={pasted_images}, total_pastes={total_pastes}")

    print(f"\n  Done: {pasted_images} images augmented, {total_pastes} defects pasted")
    print(f"  Output: {output_img_dir} ({len(list(output_img_dir.glob('*')))} files)")

    return {
        "pasted_images": pasted_images,
        "total_pastes": total_pastes,
        "defects_in_bank": len(defect_bank),
    }
