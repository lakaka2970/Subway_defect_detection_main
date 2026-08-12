"""Offline defect-aware Copy-Paste augmentation for small-object detection.

Extracts small-defect patches from training images and pastes them onto
other images at non-overlapping locations, multiplying small-object
training instances.

v2 improvements:
- Poisson blending (cv2.seamlessClone) for realistic boundary transitions
- Adaptive scale jitter (±15%) to increase size diversity
- Color harmonization before pasting to reduce domain gap
- Class-balanced sampling from defect bank (minority classes oversampled)

v3 fixes (2026-07-27):
- Fixed hardcoded 1280×1280 resolution in bbox size filtering (now uses
  actual image dimensions)
- Replaced global RNG state mutation with isolated Generator instances
- Added input validation and cv2.imwrite error checking
- Cleaned up _yolo_to_xyxy API (no longer requires dummy cls_id at [0])
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _yolo_to_xyxy(cx: float, cy: float, bw: float, bh: float,
                  w: int, h: int) -> Tuple[int, int, int, int]:
    """Convert YOLO normalized (cx, cy, bw, bh) to pixel (x1, y1, x2, y2)."""
    x1 = int((cx - bw / 2) * w)
    y1 = int((cy - bh / 2) * h)
    x2 = int((cx + bw / 2) * w)
    y2 = int((cy + bh / 2) * h)
    return max(0, x1), max(0, y1), min(w, x2), min(h, y2)


def _xyxy_to_yolo(x1: int, y1: int, x2: int, y2: int,
                  w: int, h: int) -> List[float]:
    """Convert pixel (x1, y1, x2, y2) to YOLO normalized [cx, cy, bw, bh]."""
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid image dimensions: w={w}, h={h}")
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


def _harmonize_color(patch: np.ndarray, target_roi: np.ndarray) -> np.ndarray:
    """Match patch color statistics to the target ROI (Reinhard transfer).

    Prevents pasted defects from looking like they belong to a different
    lighting / camera condition than the host image.
    """
    if target_roi.size == 0 or patch.size == 0:
        return patch
    src = patch.astype(np.float32)
    ref = target_roi.astype(np.float32)
    for c in range(3):
        s_mean, s_std = src[:, :, c].mean(), src[:, :, c].std() + 1e-6
        r_mean, r_std = ref[:, :, c].mean(), ref[:, :, c].std() + 1e-6
        src[:, :, c] = (src[:, :, c] - s_mean) * (r_std / s_std) + r_mean
    return np.clip(src, 0, 255).astype(np.uint8)


def _poisson_blend(patch: np.ndarray, canvas: np.ndarray,
                   px: int, py: int, ph: int, pw: int) -> bool:
    """Blend patch into canvas at (px, py) using Poisson seamless cloning.

    Falls back to alpha blending if seamlessClone fails (e.g. patch too
    small or touches the image border).
    Returns True on success.
    """
    # seamlessClone requires the mask to be fully inside the destination
    center = (px + pw // 2, py + ph // 2)
    mask = np.ones((ph, pw), dtype=np.uint8) * 255
    try:
        result = cv2.seamlessClone(patch, canvas, mask, center, cv2.NORMAL_CLONE)
        canvas[:] = result
        return True
    except cv2.error:
        return False


def _build_class_balanced_bank(
    defect_bank: List[Tuple[np.ndarray, int, int, int]],
) -> Tuple[List[Tuple[np.ndarray, int, int, int]], List[float]]:
    """Compute per-item sampling weights that oversample minority classes."""
    cls_counts: Dict[int, int] = {}
    for _, cls_id, _, _ in defect_bank:
        cls_counts[cls_id] = cls_counts.get(cls_id, 0) + 1
    if not cls_counts:
        return defect_bank, [1.0] * len(defect_bank)
    max_count = max(cls_counts.values())
    weights = []
    for _, cls_id, _, _ in defect_bank:
        # Inverse-frequency weighting: rare classes get higher weight
        weights.append(max_count / (cls_counts[cls_id] + 1e-6))
    total = sum(weights)
    weights = [w / total for w in weights]
    return defect_bank, weights


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
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    output_img_dir = Path(output_img_dir)
    output_label_dir = Path(output_label_dir)

    if not img_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

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

        img: Optional[np.ndarray] = None
        for line in lines:
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw_norm, bh_norm = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Load image to get actual dimensions for size filtering
            if img is None:
                img = cv2.imread(str(img_path))
                if img is None:
                    break

            h_img, w_img = img.shape[:2]
            bw_px = bw_norm * w_img
            bh_px = bh_norm * h_img
            side = max(bw_px, bh_px)
            if side < min_bbox_size or side > max_bbox_size:
                continue

            x1, y1, x2, y2 = _yolo_to_xyxy(cx, cy, bw_norm, bh_norm, w_img, h_img)
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

    # Build class-balanced sampling weights
    defect_bank, bank_weights = _build_class_balanced_bank(defect_bank)

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
        if rng.random() > paste_prob:
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
                box = _yolo_to_xyxy(
                    float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4]),
                    w_img, h_img,
                )
                existing_boxes.append(box)

        # Paste defects
        n_pastes = rng.randint(1, max_pastes)
        new_lines: List[str] = []
        pasted_this_img = 0

        for _ in range(n_pastes):
            # Class-balanced sampling from defect bank
            idx_choice = rng.choices(range(len(defect_bank)), weights=bank_weights, k=1)[0]
            patch, cls_id, pw, ph = defect_bank[idx_choice]

            # Adaptive scale jitter (±15%) for size diversity
            scale = rng.uniform(0.85, 1.15)
            new_pw = max(4, int(pw * scale))
            new_ph = max(4, int(ph * scale))
            if new_pw != pw or new_ph != ph:
                patch = cv2.resize(patch, (new_pw, new_ph), interpolation=cv2.INTER_LINEAR)
                pw, ph = new_pw, new_ph

            # Find valid paste location
            attempts = 0
            placed = False
            while attempts < 20 and not placed:
                attempts += 1
                x_hi = max(edge_margin + 1, w_img - pw - edge_margin)
                y_hi = max(edge_margin + 1, h_img - ph - edge_margin)
                px = rng.randint(edge_margin, x_hi)
                py = rng.randint(edge_margin, y_hi)
                paste_box = (px, py, px + pw, py + ph)

                # Check IoU with existing + already pasted boxes
                overlap = False
                for eb in existing_boxes:
                    if _iou(paste_box, eb) > iou_threshold:
                        overlap = True
                        break
                if overlap:
                    continue

                roi = img[py:py + ph, px:px + pw]
                if roi.shape[0] != ph or roi.shape[1] != pw:
                    continue

                # Color harmonization: match patch to local background
                patch_harmonized = _harmonize_color(patch, roi)

                # Poisson blending for seamless boundaries (fallback: alpha blend)
                if not _poisson_blend(patch_harmonized, img, px, py, ph, pw):
                    img[py:py + ph, px:px + pw] = cv2.addWeighted(
                        patch_harmonized, alpha_blend, roi, 1 - alpha_blend, 0
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
            out_path = output_img_dir / out_name
            if not cv2.imwrite(str(out_path), img):
                print(f"    WARNING: Failed to write {out_path}")
                continue
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
