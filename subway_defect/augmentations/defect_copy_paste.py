"""
Defect-aware Copy-Paste augmentation for small-object defect detection.

Unlike Ultralytics' built-in CopyPaste (which copies entire images), this
module extracts individual small-defect regions from source images and
pastes them onto target images — multiplying the number of small-object
training instances while preserving realistic context.

Key design decisions:
  - Only pastes **small objects** (< max_bbox_size px) — large defects don't
    benefit from copy-paste and violate realism.
  - IoU check against existing annotations prevents label collisions.
  - Edge margin prevents pasted objects from being cut by mosaic boundaries.
  - Adaptive blending (alpha compositing) reduces hard-edge artefacts.

Usage (offline, pre-training)::

    from subway_defect.augmentations.defect_copy_paste import copy_paste_defects

    copy_paste_defects(
        img_dir=Path("data/subway_crops/train/images"),
        label_dir=Path("data/subway_crops/train/labels"),
        output_img_dir=Path("data/subway_crops_cp/train/images"),
        output_label_dir=Path("data/subway_crops_cp/train/labels"),
    )
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_PASTE_PROB = 0.30       # probability a target image receives pastes
DEFAULT_MAX_PASTES = 3          # max pasted defects per target image
DEFAULT_MIN_BBOX_SIZE = 8       # min bbox side in px (smaller = noise)
DEFAULT_MAX_BBOX_SIZE = 32      # max bbox side in px (larger = unrealistic)
DEFAULT_EDGE_MARGIN = 50        # px from image edge to avoid
DEFAULT_IOU_THRESHOLD = 0.10    # max allowed IoU with existing annotations
DEFAULT_ALPHA_BLEND = 0.85      # alpha for pasted region (1.0 = hard edge)
SEED = 42


# ── Core logic ───────────────────────────────────────────────────────────────

def _box_iou_px(
    box1: Tuple[int, int, int, int],
    box2: Tuple[int, int, int, int],
) -> float:
    """Compute IoU between two pixel-coordinate boxes (x1, y1, x2, y2)."""
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2
    inter_x1 = max(x1a, x1b)
    inter_y1 = max(y1a, y1b)
    inter_x2 = min(x2a, x2b)
    inter_y2 = min(y2a, y2b)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _extract_defect_patches(
    img: np.ndarray,
    labels: List[Tuple[int, float, float, float, float]],
    img_w: int, img_h: int,
    min_size: int = DEFAULT_MIN_BBOX_SIZE,
    max_size: int = DEFAULT_MAX_BBOX_SIZE,
) -> List[Tuple[np.ndarray, np.ndarray, int]]:
    """Extract small-defect patches from an image.

    Returns:
        List of ``(patch_bgr, patch_mask, cls_id)`` tuples for small defects.
    """
    patches: List[Tuple[np.ndarray, np.ndarray, int]] = []
    for cls_id, xc_n, yc_n, w_n, h_n in labels:
        bx = int(xc_n * img_w)
        by = int(yc_n * img_h)
        bw = int(w_n * img_w)
        bh = int(h_n * img_h)

        # Only small defects
        max_side = max(bw, bh)
        if max_side < min_size or max_side > max_size:
            continue

        x1 = max(0, bx - bw // 2 - 2)   # +2px margin for cleaner extraction
        y1 = max(0, by - bh // 2 - 2)
        x2 = min(img_w, bx + bw // 2 + 2)
        y2 = min(img_h, by + bh // 2 + 2)

        if x2 <= x1 or y2 <= y1:
            continue

        patch = img[y1:y2, x1:x2].copy()

        # Create a simple mask: threshold-based foreground detection
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Invert if the defect is darker than background (common for metal defects)
        if np.mean(mask) > 127:
            mask = 255 - mask

        patches.append((patch, mask, cls_id))

    return patches


def _random_paste_location(
    patch_h: int, patch_w: int,
    img_w: int, img_h: int,
    existing_boxes: List[Tuple[int, int, int, int]],
    edge_margin: int = DEFAULT_EDGE_MARGIN,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    max_attempts: int = 20,
) -> Optional[Tuple[int, int]]:
    """Find a valid paste location that avoids existing annotations and edges.

    Returns:
        ``(x1, y1)`` top-left corner, or None if no valid location found.
    """
    for _ in range(max_attempts):
        x1 = random.randint(edge_margin, max(edge_margin, img_w - patch_w - edge_margin))
        y1 = random.randint(edge_margin, max(edge_margin, img_h - patch_h - edge_margin))
        x2 = x1 + patch_w
        y2 = y1 + patch_h

        if x2 > img_w - edge_margin or y2 > img_h - edge_margin:
            continue

        candidate = (x1, y1, x2, y2)
        valid = True
        for eb in existing_boxes:
            if _box_iou_px(candidate, eb) > iou_threshold:
                valid = False
                break
        if valid:
            return (x1, y1)

    return None


def copy_paste_defects(
    img_dir: Path,
    label_dir: Path,
    output_img_dir: Path,
    output_label_dir: Path,
    paste_prob: float = DEFAULT_PASTE_PROB,
    max_pastes: int = DEFAULT_MAX_PASTES,
    min_bbox_size: int = DEFAULT_MIN_BBOX_SIZE,
    max_bbox_size: int = DEFAULT_MAX_BBOX_SIZE,
    edge_margin: int = DEFAULT_EDGE_MARGIN,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    alpha_blend: float = DEFAULT_ALPHA_BLEND,
    seed: int = SEED,
    dry_run: bool = False,
) -> Dict:
    """Apply defect-aware Copy-Paste augmentation to a dataset.

    For each target image, with probability *paste_prob*, extracts small
    defect patches from randomly-chosen source images and pastes them onto
    the target at locations that don't collide with existing annotations.

    Args:
        img_dir: Source training images.
        label_dir: Source YOLO-format labels.
        output_img_dir: Where to write augmented images (originals + pasted).
        output_label_dir: Where to write augmented labels.
        paste_prob: Probability a target image receives pastes.
        max_pastes: Maximum pasted defects per target image.
        min_bbox_size: Minimum defect side length in px.
        max_bbox_size: Maximum defect side length in px.
        edge_margin: Margin from image edges in px.
        iou_threshold: Max allowed IoU between pasted and existing boxes.
        alpha_blend: Blending alpha for pasted region (1.0 = hard edge).
        seed: Random seed.
        dry_run: If True, only print statistics.

    Returns:
        Statistics dict.
    """
    random.seed(seed)
    np.random.seed(seed)

    # ── Prepare image index ──────────────────────────────────────────────
    img_paths = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    if not img_paths:
        print(f"ERROR: No images found in {img_dir}")
        return {}

    print(f"  Found {len(img_paths)} images in {img_dir}")

    # ── Helper: load labels for one image ────────────────────────────────
    def _load_one_label(imp: Path) -> List[Tuple[int, float, float, float, float]]:
        lbl_path = label_dir / f"{imp.stem}.txt"
        labels: List[Tuple[int, float, float, float, float]] = []
        if lbl_path.exists():
            for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        labels.append((int(parts[0]), float(parts[1]), float(parts[2]),
                                       float(parts[3]), float(parts[4])))
                    except (ValueError, IndexError):
                        continue
        return labels

    # ── Pass 1: scan images one-at-a-time to extract small-defect patches ──
    print("Extracting small-defect patches (streaming)...")
    patch_pool: List[Tuple[np.ndarray, np.ndarray, int, Tuple[int, int]]] = []
    # (patch_img, mask, cls_id, (orig_w, orig_h))

    for i, imp in enumerate(img_paths):
        img = cv2.imread(str(imp))
        if img is None:
            continue
        labels = _load_one_label(imp)
        h, w = img.shape[:2]
        patches = _extract_defect_patches(img, labels, w, h, min_bbox_size, max_bbox_size)
        for patch, mask, cls_id in patches:
            patch_h, patch_w = patch.shape[:2]
            patch_pool.append((patch, mask, cls_id, (patch_w, patch_h)))
        # Release image memory immediately — we only need the patches
        del img, labels

        if (i + 1) % 500 == 0:
            print(f"    Scanned {i + 1}/{len(img_paths)} images, "
                  f"{len(patch_pool)} patches collected")

    print(f"  Extracted {len(patch_pool)} small-defect patches "
          f"(size range: {min_bbox_size}-{max_bbox_size} px)")

    if not patch_pool:
        print("WARNING: No extractable small-defect patches found. "
              "Check min/max bbox size settings.")
        return {"patches_extracted": 0, "images_augmented": 0, "total_pastes": 0}

    # ── Pass 2: apply copy-paste one image at a time ────────────────────
    if not dry_run:
        output_img_dir.mkdir(parents=True, exist_ok=True)
        output_label_dir.mkdir(parents=True, exist_ok=True)

    images_augmented = 0
    total_pastes = 0

    for imp in tqdm(img_paths, desc="Copy-paste augmentation", unit="img"):
        img = cv2.imread(str(imp))
        if img is None:
            continue
        labels = _load_one_label(imp)
        h, w = img.shape[:2]

        # Copy original (all images go to output, even if not augmented)
        if not dry_run:
            cv2.imwrite(str(output_img_dir / imp.name), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

        # Decide whether to paste on this image
        if random.random() > paste_prob:
            # Just copy original label
            if not dry_run:
                lbl_out = output_label_dir / f"{imp.stem}.txt"
                lbl_lines = [f"{c} {xc:.6f} {yc:.6f} {w_:.6f} {h_:.6f}"
                            for c, xc, yc, w_, h_ in labels]
                lbl_out.write_text("\n".join(lbl_lines) + "\n" if lbl_lines else "\n",
                                   encoding="utf-8")
            del img, labels
            continue

        # Get existing bboxes in pixel coords for IoU checking
        existing_boxes: List[Tuple[int, int, int, int]] = []
        for c, xc_n, yc_n, w_n, h_n in labels:
            bx = int(xc_n * w)
            by = int(yc_n * h)
            bw = int(w_n * w)
            bh = int(h_n * h)
            existing_boxes.append((bx - bw // 2, by - bh // 2,
                                   bx + bw // 2, by + bh // 2))

        # Select patches to paste
        n_to_paste = random.randint(1, max_pastes)
        augmented_img = img.copy()
        new_labels: List[Tuple[int, float, float, float, float]] = list(labels)

        for _ in range(n_to_paste):
            if not patch_pool:
                break

            patch, mask, cls_id, (patch_w, patch_h) = random.choice(patch_pool)

            loc = _random_paste_location(patch_h, patch_w, w, h, existing_boxes,
                                         edge_margin, iou_threshold)
            if loc is None:
                continue

            px1, py1 = loc
            px2, py2 = px1 + patch_w, py1 + patch_h

            # Blend patch into target image
            roi = augmented_img[py1:py2, px1:px2].astype(np.float32)
            patch_f = patch.astype(np.float32)

            if mask.shape[:2] != patch.shape[:2]:
                mask = cv2.resize(mask, (patch_w, patch_h))

            mask_f = (mask / 255.0).astype(np.float32)
            if len(mask_f.shape) == 2:
                mask_f = mask_f[..., np.newaxis]

            blended = patch_f * mask_f * alpha_blend + roi * (1 - mask_f * alpha_blend)
            augmented_img[py1:py2, px1:px2] = blended.clip(0, 255).astype(np.uint8)

            # Add new label
            new_xc = (px1 + patch_w / 2) / w
            new_yc = (py1 + patch_h / 2) / h
            new_w = patch_w / w
            new_h = patch_h / h
            new_labels.append((cls_id, new_xc, new_yc, new_w, new_h))

            # Add to existing boxes to prevent overlaps within this image
            existing_boxes.append((px1, py1, px2, py2))
            total_pastes += 1

        # Save augmented image and merged labels
        if not dry_run:
            cv2.imwrite(str(output_img_dir / imp.name), augmented_img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            lbl_lines = [f"{c} {xc:.6f} {yc:.6f} {w_:.6f} {h_:.6f}"
                        for c, xc, yc, w_, h_ in new_labels]
            lbl_out = output_label_dir / f"{imp.stem}.txt"
            lbl_out.write_text("\n".join(lbl_lines) + "\n" if lbl_lines else "\n",
                               encoding="utf-8")

        images_augmented += 1
        # Release per-image memory
        del img, augmented_img, labels, new_labels

    stats = {
        "total_images": len(img_paths),
        "patches_extracted": len(patch_pool),
        "images_augmented": images_augmented,
        "total_pastes": total_pastes,
        "avg_pastes_per_image": total_pastes / max(1, images_augmented),
    }

    print(f"\n  Copy-Paste augmentation complete:")
    print(f"    Images augmented: {images_augmented}/{len(img_paths)}")
    print(f"    Total pastes:     {total_pastes}")
    print(f"    Avg pastes/img:   {stats['avg_pastes_per_image']:.1f}")

    return stats
