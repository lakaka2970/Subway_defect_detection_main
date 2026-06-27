#!/usr/bin/env python3
"""
Generate native-resolution training crops from 5120×5120 source images.

This is the **P0 structural fix** for the "defects shrink to sub-pixel"
problem: instead of resizing the entire 5120px image to 1024px (making a
40px defect just 8px), we crop 1024/1280 patches at native resolution.
Defects retain their original pixel size — a 40px bolt becomes a 40px
feature for the model to learn.

Produces:
  - **Positive crops**: centered on each defect bbox with random offset
    (±100-300 px), preserving 2-4× structural context
  - **Negative crops**: defect-free but visually similar regions (bolts,
    insulators, wire clamps, reflective areas) — essential for reducing
    false positives
  - **Validation split**: grouped by **source image** (not crop), so
    no same-source-image crops leak between train and val

Output layout::

    data/subway_crops/
    ├── train/
    │   ├── images/       # 1024×1024 (or 1280×1280) .jpg crops
    │   └── labels/       # YOLO-format .txt labels (adjusted to crop coords)
    ├── val/
    │   ├── images/
    │   └── labels/
    └── subway_crops.yaml # nc=7, names=[VHBNM, VHBNL, ...]

Usage::

    # Generate 1024px crops (default)
    python scripts/generate_native_crops.py

    # Generate 1280px crops with more negatives
    python scripts/generate_native_crops.py --crop-size 1280 --negatives-per-image 20

    # Dry-run: print statistics without generating
    python scripts/generate_native_crops.py --dry-run

    # Custom source / output directories
    python scripts/generate_native_crops.py \\
        --src data/Defect_dataset/images/train \\
        --labels data/Defect_dataset/labels/train \\
        --output data/subway_crops
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required. Install with: pip install opencv-python")
    sys.exit(1)

try:
    import yaml
except ImportError:
    yaml = None

# ── Constants ────────────────────────────────────────────────────────────

SEED = 42
# Import class registry from project (when run as module)
try:
    from subway_defect.classes import TRAIN_CLASSES as CLASS_NAMES, TRAIN_NC as NC
except ImportError:
    # Fallback for standalone execution
    CLASS_NAMES = [
        "VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
    ]
    NC = len(CLASS_NAMES)

# Default source directories (relative to project root)
DEFAULT_SRC_IMG = Path("data/Defect_dataset/images/train")
DEFAULT_SRC_LBL = Path("data/Defect_dataset/labels/train")
DEFAULT_OUTPUT = Path("data/subway_crops")

# Minimum bbox visibility fraction in crop (bbox must be at least this
# much inside the crop window to be included as a positive label)
MIN_BBOX_VISIBILITY = 0.60


# ── Helpers ──────────────────────────────────────────────────────────────

def _load_labels(label_dir: Path) -> Dict[str, List[Tuple[int, float, float, float, float]]]:
    """Load all YOLO-format labels, keyed by image stem.

    Returns:
        ``{stem: [(cls_id, xc_norm, yc_norm, w_norm, h_norm), ...]}``
    """
    labels: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
    for lbl_file in sorted(label_dir.glob("*.txt")):
        stem = lbl_file.stem
        entries: List[Tuple[int, float, float, float, float]] = []
        for line in lbl_file.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except (ValueError, IndexError):
                continue
            entries.append((cls_id, xc, yc, w, h))
        if entries:
            labels[stem] = entries
    return labels


def _find_image(stem: str, img_dir: Path) -> Optional[Path]:
    """Find an image file by stem (tries common extensions)."""
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _bbox_overlap(
    cx: float, cy: float,  # crop center (px)
    crop_w: int, crop_h: int,
    bx_norm: float, by_norm: float, bw_norm: float, bh_norm: float,
    img_w: int, img_h: int,
) -> float:
    """Compute IoU between a crop window and a normalized bbox.

    Returns the fraction of the bbox that falls within the crop (0-1).
    """
    # Convert bbox to pixel coords
    bx = bx_norm * img_w
    by = by_norm * img_h
    bw = bw_norm * img_w
    bh = bh_norm * img_h
    b_x1, b_y1 = bx - bw / 2, by - bh / 2
    b_x2, b_y2 = bx + bw / 2, by + bh / 2

    # Crop coords
    c_x1 = cx - crop_w / 2
    c_y1 = cy - crop_h / 2
    c_x2 = cx + crop_w / 2
    c_y2 = cy + crop_h / 2

    # Intersection
    inter_x1 = max(b_x1, c_x1)
    inter_y1 = max(b_y1, c_y1)
    inter_x2 = min(b_x2, c_x2)
    inter_y2 = min(b_y2, c_y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    bbox_area = bw * bh
    if bbox_area <= 0:
        return 0.0
    return inter_area / bbox_area


def _normalize_bbox_to_crop(
    bx_norm: float, by_norm: float, bw_norm: float, bh_norm: float,
    img_w: int, img_h: int,
    cx: float, cy: float, crop_w: int, crop_h: int,
) -> Optional[Tuple[float, float, float, float]]:
    """Convert a normalized (image-coord) bbox to a normalized (crop-coord) bbox.

    Returns ``(xc_crop, yc_crop, w_crop, h_crop)`` normalized to crop dimensions,
    or ``None`` if the bbox is completely outside the crop.
    """
    bx = bx_norm * img_w
    by = by_norm * img_h
    bw = bw_norm * img_w
    bh = bh_norm * img_h

    # Shift to crop origin
    crop_x1 = cx - crop_w / 2
    crop_y1 = cy - crop_h / 2

    bx_crop = bx - crop_x1
    by_crop = by - crop_y1

    # Normalize by crop size
    xc_crop = bx_crop / crop_w
    yc_crop = by_crop / crop_h
    w_crop = bw / crop_w
    h_crop = bh / crop_h

    # Clamp
    xc_crop = max(0.0, min(1.0, xc_crop))
    yc_crop = max(0.0, min(1.0, yc_crop))
    w_crop = min(1.0, w_crop)
    h_crop = min(1.0, h_crop)

    if w_crop <= 0.001 or h_crop <= 0.001:
        return None

    return (xc_crop, yc_crop, w_crop, h_crop)


def _compute_debiased_offset(
    img_w: int, img_h: int,
    bx: int, by: int,
    crop_size: int,
    rng: random.Random,
) -> Tuple[int, int]:
    """Compute crop center (cx, cy) with position de-biasing.

    Instead of always centering ± small random offset (which teaches the model
    that defects always appear near the crop center), this function varies the
    defect's position within the crop systematically:

    - 30% center: defect near crop center (±50-100px jitter)
    - 30% off-center: defect in one of 4 quadrants (200-400px from center)
    - 25% near-edge: defect within 100-250px of one crop edge
    - 15% corner: defect within 200px of two adjacent crop edges

    Returns (cx, cy) constrained to valid crop bounds.
    """
    zone = rng.choices(
        ["center", "off_center", "near_edge", "corner"],
        weights=[0.30, 0.30, 0.25, 0.15],
        k=1,
    )[0]

    half = crop_size // 2

    if zone == "center":
        # Defect near crop center with mild jitter
        dx = rng.randint(50, 100) * rng.choice([-1, 1])
        dy = rng.randint(50, 100) * rng.choice([-1, 1])
        cx = bx + dx
        cy = by + dy

    elif zone == "off_center":
        # Defect in one quadrant of the crop (200-400px from center)
        qx = rng.choice([-1, 1])
        qy = rng.choice([-1, 1])
        dx = rng.randint(200, 400) * qx
        dy = rng.randint(200, 400) * qy
        cx = bx + dx
        cy = by + dy

    elif zone == "near_edge":
        # Defect near one of the 4 crop edges
        edge = rng.choice(["top", "bottom", "left", "right"])
        margin = rng.randint(100, 250)
        if edge == "top":
            cx = bx + rng.randint(-150, 150)
            cy = by + (half - margin)  # defect near top of crop
        elif edge == "bottom":
            cx = bx + rng.randint(-150, 150)
            cy = by - (half - margin)  # defect near bottom of crop
        elif edge == "left":
            cx = bx + (half - margin)  # defect near left of crop
            cy = by + rng.randint(-150, 150)
        else:  # "right"
            cx = bx - (half - margin)  # defect near right of crop
            cy = by + rng.randint(-150, 150)

    else:  # "corner"
        # Defect near one of the 4 crop corners
        corner_x = rng.choice([-1, 1])
        corner_y = rng.choice([-1, 1])
        margin_x = rng.randint(50, 200)
        margin_y = rng.randint(50, 200)
        cx = bx + (half - margin_x) * corner_x
        cy = by + (half - margin_y) * corner_y

    # Clamp to valid crop center range
    cx = max(half, min(img_w - half, int(cx)))
    cy = max(half, min(img_h - half, int(cy)))
    return cx, cy


# ── Main crop generation ─────────────────────────────────────────────────

def generate_crops(
    img_dir: Path,
    label_dir: Path,
    output_dir: Path,
    crop_size: int = 1024,
    stride: int = 512,
    negatives_per_image: int = 25,
    val_ratio: float = 0.2,
    random_offset_range: Tuple[int, int] = (100, 300),
    dry_run: bool = False,
    balance: bool = False,
    minority_multiplier: int = 2,
    debiasing: bool = False,
) -> Dict:
    """Generate native-resolution training crops.

    Args:
        img_dir: Directory containing source images (5120×5120).
        label_dir: Directory containing YOLO-format labels.
        output_dir: Root output directory.
        crop_size: Size of square crops in pixels.
        stride: Sliding-window stride for negative crop sampling.
        negatives_per_image: Number of negative crops to sample per source image.
        val_ratio: Fraction of source images to reserve for validation.
        random_offset_range: (min, max) px for random offset from bbox center.
        dry_run: If True, only print statistics without generating files.
        balance: If True, minority classes (below median count) get extra crops.
        minority_multiplier: Extra crops per defect for minority classes.
        debiasing: If True, systematically vary defect position within crop
            (center / off-center / near-edge / corner) to prevent the model
            from learning that defects always appear at the crop center.

    Returns:
        Statistics dict.
    """
    random.seed(SEED)
    labels = _load_labels(label_dir)

    # Find all source images that have labels
    img_stems: List[str] = []
    for stem in sorted(labels):
        img_path = _find_image(stem, img_dir)
        if img_path:
            img_stems.append(stem)

    if not img_stems:
        print(f"ERROR: No matched image-label pairs found!")
        print(f"       Images dir: {img_dir}")
        print(f"       Labels dir: {label_dir}")
        sys.exit(1)

    print(f"Found {len(img_stems)} matched image-label pairs")

    # ── Pre-count per-class bbox instances for balancing ───────────────
    per_class_bbox_count: Dict[int, int] = defaultdict(int)
    for stem in img_stems:
        for cls_id, *_ in labels.get(stem, []):
            if cls_id < NC:
                per_class_bbox_count[cls_id] += 1

    minority_cls_ids: Set[int] = set()
    if balance and per_class_bbox_count:
        counts = sorted(per_class_bbox_count.values())
        median_count = counts[len(counts) // 2]
        minority_cls_ids = {c for c, n in per_class_bbox_count.items() if n < median_count}
        print(f"  Class-balance mode: median={median_count} bboxes/class")
        print(f"  Minority classes (×{minority_multiplier} crops): "
              f"{[CLASS_NAMES[c] for c in sorted(minority_cls_ids)]}")
        for cls_id in sorted(per_class_bbox_count):
            flag = " ★" if cls_id in minority_cls_ids else ""
            expected = per_class_bbox_count[cls_id]
            if cls_id in minority_cls_ids:
                expected *= minority_multiplier
            print(f"    {CLASS_NAMES[cls_id]:12s}: {per_class_bbox_count[cls_id]:>4d} bboxes"
                  f" → ~{expected} crops{flag}")

    # ── Train/val split by source image (NOT by crop) ─────────────────
    shuffled = sorted(img_stems)
    random.shuffle(shuffled)
    split_idx = max(1, int(len(shuffled) * (1 - val_ratio)))
    train_stems = set(shuffled[:split_idx])
    val_stems = set(shuffled[split_idx:])
    print(f"Split: {len(train_stems)} train / {len(val_stems)} val source images")

    stats = {
        "crop_size": crop_size,
        "train_images": len(train_stems),
        "val_images": len(val_stems),
        "positive_crops": {"train": 0, "val": 0},
        "negative_crops": {"train": 0, "val": 0},
        "defect_boxes": {"train": 0, "val": 0},
        "per_class_boxes": {name: 0 for name in CLASS_NAMES},
    }

    if dry_run:
        # Count expected crops
        for stem in img_stems:
            entries = labels.get(stem, [])
            split_tag = "train" if stem in train_stems else "val"
            stats["positive_crops"][split_tag] += len(entries)
            stats["negative_crops"][split_tag] += negatives_per_image
            for cls_id, *_ in entries:
                if cls_id < NC:
                    stats["per_class_boxes"][CLASS_NAMES[cls_id]] += 1
        return stats

    # ── Generate crops ─────────────────────────────────────────────────
    print(f"\nGenerating crops (this may take a while — progress per image):\n", flush=True)
    for split_tag, stems in (("train", train_stems), ("val", val_stems)):
        out_img_dir = output_dir / split_tag / "images"
        out_lbl_dir = output_dir / split_tag / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_count = 0
        box_count = 0
        total_stems = len(stems)

        for idx, stem in enumerate(sorted(stems)):
            img_path = _find_image(stem, img_dir)
            if img_path is None:
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue
            img_h, img_w = img.shape[:2]

            entries = labels.get(stem, [])
            used_centers: Set[Tuple[int, int]] = set()

            # ── Positive crops (one per defect bbox; extra for minority) ─
            for cls_id, xc_norm, yc_norm, w_norm, h_norm in entries:
                # Bbox center in pixel coords
                bx = int(xc_norm * img_w)
                by = int(yc_norm * img_h)

                # Class-balanced: minority classes get extra crops with varied offsets
                n_crops = 1
                offset_range = random_offset_range
                if cls_id in minority_cls_ids:
                    n_crops = minority_multiplier
                    # Tighter offsets for extra crops → different but still valid views
                    offset_range_extra = (50, 150)

                for crop_idx in range(n_crops):
                    if debiasing:
                        # Position de-biasing: systematically vary defect
                        # position (center/off-center/edge/corner) to prevent
                        # the model from learning position shortcuts.
                        cx, cy = _compute_debiased_offset(
                            img_w, img_h, bx, by, crop_size, random,
                        )
                    elif crop_idx == 0:
                        off_min, off_max = random_offset_range
                        # Random offset from center (avoid perfect centering)
                        offset_x = random.randint(off_min, off_max) * random.choice([-1, 1])
                        offset_y = random.randint(off_min, off_max) * random.choice([-1, 1])
                        cx = max(crop_size // 2, min(img_w - crop_size // 2, bx + offset_x))
                        cy = max(crop_size // 2, min(img_h - crop_size // 2, by + offset_y))
                    else:
                        off_min, off_max = offset_range_extra
                        offset_x = random.randint(off_min, off_max) * random.choice([-1, 1])
                        offset_y = random.randint(off_min, off_max) * random.choice([-1, 1])
                        cx = max(crop_size // 2, min(img_w - crop_size // 2, bx + offset_x))
                        cy = max(crop_size // 2, min(img_h - crop_size // 2, by + offset_y))

                    # Quantize to reduce near-duplicate crops
                    cx_q = (cx // 50) * 50
                    cy_q = (cy // 50) * 50
                    if (cx_q, cy_q) in used_centers:
                        # Try a different offset
                        cx = max(crop_size // 2, min(img_w - crop_size // 2,
                                                      bx + random.randint(-300, 300)))
                        cy = max(crop_size // 2, min(img_h - crop_size // 2,
                                                      by + random.randint(-300, 300)))
                        cx_q = (cx // 50) * 50
                        cy_q = (cy // 50) * 50
                        if (cx_q, cy_q) in used_centers:
                            continue
                    used_centers.add((cx_q, cy_q))

                    # Crop the image
                    x1 = cx - crop_size // 2
                    y1 = cy - crop_size // 2
                    x2 = x1 + crop_size
                    y2 = y1 + crop_size

                    # Handle edge cases — shift crop window
                    if x1 < 0:
                        x2 -= x1
                        x1 = 0
                    if y1 < 0:
                        y2 -= y1
                        y1 = 0
                    if x2 > img_w:
                        x1 -= (x2 - img_w)
                        x2 = img_w
                    if y2 > img_h:
                        y1 -= (y2 - img_h)
                        y2 = img_h
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(img_w, x2)
                    y2 = min(img_h, y2)

                    # If crop is now too small (near edge), pad
                    actual_w, actual_h = x2 - x1, y2 - y1
                    if actual_w < crop_size * 0.5 or actual_h < crop_size * 0.5:
                        continue

                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    # Pad to target size if needed
                    if actual_w < crop_size or actual_h < crop_size:
                        pad_r = max(0, crop_size - actual_w)
                        pad_b = max(0, crop_size - actual_h)
                        crop = cv2.copyMakeBorder(crop, 0, pad_b, 0, pad_r,
                                                  cv2.BORDER_CONSTANT, value=(114, 114, 114))

                    # Find labels that overlap this crop
                    crop_lines: List[str] = []
                    for e_cls, e_xc, e_yc, e_w, e_h in entries:
                        overlap = _bbox_overlap(
                            (x1 + x2) / 2, (y1 + y2) / 2,  # crop center
                            crop_size, crop_size,
                            e_xc, e_yc, e_w, e_h, img_w, img_h,
                        )
                        if overlap >= MIN_BBOX_VISIBILITY:
                            nbox = _normalize_bbox_to_crop(
                                e_xc, e_yc, e_w, e_h, img_w, img_h,
                                (x1 + x2) / 2, (y1 + y2) / 2,
                                crop_size, crop_size,
                            )
                            if nbox:
                                crop_lines.append(f"{e_cls} {nbox[0]:.6f} {nbox[1]:.6f} "
                                                  f"{nbox[2]:.6f} {nbox[3]:.6f}")
                                if e_cls < NC:
                                    stats["per_class_boxes"][CLASS_NAMES[e_cls]] += 1

                    # Save
                    crop_name = f"{stem}_p{img_count:04d}"
                    cv2.imwrite(str(out_img_dir / f"{crop_name}.jpg"), crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    (out_lbl_dir / f"{crop_name}.txt").write_text(
                        "\n".join(crop_lines) + "\n" if crop_lines else "\n",
                        encoding="utf-8",
                    )
                    img_count += 1
                    box_count += len(crop_lines)
                    stats["positive_crops"][split_tag] += 1
                    stats["defect_boxes"][split_tag] += len(crop_lines)

            # ── Negative crops (defect-free but structurally similar) ─
            neg_count = 0
            max_attempts = negatives_per_image * 10
            for _ in range(max_attempts):
                if neg_count >= negatives_per_image:
                    break

                # Random center (avoid clustering around existing bboxes)
                cx = random.randint(crop_size // 2, img_w - crop_size // 2)
                cy = random.randint(crop_size // 2, img_h - crop_size // 2)

                # Check: does this crop contain any defect bbox?
                has_defect = False
                for _, e_xc, e_yc, e_w, e_h in entries:
                    overlap = _bbox_overlap(cx, cy, crop_size, crop_size,
                                            e_xc, e_yc, e_w, e_h, img_w, img_h)
                    if overlap > 0.05:  # even 5% overlap = not truly negative
                        has_defect = True
                        break
                if has_defect:
                    continue

                # Crop
                x1, y1 = cx - crop_size // 2, cy - crop_size // 2
                x2, y2 = x1 + crop_size, y1 + crop_size
                if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
                    continue

                crop = img[y1:y2, x1:x2]

                # Skip empty/uniform crops (likely sky, tunnel wall, etc.)
                std = crop.std()
                if std < 10:  # too uniform — nothing to learn
                    continue

                crop_name = f"{stem}_n{neg_count:04d}"
                cv2.imwrite(str(out_img_dir / f"{crop_name}.jpg"), crop,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                # Empty label file (no objects)
                (out_lbl_dir / f"{crop_name}.txt").write_text("\n", encoding="utf-8")
                neg_count += 1
                stats["negative_crops"][split_tag] += 1

            # Progress: print every image (flush for remote/container shells)
            print(f"  [{split_tag}] {idx+1}/{total_stems} {stem}: "
                  f"{len(entries)} defects, {len(used_centers)} pos + {neg_count} neg crops"
                  f"  |  cumulative: {img_count} pos, {box_count} boxes",
                  flush=True)

        print(f"  [{split_tag}] DONE: {img_count} positive + {neg_count} negative crops, "
              f"{box_count} boxes")

    # ── Write data.yaml ─────────────────────────────────────────────────
    data_yaml = output_dir / "subway_crops.yaml"
    if not dry_run:
        config = {
            "path": str(output_dir.resolve()),
            "train": "train/images",
            "val": "val/images",
            "nc": NC,
            "names": CLASS_NAMES,
            "# crop_size": crop_size,
            "# generated_by": "scripts/generate_native_crops.py",
        }
        if yaml is not None:
            data_yaml.write_text(
                yaml.dump(config, default_flow_style=False, allow_unicode=True),
                encoding="utf-8",
            )
        else:
            lines = [
                f"path: {output_dir.resolve()}",
                "train: train/images",
                "val: val/images",
                f"nc: {NC}",
                f"names: {CLASS_NAMES}",
            ]
            data_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nDataset config: {data_yaml}")

    return stats


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate native-resolution training crops from 5120px source images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_native_crops.py
  python scripts/generate_native_crops.py --crop-size 1280 --negatives-per-image 20
  python scripts/generate_native_crops.py --dry-run
  python scripts/generate_native_crops.py --src data/custom/images --labels data/custom/labels
""",
    )
    parser.add_argument(
        "--src", type=Path, default=DEFAULT_SRC_IMG,
        help=f"Source image directory (default: {DEFAULT_SRC_IMG})",
    )
    parser.add_argument(
        "--labels", type=Path, default=DEFAULT_SRC_LBL,
        help=f"Source label directory (default: {DEFAULT_SRC_LBL})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output root directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--crop-size", type=int, default=1024,
        help="Crop size in pixels (default: 1024; 1280 for Stage 2 main training)",
    )
    parser.add_argument(
        "--stride", type=int, default=512,
        help="Stride for negative crop sliding window (default: 512)",
    )
    parser.add_argument(
        "--negatives-per-image", type=int, default=25,
        help="Number of negative crops per source image (default: 25, was 15 — "
             "increased for better FP suppression on SVHBNM)",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.2,
        help="Validation split ratio by source image (default: 0.2)",
    )
    parser.add_argument(
        "--max-offset", type=int, default=300,
        help="Maximum random offset from bbox center in px (default: 300)",
    )
    parser.add_argument(
        "--balance", action="store_true",
        help="Class-balanced mode: minority classes (below median) get 2-3 crops per "
             "defect with varied offsets, instead of 1. Mitigates class imbalance.",
    )
    parser.add_argument(
        "--minority-multiplier", type=int, default=2,
        help="Extra crops per defect for minority classes in --balance mode (default: 2)",
    )
    parser.add_argument(
        "--debiasing", action="store_true",
        help="Position de-biasing: systematically vary defect position within the crop "
             "(center 30% / off-center 30% / near-edge 25% / corner 15%). "
             "Prevents the model from learning that defects always appear at crop center. "
             "Recommended for training to reduce position bias.",
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print statistics without generating files",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("  Native-Resolution Crop Generator")
    print("=" * 60)
    print(f"  Source images : {args.src}")
    print(f"  Source labels : {args.labels}")
    print(f"  Output dir    : {args.output}")
    print(f"  Crop size     : {args.crop_size}×{args.crop_size}")
    print(f"  Neg/img       : {args.negatives_per_image}")
    print(f"  Val ratio     : {args.val_ratio}")
    print(f"  Max offset    : ±{args.max_offset} px")
    print(f"  Dry run       : {args.dry_run}")
    print()

    stats = generate_crops(
        img_dir=args.src,
        label_dir=args.labels,
        output_dir=args.output,
        crop_size=args.crop_size,
        stride=args.stride,
        negatives_per_image=args.negatives_per_image,
        val_ratio=args.val_ratio,
        random_offset_range=(100, args.max_offset),
        dry_run=args.dry_run,
        balance=args.balance,
        minority_multiplier=args.minority_multiplier,
        debiasing=args.debiasing,
    )

    print(f"\n{'=' * 60}")
    print("  Generation Summary")
    print(f"{'=' * 60}")
    total_pos = stats["positive_crops"]["train"] + stats["positive_crops"]["val"]
    total_neg = stats["negative_crops"]["train"] + stats["negative_crops"]["val"]
    total_boxes = stats["defect_boxes"]["train"] + stats["defect_boxes"]["val"]
    print(f"  Train crops : {stats['positive_crops']['train']} pos + "
          f"{stats['negative_crops']['train']} neg = "
          f"{stats['positive_crops']['train'] + stats['negative_crops']['train']}")
    print(f"  Val crops   : {stats['positive_crops']['val']} pos + "
          f"{stats['negative_crops']['val']} neg = "
          f"{stats['positive_crops']['val'] + stats['negative_crops']['val']}")
    print(f"  Total boxes : {total_boxes}")
    print(f"  Per-class boxes:")
    for name in CLASS_NAMES:
        cnt = stats["per_class_boxes"].get(name, 0)
        bar = "█" * min(40, cnt // max(1, total_boxes // 40))
        print(f"    {name:<12s}: {cnt:>5d}  {bar}")

    if not args.dry_run:
        print(f"\n  Next: python -m subway_defect.train.train_defect \\")
        print(f"          --data {args.output / 'subway_crops.yaml'} \\")
        print(f"          --model subway_defect/models/yolo11s-P2-EMA-SimAM.yaml \\")
        print(f"          --coco_pretrain --device 0 --stages 1 2 3 --pretrain-config-dir")


if __name__ == "__main__":
    main()
