#!/usr/bin/env python3
"""Step 5: Generate synthetic missing-defect samples via inpainting (parallel).

Uses ``generate_missing_defect`` from ``subway_defect.synthetic.defect_synthesis``
to inpaint-remove the largest instance of an under-represented class, creating a
synthetic "missing component" training example.

Focuses on the five classes with the fewest training images (class 3 first).
Classes 0 and 2 (265 images each) are already well-represented and are skipped.

Performance: Pre-indexes images by class (one pass), then uses multiprocessing
for parallel inpainting across all CPU cores.

Usage:
    python scripts/generate_synthetic_defects.py
    python scripts/generate_synthetic_defects.py --target_class 3 --limit 20 --workers 8
"""

import argparse
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# Classes ordered by representation scarcity (fewest images first).
# Class 0 (VHBNM = 265) and class 2 (SVHBNM = 265) are skipped.
TARGET_CLASSES = [3, 6, 1, 5, 4]

# How many synthetic samples to generate per class
SYNTH_LIMITS: dict[int, int] = {3: 100, 6: 50, 1: 50, 5: 50, 4: 50}


# ---------------------------------------------------------------------------
# Standalone worker function (module-level → picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _inpaint_one(
    args: tuple,
) -> tuple | None:
    """Inpaint one image. Returns (out_img_path, out_lbl_path) or None."""
    image_path, label_path, output_img_dir, output_label_dir, target_class, suffix = args

    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    with open(label_path) as f:
        lines = f.readlines()

    # Collect boxes for the target class
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

    # Use the largest instance
    cx, cy, bw, bh = max(boxes, key=lambda b: b[2] * b[3])

    # Convert to pixel coords with small expansion
    x1 = max(0, int((cx - bw / 2) * w) - 3)
    y1 = max(0, int((cy - bh / 2) * h) - 3)
    x2 = min(w, int((cx + bw / 2) * w) + 3)
    y2 = min(h, int((cy + bh / 2) * h) + 3)

    # Inpaint the masked region
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    inpainted = cv2.inpaint(img, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # Save image
    stem = image_path.stem
    out_img = output_img_dir / f"{stem}{suffix}.jpg"
    cv2.imwrite(str(out_img), inpainted)

    # Write updated labels (remove the inpainted box)
    out_label = output_label_dir / f"{stem}{suffix}.txt"
    with open(out_label, "w") as f:
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            if cls_id == target_class:
                bx_cx = float(parts[1])
                bx_cy = float(parts[2])
                bx_bw = float(parts[3])
                bx_bh = float(parts[4])
                # Check IoU with the inpainted box
                inter_x = max(0.0, min(cx + bw / 2, bx_cx + bx_bw / 2)
                              - max(cx - bw / 2, bx_cx - bx_bw / 2))
                inter_y = max(0.0, min(cy + bh / 2, bx_cy + bx_bh / 2)
                              - max(cy - bh / 2, bx_cy - bx_bh / 2))
                if inter_x <= 0 or inter_y <= 0:
                    f.write(line)
            else:
                f.write(line)

    return (str(out_img), str(out_label))


def _is_original(stem: str) -> bool:
    """Return True if *stem* looks like an original (not already augmented)."""
    return "_aug" not in stem and "_synth" not in stem


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cpu_count = os.cpu_count() or 4

    parser = argparse.ArgumentParser(
        description="Generate synthetic missing-defect samples (parallel)",
    )
    parser.add_argument(
        "--train_images", default="data/Defect_dataset/images/train",
    )
    parser.add_argument(
        "--train_labels", default="data/Defect_dataset/labels/train",
    )
    parser.add_argument(
        "--target_classes", type=int, nargs="*", default=None,
        help="Classes to process (default: under-represented 3,6,1,5,4)",
    )
    parser.add_argument(
        "--limit_per_class", type=int, default=0,
        help="Max synthetic per class (0 = use built-in limits, -1 = unlimited)",
    )
    parser.add_argument(
        "--workers", type=int, default=cpu_count,
        help=f"Number of worker processes (default: {cpu_count})",
    )
    args = parser.parse_args()

    images_dir = Path(args.train_images)
    labels_dir = Path(args.train_labels)

    if not images_dir.is_dir():
        print(f"ERROR: {images_dir} not found. Run split_dataset.py first.")
        sys.exit(1)

    target_classes = args.target_classes or TARGET_CLASSES

    # ── Phase 1: Pre-index images by class (single pass) ──
    all_images = sorted(images_dir.glob("*.jpg"))
    originals = [p for p in all_images if _is_original(p.stem)]

    print(f"\n[Step 5] Pre-indexing {len(originals)} original images by class...")

    class_to_images: dict[int, list[tuple[Path, Path]]] = defaultdict(list)
    for img_path in tqdm(originals, desc="  Indexing", unit="img"):
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        classes_in_file: set[int] = set()
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    classes_in_file.add(int(parts[0]))
        for cid in classes_in_file:
            class_to_images[cid].append((img_path, lbl_path))

    for cid in sorted(class_to_images):
        print(f"  Class {cid}: {len(class_to_images[cid])} images")

    # ── Phase 2: Parallel inpainting ──
    total_generated = 0
    for cls_id in target_classes:
        candidates = class_to_images.get(cls_id, [])
        if not candidates:
            print(f"\n  Class {cls_id}: no images found, skipping")
            continue

        limit = args.limit_per_class or SYNTH_LIMITS.get(cls_id, 50)
        if limit == -1:
            limit = len(candidates)
        limit = min(limit, len(candidates))

        print(f"\n[Step 5] Class {cls_id}: generating up to {limit} synthetic "
              f"(from {len(candidates)} candidates, {args.workers} workers)")

        suffix = f"_synth_missing_{cls_id}"
        tasks = [
            (img_path, lbl_path, images_dir, labels_dir, cls_id, suffix)
            for img_path, lbl_path in candidates[:limit]
        ]

        generated = 0
        if args.workers > 1:
            # ── Parallel mode ──
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(_inpaint_one, task): task for task in tasks}
                with tqdm(total=len(tasks), desc=f"  Inpainting class {cls_id}", unit="img") as pbar:
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            if result is not None:
                                generated += 1
                        except Exception:
                            pass
                        pbar.update(1)
        else:
            # ── Sequential fallback ──
            for task in tqdm(tasks, desc=f"  Inpainting class {cls_id}", unit="img"):
                try:
                    result = _inpaint_one(task)
                    if result is not None:
                        generated += 1
                except Exception:
                    pass

        print(f"         Class {cls_id}: {generated} synthetic samples generated")
        total_generated += generated

    print(f"\n         Total synthetic samples: {total_generated}")


if __name__ == "__main__":
    main()
