#!/usr/bin/env python3
"""
Synthetic defect generation via image inpainting.

Generates "missing component" training samples by removing annotated
components from normal images using OpenCV inpainting.

Usage:
    python synthetic/defect_synthesis.py \
        --images datasets/images/train/ \
        --labels datasets/labels/train/ \
        --output datasets/synthetic/ \
        --target_class 0
"""

import argparse
from pathlib import Path

import cv2
import numpy as np


def generate_missing_defect(
    image_path: Path,
    label_path: Path,
    output_img_dir: Path,
    output_label_dir: Path,
    target_class: int,
    suffix: str = "_synth_missing",
):
    """Generate a missing-component sample by inpainting one instance.

    Reads a YOLO-format label, selects the largest bounding box for
    the target class, paints a mask, inpaints the region, and writes
    the altered image plus updated label.

    Args:
        image_path: Source image path.
        label_path: YOLO-format .txt label path.
        output_img_dir: Output directory for synthetic images.
        output_label_dir: Output directory for synthetic labels.
        target_class: Class index to "remove" via inpainting.
        suffix: Filename suffix for the synthetic sample.

    Returns:
        Path to generated image, or None if no suitable instance found.
    """
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

    # Ensure output directories exist
    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

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

    return out_img


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic missing-component defects")
    parser.add_argument("--images", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target_class", type=int, required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    out_img_dir = Path(args.output) / "images"
    out_lbl_dir = Path(args.output) / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    image_files = (sorted(images_dir.glob("*.jpg"))
                   + sorted(images_dir.glob("*.png")))
    generated = 0
    for img_path in image_files:
        if args.limit and generated >= args.limit:
            break
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        result = generate_missing_defect(
            img_path, lbl_path, out_img_dir, out_lbl_dir,
            args.target_class)
        if result:
            generated += 1
            print(f"[{generated}] {result.name}")

    print(f"Done. {generated} synthetic samples written to {args.output}")


if __name__ == "__main__":
    main()
