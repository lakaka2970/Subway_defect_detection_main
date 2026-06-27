#!/usr/bin/env python3
"""
Offline defect-aware Copy-Paste augmentation for small-object detection.

Extracts small-defect patches (< 32×32 px) from training images and pastes
them onto other images at non-overlapping locations. This multiplies the
number of small-object training instances — critical for classes like
SVHBNM (8-10 px bolt defects) where data scarcity causes both low Recall
and low Precision.

Usage::

    # Generate copy-paste augmented dataset from subway_crops
    python scripts/generate_defect_copy_paste.py \\
        --src data/subway_crops/train \\
        --output data/subway_crops_cp/train

    # Dry-run: print statistics without generating
    python scripts/generate_defect_copy_paste.py --dry-run

Output::

    data/subway_crops_cp/
    └── train/
        ├── images/    (original + copy-paste augmented images)
        └── labels/    (merged labels with pasted defects)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SEED = 42
DEFAULT_SRC = Path("data/subway_crops/train")
DEFAULT_OUTPUT = Path("data/subway_crops_cp/train")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Defect-aware Copy-Paste augmentation for small-object detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_defect_copy_paste.py
  python scripts/generate_defect_copy_paste.py --paste-prob 0.5 --max-pastes 5
  python scripts/generate_defect_copy_paste.py --dry-run
""",
    )
    parser.add_argument(
        "--src", type=Path, default=DEFAULT_SRC,
        help=f"Source training directory with images/ and labels/ (default: {DEFAULT_SRC})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output directory for augmented data (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--paste-prob", type=float, default=0.30,
        help="Probability a target image receives pastes (default: 0.30)",
    )
    parser.add_argument(
        "--max-pastes", type=int, default=3,
        help="Max pasted defects per target image (default: 3)",
    )
    parser.add_argument(
        "--min-bbox-size", type=int, default=8,
        help="Min defect side length in px (default: 8)",
    )
    parser.add_argument(
        "--max-bbox-size", type=int, default=32,
        help="Max defect side length in px — only paste small objects (default: 32)",
    )
    parser.add_argument(
        "--edge-margin", type=int, default=50,
        help="Margin from image edges in px to avoid mosaic cutting (default: 50)",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.10,
        help="Max allowed IoU between pasted and existing boxes (default: 0.10)",
    )
    parser.add_argument(
        "--alpha-blend", type=float, default=0.85,
        help="Blending alpha (1.0=hard edge, 0.0=invisible). Default: 0.85",
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

    # ── Import module ──────────────────────────────────────────────────
    try:
        from subway_defect.augmentations.defect_copy_paste import copy_paste_defects
    except (ModuleNotFoundError, ImportError):
        print("ERROR: Cannot import defect_copy_paste module.")
        print("Run: pip install -e .    or add the project root to PYTHONPATH.")
        sys.exit(1)

    # ── Resolve paths ──────────────────────────────────────────────────
    src_img_dir = args.src / "images"
    src_lbl_dir = args.src / "labels"
    out_img_dir = args.output / "images"
    out_lbl_dir = args.output / "labels"

    if not src_img_dir.is_dir():
        print(f"ERROR: Source image directory not found: {src_img_dir}")
        sys.exit(1)
    if not src_lbl_dir.is_dir():
        print(f"ERROR: Source label directory not found: {src_lbl_dir}")
        sys.exit(1)

    print("=" * 60)
    print("  Defect-Aware Copy-Paste Augmentation")
    print("=" * 60)
    print(f"  Source images  : {src_img_dir}")
    print(f"  Source labels  : {src_lbl_dir}")
    print(f"  Output images  : {out_img_dir}")
    print(f"  Output labels  : {out_lbl_dir}")
    print(f"  Paste prob     : {args.paste_prob}")
    print(f"  Max pastes/img : {args.max_pastes}")
    print(f"  Bbox size range: {args.min_bbox_size}-{args.max_bbox_size} px")
    print(f"  Edge margin    : {args.edge_margin} px")
    print(f"  IoU threshold  : {args.iou_threshold}")
    print(f"  Alpha blend    : {args.alpha_blend}")
    print(f"  Dry run        : {args.dry_run}")
    print()

    stats = copy_paste_defects(
        img_dir=src_img_dir,
        label_dir=src_lbl_dir,
        output_img_dir=out_img_dir,
        output_label_dir=out_lbl_dir,
        paste_prob=args.paste_prob,
        max_pastes=args.max_pastes,
        min_bbox_size=args.min_bbox_size,
        max_bbox_size=args.max_bbox_size,
        edge_margin=args.edge_margin,
        iou_threshold=args.iou_threshold,
        alpha_blend=args.alpha_blend,
        seed=args.seed,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print(f"\n  Augmented dataset ready at: {args.output}")
        print(f"  Next: update data YAML to point to this directory")


if __name__ == "__main__":
    main()
