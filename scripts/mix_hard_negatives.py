#!/usr/bin/env python3
"""Mix hard negative crops into the training dataset for Stage 5.

Copies hard negative crops from data/hard_negatives/{class}/ into
data/subway_crops/train/images/ with empty label files (background).

Usage::

    # Mix hard negatives (1:1 ratio with positives)
    python scripts/mix_hard_negatives.py

    # Custom ratio (0.5 = half as many negatives as positives)
    python scripts/mix_hard_negatives.py --ratio 0.5

    # Dry-run
    python scripts/mix_hard_negatives.py --dry-run

    # Undo: remove previously mixed hard negatives
    python scripts/mix_hard_negatives.py --undo
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
HARD_NEG_DIR = _PROJECT_ROOT / "data" / "hard_negatives"
TRAIN_IMG_DIR = _PROJECT_ROOT / "data" / "subway_crops" / "train" / "images"
TRAIN_LBL_DIR = _PROJECT_ROOT / "data" / "subway_crops" / "train" / "labels"

# Prefix to identify hard negative files (for --undo)
HN_PREFIX = "hn_"

SEED = 42


def count_positives() -> int:
    """Count existing positive training images."""
    if not TRAIN_IMG_DIR.is_dir():
        return 0
    return len([f for f in TRAIN_IMG_DIR.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
                and not f.name.startswith(HN_PREFIX)])


def collect_hard_negatives() -> dict[str, list[Path]]:
    """Collect all hard negative crops organized by class."""
    result: dict[str, list[Path]] = {}
    if not HARD_NEG_DIR.is_dir():
        return result
    for cls_dir in sorted(HARD_NEG_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        crops = [f for f in cls_dir.iterdir()
                 if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        if crops:
            result[cls_dir.name] = crops
    return result


def mix(ratio: float, dry_run: bool = False) -> None:
    """Mix hard negatives into training data."""
    random.seed(SEED)

    n_pos = count_positives()
    hn_by_class = collect_hard_negatives()
    n_hn_total = sum(len(v) for v in hn_by_class.values())

    if n_hn_total == 0:
        print("ERROR: No hard negatives found in data/hard_negatives/")
        print("  Run first: python scripts/collect_hard_negatives.py --model weights/stage4_best_finetune.pt")
        sys.exit(1)

    target_hn = int(n_pos * ratio)
    print(f"  Positive training images: {n_pos}")
    print(f"  Available hard negatives: {n_hn_total}")
    print(f"  Target negatives ({ratio:.0%} of positives): {target_hn}")
    print()

    # Sample proportionally from each class
    selected: list[Path] = []
    for cls_name, crops in sorted(hn_by_class.items()):
        cls_target = max(1, int(target_hn * len(crops) / n_hn_total))
        cls_target = min(cls_target, len(crops))
        sampled = random.sample(crops, cls_target)
        selected.extend(sampled)
        print(f"    {cls_name:12s}: {len(crops):4d} available → {cls_target:4d} selected")

    random.shuffle(selected)
    print(f"\n  Total selected: {len(selected)}")

    if dry_run:
        print("  [DRY-RUN] No files copied")
        return

    # Copy to training directory with empty labels
    TRAIN_IMG_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_LBL_DIR.mkdir(parents=True, exist_ok=True)

    copied = 0
    for src in selected:
        dst_name = f"{HN_PREFIX}{src.parent.name}_{src.name}"
        dst_img = TRAIN_IMG_DIR / dst_name
        dst_lbl = TRAIN_LBL_DIR / (Path(dst_name).stem + ".txt")

        shutil.copy2(src, dst_img)
        dst_lbl.write_text("")  # empty label = background
        copied += 1

    print(f"  Copied {copied} hard negatives to {TRAIN_IMG_DIR}")
    print(f"  Empty labels created in {TRAIN_LBL_DIR}")


def undo() -> None:
    """Remove previously mixed hard negatives."""
    removed_img = 0
    removed_lbl = 0

    for f in TRAIN_IMG_DIR.glob(f"{HN_PREFIX}*"):
        f.unlink()
        removed_img += 1

    for f in TRAIN_LBL_DIR.glob(f"{HN_PREFIX}*"):
        f.unlink()
        removed_lbl += 1

    print(f"  Removed {removed_img} images and {removed_lbl} labels")


def main():
    parser = argparse.ArgumentParser(description="Mix hard negatives into training data")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Negative:positive ratio (default: 1.0 = 1:1)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--undo", action="store_true",
                        help="Remove previously mixed hard negatives")
    args = parser.parse_args()

    print("=" * 60)
    print("  Hard Negative Mixer")
    print("=" * 60)

    if args.undo:
        undo()
    else:
        mix(args.ratio, args.dry_run)


if __name__ == "__main__":
    main()
