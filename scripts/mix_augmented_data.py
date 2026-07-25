#!/usr/bin/env python3
"""
Mix augmented data into the training set.

Merges scene-augmented images (*_aug*.jpg) and copy-paste augmented images
from subway_crops_cp/ back into the main subway_crops training set.
Performs source-level deduplication to prevent same-source adjacent frames
from being over-represented.

Usage::

    # Mix scene augmentations into subway_crops
    python scripts/mix_augmented_data.py --mode scene

    # Mix copy-paste augmented data
    python scripts/mix_augmented_data.py --mode copy-paste

    # Mix both
    python scripts/mix_augmented_data.py --mode all

    # Dry-run: print statistics
    python scripts/mix_augmented_data.py --mode all --dry-run
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

SEED = 42
DEFAULT_CROPS = Path("data/subway_crops")
DEFAULT_CP = Path("data/subway_crops_cp")


def extract_source_id(stem: str) -> str:
    """Extract source image ID from crop filename.

    Crop names follow pattern: {source_stem}_p{idx} or {source_stem}_n{idx}
    or {source_stem}_ms{idx} or {source_stem}_aug{idx}_{augname}
    or {source_stem}_cp{idx}
    """
    # Remove crop suffix patterns
    stem = re.sub(r'_(p|n|ms|cp)\d{4}$', '', stem)
    stem = re.sub(r'_aug\d+_\w+$', '', stem)
    # Remove tile coordinate suffix (e.g., _1_22)
    stem = re.sub(r'_\d+_\d+$', '', stem)
    return stem


def count_per_source(images_dir: Path) -> Dict[str, int]:
    """Count images per source ID."""
    source_counts: Dict[str, int] = defaultdict(int)
    for img in images_dir.glob("*.jpg"):
        source_id = extract_source_id(img.stem)
        source_counts[source_id] += 1
    return dict(source_counts)


def mix_scene_augmentations(
    crops_dir: Path,
    dry_run: bool = False,
    max_per_source: int = 5,
) -> Dict[str, int]:
    """Mix scene-augmented images into the training set.

    Scene augmentations are already in the train/images/ directory with
    *_aug* suffix. This function validates and reports on them.

    Args:
        crops_dir: Path to subway_crops root.
        dry_run: Print stats without modifying.
        max_per_source: Max augmented images per source (dedup).

    Returns:
        Statistics dict.
    """
    train_img = crops_dir / "train" / "images"
    train_lbl = crops_dir / "train" / "labels"

    if not train_img.is_dir():
        print(f"  ERROR: {train_img} not found")
        return {}

    # Find all augmented images
    aug_images = sorted(train_img.glob("*_aug*.jpg"))
    print(f"  Found {len(aug_images)} scene-augmented images in train/")

    # Count per source
    source_counts: Dict[str, int] = defaultdict(int)
    for img in aug_images:
        source_id = extract_source_id(img.stem)
        source_counts[source_id] += 1

    # Identify over-represented sources
    over_limit = {s: c for s, c in source_counts.items() if c > max_per_source}
    to_remove = []
    for source_id, count in over_limit.items():
        # Keep only max_per_source, remove the rest
        source_augs = [img for img in aug_images if extract_source_id(img.stem) == source_id]
        random.shuffle(source_augs)
        to_remove.extend(source_augs[max_per_source:])

    stats = {
        "total_aug": len(aug_images),
        "sources": len(source_counts),
        "over_limit_sources": len(over_limit),
        "removed": len(to_remove),
        "kept": len(aug_images) - len(to_remove),
    }

    if not dry_run and to_remove:
        for img in to_remove:
            img.unlink(missing_ok=True)
            lbl = train_lbl / (img.stem + ".txt")
            lbl.unlink(missing_ok=True)
        print(f"  Removed {len(to_remove)} over-represented augmented images")

    return stats


def mix_copy_paste(
    crops_dir: Path,
    cp_dir: Path,
    dry_run: bool = False,
) -> Dict[str, int]:
    """Mix copy-paste augmented images into the training set.

    Copies *_cp* images from subway_crops_cp/train/ into subway_crops/train/.

    Args:
        crops_dir: Path to subway_crops root.
        cp_dir: Path to subway_crops_cp root.
        dry_run: Print stats without modifying.

    Returns:
        Statistics dict.
    """
    cp_train_img = cp_dir / "train" / "images"
    cp_train_lbl = cp_dir / "train" / "labels"
    train_img = crops_dir / "train" / "images"
    train_lbl = crops_dir / "train" / "labels"

    if not cp_train_img.is_dir():
        print(f"  ERROR: {cp_train_img} not found")
        return {}

    # Find copy-paste augmented images (only _cp suffixed, not originals)
    cp_images = sorted(cp_train_img.glob("*_cp*.jpg"))
    print(f"  Found {len(cp_images)} copy-paste images in {cp_dir}")

    # Check which ones are already in the target
    existing = {f.name for f in train_img.glob("*.jpg")}
    new_images = [img for img in cp_images if img.name not in existing]

    stats = {
        "total_cp": len(cp_images),
        "already_present": len(cp_images) - len(new_images),
        "new_to_add": len(new_images),
    }

    if not dry_run and new_images:
        for img in new_images:
            shutil.copy2(img, train_img / img.name)
            lbl = cp_train_lbl / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy2(lbl, train_lbl / lbl.name)
        print(f"  Copied {len(new_images)} new copy-paste images to {train_img}")

    return stats


def report_dataset_stats(crops_dir: Path) -> None:
    """Print final dataset statistics after mixing."""
    train_img = crops_dir / "train" / "images"
    train_lbl = crops_dir / "train" / "labels"

    if not train_img.is_dir():
        return

    all_images = list(train_img.glob("*.jpg"))
    total = len(all_images)

    # Count positives (have non-empty label) vs negatives
    positives = 0
    negatives = 0
    per_class = defaultdict(int)

    for img in all_images:
        lbl = train_lbl / (img.stem + ".txt")
        if lbl.exists():
            content = lbl.read_text(encoding="utf-8").strip()
            if content:
                positives += 1
                for line in content.splitlines():
                    parts = line.strip().split()
                    if parts:
                        per_class[int(parts[0])] += 1
            else:
                negatives += 1
        else:
            negatives += 1

    print(f"\n  Final Dataset Stats:")
    print(f"    Total images:  {total}")
    print(f"    Positives:     {positives} ({positives/total*100:.1f}%)")
    print(f"    Negatives:     {negatives} ({negatives/total*100:.1f}%)")
    print(f"    Neg ratio:     {negatives/total*100:.1f}%")

    class_names = ["VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM"]
    total_boxes = sum(per_class.values())
    print(f"    Total boxes:   {total_boxes}")
    if total_boxes > 0:
        counts = [per_class.get(i, 0) for i in range(7)]
        imbalance = max(counts) / max(min(counts), 1)
        print(f"    Imbalance:     {imbalance:.1f}x")
        for i, name in enumerate(class_names):
            cnt = per_class.get(i, 0)
            print(f"      {name:8s}: {cnt:>5d} ({cnt/total_boxes*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Mix augmented data into training set"
    )
    parser.add_argument(
        "--mode", choices=["scene", "copy-paste", "all"], default="all",
        help="Which augmentations to mix (default: all)",
    )
    parser.add_argument(
        "--crops-dir", type=Path, default=DEFAULT_CROPS,
        help=f"Path to subway_crops root (default: {DEFAULT_CROPS})",
    )
    parser.add_argument(
        "--cp-dir", type=Path, default=DEFAULT_CP,
        help=f"Path to subway_crops_cp root (default: {DEFAULT_CP})",
    )
    parser.add_argument(
        "--max-aug-per-source", type=int, default=5,
        help="Max scene-augmented images per source (default: 5)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print stats only")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("  Mix Augmented Data")
    print("=" * 60)
    print(f"  Mode:       {args.mode}")
    print(f"  Crops dir:  {args.crops_dir}")
    print(f"  CP dir:     {args.cp_dir}")
    print(f"  Dry run:    {args.dry_run}")
    print()

    if args.mode in ("scene", "all"):
        print("  ── Scene Augmentations ──")
        stats = mix_scene_augmentations(
            args.crops_dir, dry_run=args.dry_run,
            max_per_source=args.max_aug_per_source,
        )
        for k, v in stats.items():
            print(f"    {k}: {v}")
        print()

    if args.mode in ("copy-paste", "all"):
        print("  ── Copy-Paste Augmentations ──")
        stats = mix_copy_paste(
            args.crops_dir, args.cp_dir, dry_run=args.dry_run,
        )
        for k, v in stats.items():
            print(f"    {k}: {v}")
        print()

    if not args.dry_run:
        report_dataset_stats(args.crops_dir)


if __name__ == "__main__":
    main()
