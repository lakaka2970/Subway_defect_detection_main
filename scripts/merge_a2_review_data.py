#!/usr/bin/env python3
"""
Merge A2 human review results into the training dataset.

Reads the A2 review CSVs from output/7.14训练结果/整理结果/ and merges
actionable data into data/subway_crops/train/:

  - 02_严格困难负样本候选 (90): Add as negative samples (empty labels)
  - 03_有效模型漏检 (68): Add as positive samples with GT labels
  - 04_标注修订 (6): Add corrected labels (add_label / correct_bbox)
  - 01_可直接保留 (727): Freeze as evaluation baseline (separate dir)

Source-id deduplication prevents data leakage.

Usage::

    python scripts/merge_a2_review_data.py
    python scripts/merge_a2_review_data.py --dry-run
"""

from __future__ import annotations

import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
A2_DIR = PROJECT_ROOT / "output" / "7.14训练结果"
FILTERED_IMAGES = A2_DIR / "筛选图片"
RESULTS_DIR = A2_DIR / "整理结果"
TRAIN_IMG = PROJECT_ROOT / "data" / "subway_crops" / "train" / "images"
TRAIN_LBL = PROJECT_ROOT / "data" / "subway_crops" / "train" / "labels"
FROZEN_EVAL = PROJECT_ROOT / "data" / "subway_crops" / "frozen_eval"

# Class name → index mapping (7-class training set)
CLASS_MAP = {
    "VHBNM": 0, "VHBNL": 1, "SVHBNM": 2, "SVHBNL": 3,
    "SVHTNL": 4, "CBHPM": 5, "CBVPM": 6,
}


def read_csv(path: Path) -> List[Dict]:
    """Read a CSV file into a list of dicts."""
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def extract_source_id(crop_id: str) -> str:
    """Extract source image ID from crop_id (strip _p0000 / _n0000 suffix)."""
    import re
    return re.sub(r'_(p|n|ms)\d{4}$', '', crop_id)


def find_filtered_image(review_id: str, event_type: str, crop_id: str) -> Path | None:
    """Find the corresponding filtered image by review_id index."""
    # review_id format: A2-00005 → index 5
    idx = review_id.replace("A2-", "").lstrip("0") or "0"
    idx_str = idx.zfill(5)

    # Search for matching file pattern: {idx}_{TP/FP/FN/TN}_{source}_{crop}.jpg
    matches = list(FILTERED_IMAGES.glob(f"{idx_str}_{event_type}_*.jpg"))
    if matches:
        return matches[0]

    # Fallback: search by crop_id in filename
    matches = list(FILTERED_IMAGES.glob(f"*_{crop_id}.jpg"))
    if matches:
        return matches[0]

    return None


def get_existing_source_ids() -> Set[str]:
    """Get all source_ids already in the training set."""
    source_ids = set()
    if TRAIN_IMG.is_dir():
        for img in TRAIN_IMG.glob("*.jpg"):
            source_ids.add(extract_source_id(img.stem))
    return source_ids


def merge_hard_negatives(dry_run: bool = False) -> Dict:
    """Merge 02_严格困难负样本候选 as negative training samples."""
    rows = read_csv(RESULTS_DIR / "02_严格困难负样本候选_HN.csv")
    print(f"\n  ── 02 困难负样本 (HN): {len(rows)} 条 ──")

    existing_sources = get_existing_source_ids()
    added = 0
    skipped_leak = 0
    skipped_missing = 0

    for row in rows:
        crop_id = row.get("crop_id", "")
        source_id = row.get("source_id", "")
        review_id = row.get("review_id", "")
        event_type = row.get("event_type", "FP")

        # Source-id deduplication
        if source_id in existing_sources:
            skipped_leak += 1
            continue

        # Find the filtered image
        img_path = find_filtered_image(review_id, event_type, crop_id)
        if img_path is None or not img_path.exists():
            skipped_missing += 1
            continue

        if not dry_run:
            # Copy image with _hn suffix
            dest_name = f"{crop_id}_hn.jpg"
            shutil.copy2(img_path, TRAIN_IMG / dest_name)
            # Empty label file (negative sample)
            (TRAIN_LBL / f"{crop_id}_hn.txt").write_text("", encoding="utf-8")
            existing_sources.add(source_id)

        added += 1

    print(f"    Added: {added}, Skipped (leak): {skipped_leak}, Skipped (missing): {skipped_missing}")
    return {"added": added, "skipped_leak": skipped_leak, "skipped_missing": skipped_missing}


def merge_fn_samples(dry_run: bool = False) -> Dict:
    """Merge 03_有效模型漏检 as positive training samples with GT labels."""
    rows = read_csv(RESULTS_DIR / "03_有效模型漏检_FN.csv")
    print(f"\n  ── 03 有效漏检 (FN): {len(rows)} 条 ──")

    existing_sources = get_existing_source_ids()
    added = 0
    skipped_leak = 0
    skipped_missing = 0
    per_class = defaultdict(int)

    for row in rows:
        crop_id = row.get("crop_id", "")
        source_id = row.get("source_id", "")
        review_id = row.get("review_id", "")
        event_type = row.get("event_type", "FN")
        gt_class = row.get("gt_class", "")
        bbox_str = row.get("bbox", "")
        state_label = row.get("state_label", "")

        # Source-id deduplication
        if source_id in existing_sources:
            skipped_leak += 1
            continue

        # Find the filtered image
        img_path = find_filtered_image(review_id, event_type, crop_id)
        if img_path is None or not img_path.exists():
            skipped_missing += 1
            continue

        # Parse bbox [x1, y1, x2, y2] normalized
        if not gt_class or gt_class not in CLASS_MAP:
            skipped_missing += 1
            continue

        cls_idx = CLASS_MAP[gt_class]

        if not dry_run:
            dest_name = f"{crop_id}_fn.jpg"
            shutil.copy2(img_path, TRAIN_IMG / dest_name)

            # Write YOLO label from bbox
            # bbox format: [x1_norm, y1_norm, x2_norm, y2_norm]
            try:
                bbox = eval(bbox_str)  # [x1, y1, x2, y2]
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    # Clamp to [0, 1]
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    w = max(0.01, min(1, w))
                    h = max(0.01, min(1, h))
                    label_line = f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    (TRAIN_LBL / f"{crop_id}_fn.txt").write_text(
                        label_line + "\n", encoding="utf-8"
                    )
                else:
                    (TRAIN_LBL / f"{crop_id}_fn.txt").write_text("", encoding="utf-8")
            except Exception:
                (TRAIN_LBL / f"{crop_id}_fn.txt").write_text("", encoding="utf-8")

            existing_sources.add(source_id)

        added += 1
        per_class[gt_class] += 1

    print(f"    Added: {added}, Skipped (leak): {skipped_leak}, Skipped (missing): {skipped_missing}")
    if per_class:
        print(f"    Per-class: {dict(per_class)}")
    return {"added": added, "skipped_leak": skipped_leak, "per_class": dict(per_class)}


def merge_annotation_corrections(dry_run: bool = False) -> Dict:
    """Merge 04_标注修订 — add missing labels or correct existing ones."""
    rows = read_csv(RESULTS_DIR / "04_标注修订.csv")
    print(f"\n  ── 04 标注修订: {len(rows)} 条 ──")

    existing_sources = get_existing_source_ids()
    added = 0
    skipped = 0

    for row in rows:
        crop_id = row.get("crop_id", "")
        source_id = row.get("source_id", "")
        review_id = row.get("review_id", "")
        event_type = row.get("event_type", "FP")
        action = row.get("annotation_action", "")
        gt_class = row.get("gt_class", "") or row.get("predicted_class", "")
        bbox_str = row.get("bbox", "")

        if action not in ("add_label", "correct_bbox", "correct_class"):
            skipped += 1
            continue

        if source_id in existing_sources:
            skipped += 1
            continue

        img_path = find_filtered_image(review_id, event_type, crop_id)
        if img_path is None or not img_path.exists():
            skipped += 1
            continue

        cls_idx = CLASS_MAP.get(gt_class)
        if cls_idx is None:
            skipped += 1
            continue

        if not dry_run:
            dest_name = f"{crop_id}_rev.jpg"
            shutil.copy2(img_path, TRAIN_IMG / dest_name)

            try:
                bbox = eval(bbox_str)
                if len(bbox) == 4:
                    x1, y1, x2, y2 = bbox
                    cx = max(0, min(1, (x1 + x2) / 2))
                    cy = max(0, min(1, (y1 + y2) / 2))
                    w = max(0.01, min(1, x2 - x1))
                    h = max(0.01, min(1, y2 - y1))
                    label_line = f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                    (TRAIN_LBL / f"{crop_id}_rev.txt").write_text(
                        label_line + "\n", encoding="utf-8"
                    )
                else:
                    (TRAIN_LBL / f"{crop_id}_rev.txt").write_text("", encoding="utf-8")
            except Exception:
                (TRAIN_LBL / f"{crop_id}_rev.txt").write_text("", encoding="utf-8")

            existing_sources.add(source_id)

        added += 1

    print(f"    Added: {added}, Skipped: {skipped}")
    return {"added": added, "skipped": skipped}


def freeze_evaluation_baseline(dry_run: bool = False) -> Dict:
    """Freeze 01_可直接保留的正确样本 as evaluation baseline."""
    rows = read_csv(RESULTS_DIR / "01_可直接保留的正确样本.csv")
    print(f"\n  ── 01 正确样本 (冻结评估): {len(rows)} 条 ──")

    frozen_img = FROZEN_EVAL / "images"
    frozen_lbl = FROZEN_EVAL / "labels"

    added = 0
    skipped = 0

    for row in rows:
        crop_id = row.get("crop_id", "")
        review_id = row.get("review_id", "")
        event_type = row.get("event_type", "TP")

        img_path = find_filtered_image(review_id, event_type, crop_id)
        if img_path is None or not img_path.exists():
            skipped += 1
            continue

        if not dry_run:
            frozen_img.mkdir(parents=True, exist_ok=True)
            frozen_lbl.mkdir(parents=True, exist_ok=True)
            shutil.copy2(img_path, frozen_img / img_path.name)
            # Copy corresponding label if exists in train
            src_lbl = TRAIN_LBL / (crop_id + ".txt")
            if src_lbl.exists():
                shutil.copy2(src_lbl, frozen_lbl / (img_path.stem + ".txt"))
            else:
                (frozen_lbl / (img_path.stem + ".txt")).write_text("", encoding="utf-8")

        added += 1

    print(f"    Frozen: {added}, Skipped (missing): {skipped}")
    return {"frozen": added, "skipped": skipped}


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Merge A2 review data into training set")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only")
    args = parser.parse_args()

    random.seed(SEED)

    print("=" * 60)
    print("  Merge A2 Review Data into Training Set")
    print("=" * 60)
    print(f"  A2 results:    {RESULTS_DIR}")
    print(f"  Filtered imgs: {FILTERED_IMAGES}")
    print(f"  Train dir:     {TRAIN_IMG}")
    print(f"  Frozen eval:   {FROZEN_EVAL}")
    print(f"  Dry run:       {args.dry_run}")

    # Count current training set
    current_train = len(list(TRAIN_IMG.glob("*.jpg"))) if TRAIN_IMG.is_dir() else 0
    print(f"\n  Current train images: {current_train}")

    # Execute merges in order
    stats = {}
    stats["hn"] = merge_hard_negatives(dry_run=args.dry_run)
    stats["fn"] = merge_fn_samples(dry_run=args.dry_run)
    stats["rev"] = merge_annotation_corrections(dry_run=args.dry_run)
    stats["freeze"] = freeze_evaluation_baseline(dry_run=args.dry_run)

    # Final count
    if not args.dry_run:
        final_train = len(list(TRAIN_IMG.glob("*.jpg")))
        print(f"\n  {'='*50}")
        print(f"  Final train images: {final_train} (was {current_train}, +{final_train - current_train})")

        # Count positives vs negatives
        pos = 0
        neg = 0
        for lbl in TRAIN_LBL.glob("*.txt"):
            content = lbl.read_text(encoding="utf-8").strip()
            if content:
                pos += 1
            else:
                neg += 1
        total = pos + neg
        print(f"  Positives: {pos} ({pos/total*100:.1f}%)")
        print(f"  Negatives: {neg} ({neg/total*100:.1f}%)")
    else:
        total_added = stats["hn"]["added"] + stats["fn"]["added"] + stats["rev"]["added"]
        print(f"\n  [DRY-RUN] Would add {total_added} images to training set")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()
