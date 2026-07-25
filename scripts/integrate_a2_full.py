#!/usr/bin/env python3
"""
Integrate ALL actionable A2 audit results into the training dataset.

Extends merge_a2_review_data.py to also process:
  - 05_模型误报分类或定位错误: structure_fp / state=normal → hard negatives
  - 09_EMPTY_LABEL: sampled negatives (source-id dedup, max 200)

Already-merged data (02 HN, 03 FN, 04 Rev) is verified but not re-merged.

Usage::

    python scripts/integrate_a2_full.py
    python scripts/integrate_a2_full.py --dry-run
"""

from __future__ import annotations

import csv
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
A2_DIR = PROJECT_ROOT / "output" / "7.14训练结果"
FILTERED_IMAGES = A2_DIR / "筛选图片"
RESULTS_DIR = A2_DIR / "整理结果"
TRAIN_IMG = PROJECT_ROOT / "data" / "subway_crops" / "train" / "images"
TRAIN_LBL = PROJECT_ROOT / "data" / "subway_crops" / "train" / "labels"

MAX_EMPTY_NEGATIVES = 200  # Cap for category 09 samples


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def extract_source_id(crop_id: str) -> str:
    return re.sub(r'_(p|n|ms)\d{4}$', '', crop_id)


def find_image(review_id: str, event_type: str, crop_id: str) -> Path | None:
    idx = review_id.replace("A2-", "").lstrip("0") or "0"
    idx_str = idx.zfill(5)
    matches = list(FILTERED_IMAGES.glob(f"{idx_str}_{event_type}_*.jpg"))
    if matches:
        return matches[0]
    matches = list(FILTERED_IMAGES.glob(f"*_{crop_id}.jpg"))
    if matches:
        return matches[0]
    return None


def get_existing_source_ids() -> set[str]:
    source_ids = set()
    if TRAIN_IMG.is_dir():
        for img in TRAIN_IMG.glob("*.jpg"):
            source_ids.add(extract_source_id(img.stem))
    return source_ids


def verify_existing_merge() -> dict:
    """Verify that the initial A2 merge (02/03/04) data is present."""
    hn = len(list(TRAIN_IMG.glob("*_hn.jpg")))
    fn = len(list(TRAIN_IMG.glob("*_fn.jpg")))
    rev = len(list(TRAIN_IMG.glob("*_rev.jpg")))
    print(f"\n  ── 验证已有A2合并数据 ──")
    print(f"    Hard negatives (_hn): {hn}")
    print(f"    FN samples (_fn):     {fn}")
    print(f"    Revised (_rev):       {rev}")
    return {"hn": hn, "fn": fn, "rev": rev}


def merge_category05_negatives(dry_run: bool = False) -> dict:
    """Merge category 05 confirmed FPs as hard negatives.

    Includes:
    - structure_fp (83): structural false positives
    - class_confusion with state_label=normal: misclassified normals
    - texture_fp: texture false positives
    """
    rows = read_csv(RESULTS_DIR / "05_模型误报分类或定位错误.csv")
    print(f"\n  ── 05 模型误报 → 困难负样本: {len(rows)} 条 ──")

    # Filter to confirmed negatives (state_label=normal or error_type=structure_fp)
    negatives = [
        r for r in rows
        if r.get("state_label", "").strip() == "normal"
        or r.get("error_type", "").strip() in ("structure_fp", "texture_fp")
    ]
    print(f"    Confirmed negatives (state=normal or structure/texture FP): {len(negatives)}")

    existing_sources = get_existing_source_ids()
    added = 0
    skipped_leak = 0
    skipped_missing = 0

    for row in negatives:
        crop_id = row.get("crop_id", "")
        source_id = row.get("source_id", "")
        review_id = row.get("review_id", "")
        event_type = row.get("event_type", "FP")

        if source_id in existing_sources:
            skipped_leak += 1
            continue

        img_path = find_image(review_id, event_type, crop_id)
        if img_path is None or not img_path.exists():
            skipped_missing += 1
            continue

        if not dry_run:
            dest_name = f"{crop_id}_a2fp.jpg"
            shutil.copy2(img_path, TRAIN_IMG / dest_name)
            (TRAIN_LBL / f"{crop_id}_a2fp.txt").write_text("", encoding="utf-8")
            existing_sources.add(source_id)

        added += 1

    print(f"    Added: {added}, Skipped (leak): {skipped_leak}, Skipped (missing): {skipped_missing}")
    return {"added": added, "skipped_leak": skipped_leak}


def merge_category09_negatives(dry_run: bool = False, max_samples: int = MAX_EMPTY_NEGATIVES) -> dict:
    """Sample category 09 empty-label crops as negative training samples.

    These are crops with no annotations that were not reviewed in the A2 audit.
    We sample a subset (max 200) with source-id deduplication to add diversity
    to the negative pool.
    """
    rows = read_csv(RESULTS_DIR / "09_EMPTY_LABEL未纳入本轮审查.csv")
    print(f"\n  ── 09 EMPTY_LABEL → 采样负样本: {len(rows)} 条 (采样上限 {max_samples}) ──")

    if not rows:
        return {"added": 0}

    # Random sample
    rng = random.Random(SEED)
    if len(rows) > max_samples * 3:
        rows = rng.sample(rows, max_samples * 3)

    existing_sources = get_existing_source_ids()
    added = 0
    skipped_leak = 0
    skipped_missing = 0

    for row in rows:
        if added >= max_samples:
            break

        crop_id = row.get("crop_id", "")
        source_id = row.get("source_id", "")
        review_id = row.get("review_id", "")
        event_type = row.get("event_type", "TN")

        if source_id in existing_sources:
            skipped_leak += 1
            continue

        img_path = find_image(review_id, event_type, crop_id)
        if img_path is None or not img_path.exists():
            skipped_missing += 1
            continue

        if not dry_run:
            dest_name = f"{crop_id}_a2neg.jpg"
            shutil.copy2(img_path, TRAIN_IMG / dest_name)
            (TRAIN_LBL / f"{crop_id}_a2neg.txt").write_text("", encoding="utf-8")
            existing_sources.add(source_id)

        added += 1

    print(f"    Added: {added}, Skipped (leak): {skipped_leak}, Skipped (missing): {skipped_missing}")
    return {"added": added, "skipped_leak": skipped_leak}


def report_final_composition():
    """Report the final training set composition."""
    print(f"\n  {'='*50}")
    print(f"  训练集最终组成")
    print(f"  {'='*50}")

    if not TRAIN_IMG.is_dir():
        print("  ERROR: train/images not found")
        return

    files = list(TRAIN_IMG.glob("*.jpg"))
    categories = defaultdict(int)
    for f in files:
        stem = f.stem
        if "_aug" in stem:
            categories["场景增强 (_aug)"] += 1
        elif "_cp" in stem:
            categories["Copy-Paste (_cp)"] += 1
        elif "_hn" in stem:
            categories["A2困难负样本 (_hn)"] += 1
        elif "_fn" in stem:
            categories["A2漏检样本 (_fn)"] += 1
        elif "_rev" in stem:
            categories["A2标注修订 (_rev)"] += 1
        elif "_a2fp" in stem:
            categories["A2误报负样本 (_a2fp)"] += 1
        elif "_a2neg" in stem:
            categories["A2空标签负样本 (_a2neg)"] += 1
        else:
            categories["原始样本"] += 1

    total = sum(categories.values())
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"    {cat:30s}: {count:>6} ({pct:5.1f}%)")
    print(f"    {'总计':30s}: {total:>6}")

    # Positive vs negative
    pos = 0
    neg = 0
    for lbl in TRAIN_LBL.glob("*.txt"):
        if lbl.read_text(encoding="utf-8").strip():
            pos += 1
        else:
            neg += 1
    print(f"\n    正样本 (有标注): {pos} ({pos/(pos+neg)*100:.1f}%)")
    print(f"    负样本 (空标注): {neg} ({neg/(pos+neg)*100:.1f}%)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Full A2 integration into training set")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)

    print("=" * 60)
    print("  A2 审计成果完整整合")
    print("=" * 60)
    print(f"  Dry run: {args.dry_run}")

    # Step 1: Verify existing merge
    verify_existing_merge()

    # Step 2: Merge category 05 confirmed FPs
    stats05 = merge_category05_negatives(dry_run=args.dry_run)

    # Step 3: Sample category 09 empty labels
    stats09 = merge_category09_negatives(dry_run=args.dry_run)

    # Step 4: Report final composition
    if not args.dry_run:
        report_final_composition()
    else:
        total_new = stats05["added"] + stats09["added"]
        print(f"\n  [DRY-RUN] Would add {total_new} new images")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()