#!/usr/bin/env python3
"""Step 6: Validate the prepared dataset — integrity, format, statistics.

Checks:
  - Every image has a label, every label has an image.
  - Label format: 5 columns, class_id ∈ valid range (from classes registry).
  - No empty label files.
  - Class distribution per split.
  - Train/val source-group isolation (no leakage).

Performance: Uses ThreadPoolExecutor for parallel label file parsing.

Usage:
    python scripts/validate_dataset.py
    python scripts/validate_dataset.py --workers 8
"""

import argparse
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_CROP_SUFFIX_RE = re.compile(r"_[pn]\d+$")
_TILE_SUFFIX_RE = re.compile(r"_\d+_\d+$")

# Import class registry for validation bounds
try:
    from subway_defect.classes import TRAIN_NC_12 as TRAIN_NC
except ImportError:
    TRAIN_NC = 12  # fallback for standalone execution

MAX_CLASS_ID = TRAIN_NC - 1


def extract_source_prefix(stem: str) -> str:
    stem = _CROP_SUFFIX_RE.sub("", stem)
    m = _TILE_SUFFIX_RE.search(stem)
    return stem[: m.start()] if m else stem


def _split_dirs(root: Path, split: str) -> tuple[Path, Path]:
    """Support both images/train and train/images YOLO layouts."""
    standard_img = root / "images" / split
    standard_lbl = root / "labels" / split
    if standard_img.is_dir() or standard_lbl.is_dir():
        return standard_img, standard_lbl
    return root / split / "images", root / split / "labels"


def _parse_one_label(args: Tuple[Path, set]) -> Dict:
    """Parse a single label file. Returns stats dict."""
    lbl_path, img_stems = args
    result = {
        "stem": lbl_path.stem,
        "has_image": lbl_path.stem in img_stems,
        "empty": False,
        "classes": set(),
        "boxes_per_class": Counter(),
        "bad_format": 0,
        "errors": [],
    }

    try:
        with open(lbl_path, encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        result["errors"].append(f"Cannot read {lbl_path.name}: {e}")
        return result

    if not lines or all(not line.strip() for line in lines):
        result["empty"] = True
        return result

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            result["bad_format"] += 1
            continue
        try:
            cls_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            result["bad_format"] += 1
            continue

        if cls_id < 0 or cls_id > MAX_CLASS_ID:
            result["errors"].append(
                f"class {cls_id} out of range [0, {MAX_CLASS_ID}] in {lbl_path.name}")
        if any(not (0.0 <= c <= 1.0) for c in coords):
            result["errors"].append(
                f"coords out of [0,1] in {lbl_path.name}: {coords}")
        if coords[2] <= 0 or coords[3] <= 0:
            result["errors"].append(
                f"zero-area box in {lbl_path.name}: w={coords[2]}, h={coords[3]}")

        result["boxes_per_class"][cls_id] += 1
        result["classes"].add(cls_id)

    return result


def main() -> None:
    _MAX_WORKERS = min(os.cpu_count() or 4, 8)

    parser = argparse.ArgumentParser(description="Validate prepared dataset (parallel)")
    parser.add_argument("--dataset_root", default="data/Defect_dataset")
    parser.add_argument(
        "--workers", type=int, default=_MAX_WORKERS,
        help=f"Number of I/O threads (default: {_MAX_WORKERS}, max: {_MAX_WORKERS})",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument(
        "--allow-unlabelled", action="store_true",
        help="Treat images without labels as intentional negative samples.",
    )
    args = parser.parse_args()

    root = Path(args.dataset_root)
    errors: list[str] = []
    use_full_stem_for_leakage = root.name == "mixed_pretrain" or "public" in root.parts

    split_sources: Dict[str, set[str]] = {}
    for split in args.splits:
        img_dir, lbl_dir = _split_dirs(root, split)

        if not img_dir.is_dir():
            errors.append(f"[{split}] images/ directory missing: {img_dir}")
            continue
        if not lbl_dir.is_dir():
            errors.append(f"[{split}] labels/ directory missing: {lbl_dir}")
            continue

        imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        lbls = sorted(lbl_dir.glob("*.txt"))

        img_stems = {p.stem for p in imgs}

        # ---- Parallel label parsing ----
        class_img_counts = Counter()
        class_box_counts = Counter()
        empty_labels = 0
        bad_format = 0

        tasks = [(lbl_path, img_stems) for lbl_path in lbls]
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_parse_one_label, task): task[0]
                       for task in tasks}
            for future in as_completed(futures):
                r = future.result()
                lbl_path = futures[future]

                if not r["has_image"]:
                    errors.append(f"[{split}] label without image: {lbl_path.name}")

                if r["empty"]:
                    empty_labels += 1
                else:
                    for cid in r["classes"]:
                        class_img_counts[cid] += 1
                    for cid, count in r["boxes_per_class"].items():
                        class_box_counts[cid] += count

                bad_format += r["bad_format"]
                errors.extend(r["errors"])

        # Check for images without labels
        lbl_stems = {p.stem for p in lbls}
        orphan_imgs = img_stems - lbl_stems
        if orphan_imgs:
            message = (f"[{split}] {len(orphan_imgs)} intentional unlabelled negative image(s), "
                       f"e.g. {sorted(orphan_imgs)[:3]}")
            if args.allow_unlabelled:
                print(f"  [OK] {message}")
            else:
                errors.append(message)

        # Print per-split summary
        print(f"\n{'=' * 50}")
        print(f"  {split.upper()}  |  {len(imgs)} images  |  "
              f"{sum(class_box_counts.values())} boxes")
        print(f"{'=' * 50}")
        print(f"  {'Class':<8} {'Images':>8} {'Boxes':>8}")
        print(f"  {'-'*24}")
        for cid in sorted(set(list(class_img_counts) + list(class_box_counts))):
            print(f"  {cid:<8} {class_img_counts.get(cid,0):>8} "
                  f"{class_box_counts.get(cid,0):>8}")
        if empty_labels:
            print(f"  [WARN] Empty labels: {empty_labels}")
        if bad_format:
            print(f"  [WARN] Bad-format lines: {bad_format}")

    # ---- Cross-split leakage check ----
    print(f"\n{'=' * 50}")
    print("  Leakage check (train <-> val source overlap)")
    for split in args.splits:
        store: set[str] = set()
        img_dir, _ = _split_dirs(root, split)
        if img_dir.is_dir():
            for p in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
                store.add(p.stem if use_full_stem_for_leakage else extract_source_prefix(p.stem))
        split_sources[split] = store
    for i, left in enumerate(args.splits):
        for right in args.splits[i + 1:]:
            overlap = split_sources.get(left, set()) & split_sources.get(right, set())
            if overlap:
                errors.append(f"DATA LEAKAGE: {len(overlap)} source(s) appear in both "
                              f"{left} and {right}: {list(overlap)[:10]}")
            else:
                print(f"  [OK] No source overlap between {left} "
                      f"({len(split_sources.get(left, set()))}) and {right} "
                      f"({len(split_sources.get(right, set()))})")

    # ---- Final verdict ----
    print(f"\n{'=' * 50}")
    if errors:
        print(f"  [FAIL] VALIDATION FAILED - {len(errors)} issue(s):")
        for e in errors:
            print(f"     * {e}")
        sys.exit(1)
    else:
        print(f"  [PASS] VALIDATION PASSED - dataset is ready for training")


if __name__ == "__main__":
    main()
