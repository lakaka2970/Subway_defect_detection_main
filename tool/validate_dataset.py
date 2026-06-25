#!/usr/bin/env python3
"""Step 6: Validate the prepared dataset — integrity, format, statistics.

Checks:
  - Every image has a label, every label has an image.
  - Label format: 5 columns, class_id ∈ [0,6], coords ∈ [0,1].
  - No empty label files.
  - Class distribution per split.
  - Train/val source-group isolation (no leakage).

Usage:
    python tool/validate_dataset.py
"""

import argparse
import re
from collections import Counter
from pathlib import Path

_TILE_SUFFIX_RE = re.compile(r"_\d+_\d+$")


def extract_source_prefix(stem: str) -> str:
    m = _TILE_SUFFIX_RE.search(stem)
    return stem[: m.start()] if m else stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate prepared dataset")
    parser.add_argument("--dataset_root", default="data/Defect_dataset")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    errors: list[str] = []
    all_ok = True

    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split

        if not img_dir.is_dir():
            errors.append(f"[{split}] images/ directory missing: {img_dir}")
            continue
        if not lbl_dir.is_dir():
            errors.append(f"[{split}] labels/ directory missing: {lbl_dir}")
            continue

        imgs = sorted(img_dir.glob("*.jpg"))
        lbls = sorted(lbl_dir.glob("*.txt"))

        img_stems = {p.stem for p in imgs}
        lbl_stems = {p.stem for p in lbls}

        orphan_imgs = img_stems - lbl_stems
        orphan_lbls = lbl_stems - img_stems

        if orphan_imgs:
            errors.append(f"[{split}] {len(orphan_imgs)} image(s) without label, "
                          f"e.g. {list(orphan_imgs)[:3]}")
        if orphan_lbls:
            errors.append(f"[{split}] {len(orphan_lbls)} label(s) without image, "
                          f"e.g. {list(orphan_lbls)[:3]}")

        class_img_counts = Counter()
        class_box_counts = Counter()
        empty_labels = 0
        bad_format = 0

        for lbl_path in lbls:
            with open(lbl_path, encoding="utf-8") as f:
                lines = f.readlines()

            if not lines:
                empty_labels += 1
                continue

            classes_in_file: set[int] = set()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    bad_format += 1
                    if bad_format <= 3:
                        errors.append(f"[{split}] bad format in {lbl_path.name}: {line}")
                    continue
                try:
                    cls_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]
                except ValueError:
                    bad_format += 1
                    errors.append(f"[{split}] non-numeric in {lbl_path.name}: {line}")
                    continue

                if cls_id < 0 or cls_id > 6:
                    errors.append(f"[{split}] class {cls_id} out of range in {lbl_path.name}")
                if any(not (0.0 <= c <= 1.0) for c in coords):
                    errors.append(f"[{split}] coords out of [0,1] in {lbl_path.name}: {coords}")
                if coords[2] <= 0 or coords[3] <= 0:
                    errors.append(f"[{split}] zero-area box in {lbl_path.name}: w={coords[2]}, h={coords[3]}")

                class_box_counts[cls_id] += 1
                classes_in_file.add(cls_id)

            for cid in classes_in_file:
                class_img_counts[cid] += 1

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
    train_sources: set[str] = set()
    val_sources: set[str] = set()

    for split, store in (("train", train_sources), ("val", val_sources)):
        img_dir = root / "images" / split
        if img_dir.is_dir():
            for p in img_dir.glob("*.jpg"):
                store.add(extract_source_prefix(p.stem))

    overlap = train_sources & val_sources
    if overlap:
        errors.append(f"DATA LEAKAGE: {len(overlap)} source(s) appear in both "
                      f"train and val: {list(overlap)[:10]}")
    else:
        print(f"  [OK] No source overlap between train ({len(train_sources)}) "
              f"and val ({len(val_sources)})")

    # ---- Final verdict ----
    print(f"\n{'=' * 50}")
    if errors:
        print(f"  [FAIL] VALIDATION FAILED - {len(errors)} issue(s):")
        for e in errors:
            print(f"     * {e}")
    else:
        print(f"  [PASS] VALIDATION PASSED - dataset is ready for training")


if __name__ == "__main__":
    main()
