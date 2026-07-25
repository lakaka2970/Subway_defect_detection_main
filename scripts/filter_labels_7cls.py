#!/usr/bin/env python3
"""
Filter training labels to keep only classes 0-6 (7-class training set).

Removes annotations with class_id >= 7 from all label files in
data/subway_crops/train/labels/ and data/subway_crops/val/labels/.

This is necessary because the crop generation script's cls_id < 7 filter
was not applied consistently, leaving 1,381 annotations for classes 7-11
(RHTBNM, RHTBNL, BSBM, INSD, DRPS) that the nc=7 model silently ignores.

Usage::

    python scripts/filter_labels_7cls.py
    python scripts/filter_labels_7cls.py --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NC = 7  # Keep only classes 0-6


def filter_labels(labels_dir: Path, dry_run: bool = False) -> dict:
    """Filter label files to keep only class IDs < NC."""
    if not labels_dir.is_dir():
        return {"files": 0, "removed": 0, "modified": 0}

    files_modified = 0
    annotations_removed = 0
    files_emptied = 0

    for lbl in sorted(labels_dir.glob("*.txt")):
        lines = lbl.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            continue

        kept = []
        removed = 0
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                if cls_id < NC:
                    kept.append(line.strip())
                else:
                    removed += 1
            else:
                kept.append(line.strip())

        if removed > 0:
            annotations_removed += removed
            files_modified += 1
            if not kept:
                files_emptied += 1
            if not dry_run:
                lbl.write_text(
                    "\n".join(kept) + "\n" if kept else "",
                    encoding="utf-8",
                )

    return {
        "files": len(list(labels_dir.glob("*.txt"))),
        "modified": files_modified,
        "removed": annotations_removed,
        "emptied": files_emptied,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Filter labels to 7 classes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Filter Labels to {NC} Classes (0-{NC-1})")
    print("=" * 60)
    print(f"  Dry run: {args.dry_run}")

    for split in ["train", "val"]:
        labels_dir = PROJECT_ROOT / "data" / "subway_crops" / split / "labels"
        print(f"\n  ── {split} ──")
        stats = filter_labels(labels_dir, dry_run=args.dry_run)
        print(f"    Total label files: {stats['files']}")
        print(f"    Files modified:    {stats['modified']}")
        print(f"    Annotations removed: {stats['removed']}")
        print(f"    Files emptied:     {stats['emptied']}")

    # Also delete any .cache files
    for split in ["train", "val"]:
        cache = PROJECT_ROOT / "data" / "subway_crops" / split / "labels.cache"
        if cache.exists() and not args.dry_run:
            cache.unlink()
            print(f"\n  Deleted {cache}")

    print(f"\n  Done.")


if __name__ == "__main__":
    main()