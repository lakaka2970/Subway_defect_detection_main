#!/usr/bin/env python3
"""Merge the raw (un-augmented) source pool into ``data/train_data_3_raw``.

Builds the **raw** pre-augmentation dataset for the Stage 3+4 rerun. This
is Step 1 of the train_data_3 pipeline and merges exactly two raw sources:

* ``data/Defect_dataset_2/Defect_dataset`` (16-class, flat ``images/`` +
  ``labels/``) — the canonical raw parent of the (now-superseded)
  ``train_data_2``. All 16 classes present; classes 7-10 (RHTBNM/RHTBNL/
  GWCSBNM/GWCSBNL) are stripped here (they are excluded from this round).

* ``data/Defect_dataset`` (7-class, pre-split ``images/{train,val}``) —
  only the **originals** (no ``_aug*`` variants, which are pre-existing
  augmentations we must not re-augment), and only the two usable classes
  CBHPM (5) and CBVPM (6). Classes 0-4 ("垂直吊弦") carry labelling errors
  and are discarded; images with zero remaining boxes are dropped. The
  existing train/val split is **preserved** (per user instruction), so
  Defect_dataset contributes no ``test`` images.

Split policy:
* Defect_dataset_2 images are re-split 70/15/15 by source group
  (``IMG_YYYYMMDD_HHMMSS[_N]`` burst grouping), reusing the greedy
  deficit-balancing algorithm from ``expand_defect_dataset.py``.
* Defect_dataset images keep their existing train/val assignment.

Output layout (originals only — no augmentation):

    data/train_data_3_raw/
    ├── train/images  train/labels
    ├── val/images    val/labels
    ├── test/images   test/labels
    ├── classes.txt
    ├── data.yaml        (nc=16, all 16 names listed)
    └── manifest.json

Usage:
    python scripts/build_train_data_3_raw.py --dry-run
    python scripts/build_train_data_3_raw.py            # full run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}

# 16-class convention shared with train_data_2 (indices 0-15).
CLASS_NAMES = [
    "VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
    "RHTBNM", "RHTBNL", "GWCSBNM", "GWCSBNL", "GWCNM", "GWCNL",
    "BSBM", "INSD", "DRPS",
]
N_CLASSES = len(CLASS_NAMES)

# Classes excluded from this training round (zero-sample / not-yet-collected).
EXCLUDED_CLASS_IDS = {7, 8, 9, 10}

# Defect_dataset (source 2) usable classes: CBHPM, CBVPM only.
DEFECT_DATASET_KEEP_IDS = {5, 6}

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RARE_BOX_THRESHOLD = 200
MIN_HOLDOUT = 8

_IMG_RE = re.compile(r"^(IMG_\d{8}_\d{6})(?:_\d+)?$")
_AUG_RE = re.compile(r"_aug\d+_")


def source_id_of(stem: str) -> str:
    """Group burst frames: IMG_20260728_090621_1 → IMG_20260728_090621."""
    m = _IMG_RE.match(stem)
    return m.group(1) if m else stem


@dataclass
class ImageRecord:
    path: Path
    stem: str
    source_id: str
    boxes: List[List[float]] = field(default_factory=list)
    label_path: Optional[Path] = None
    has_label_file: bool = False
    majority_code: Optional[str] = None
    split: str = ""

    @property
    def is_negative(self) -> bool:
        return self.has_label_file and not self.boxes


def parse_label(text: str, n_classes: int) -> Tuple[List[List[float]], int]:
    """Parse YOLO label text → (valid boxes, dropped line count)."""
    boxes: List[List[float]] = []
    dropped = 0
    for line in text.splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            dropped += 1
            continue
        try:
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            dropped += 1
            continue
        if not (0 <= cls < n_classes) or any(not 0.0 <= v <= 1.0 for v in coords):
            dropped += 1
            continue
        if coords[2] <= 0 or coords[3] <= 0:
            dropped += 1
            continue
        boxes.append([cls] + coords)
    return boxes, dropped


def filter_boxes(boxes: List[List[float]], keep_ids: set) -> Tuple[List[List[float]], int]:
    """Keep only boxes whose class id ∈ keep_ids. Returns (kept, dropped)."""
    kept = [b for b in boxes if int(b[0]) in keep_ids]
    return kept, len(boxes) - len(kept)


def set_majority(records: List[ImageRecord]) -> None:
    """Assign ``majority_code`` from the most common remaining box class."""
    for rec in records:
        if rec.boxes:
            cnt = Counter(int(b[0]) for b in rec.boxes)
            rec.majority_code = CLASS_NAMES[cnt.most_common(1)[0][0]]


def scan_source_1(src: Path) -> Dict:
    """Scan Defect_dataset_2 (flat, 16-class), strip excluded classes."""
    img_dir, lbl_dir = src / "images", src / "labels"
    images = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES
    )
    label_files = {p.stem: p for p in lbl_dir.glob("*.txt") if p.name != "classes.txt"}

    records: List[ImageRecord] = []
    orphan_labels = sorted(set(label_files) - {p.stem for p in images})
    unlabelled = sorted({p.stem for p in images} - set(label_files))
    total_dropped = 0
    stripped_lines = 0
    for p in images:
        lf = label_files.get(p.stem)
        if lf is None:
            continue  # no label file → unlabelled, pending review → excluded
        rec = ImageRecord(path=p, stem=p.stem, source_id=source_id_of(p.stem))
        rec.has_label_file = True
        rec.label_path = lf
        boxes, dropped = parse_label(lf.read_text(encoding="utf-8"), N_CLASSES)
        total_dropped += dropped
        boxes, n_stripped = filter_boxes(boxes, set(range(N_CLASSES)) - EXCLUDED_CLASS_IDS)
        stripped_lines += n_stripped
        rec.boxes = boxes
        records.append(rec)
    return {
        "records": records,
        "orphan_labels": orphan_labels,
        "unlabelled_images": unlabelled,
        "dropped_lines": total_dropped,
        "stripped_excluded_lines": stripped_lines,
    }


def scan_source_2(root: Path) -> Dict:
    """Scan Defect_dataset originals (train/val), keep class 5/6 only.

    Only original (non-``_aug*``) images are used; their existing
    train/val assignment is preserved. Classes 0-4 are stripped; images
    with zero remaining boxes are dropped.
    """
    records: List[ImageRecord] = []
    dropped_vertical = 0
    dropped_images = 0
    for split in ("train", "val"):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        if not img_dir.is_dir():
            continue
        images = sorted(
            p for p in img_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES
        )
        for p in images:
            if _AUG_RE.search(p.stem):
                continue  # pre-existing augmentation variant → skip
            rec = ImageRecord(path=p, stem=p.stem, source_id=p.stem, split=split)
            lf = lbl_dir / (p.stem + ".txt")
            if lf.exists():
                rec.has_label_file = True
                rec.label_path = lf
                boxes, _ = parse_label(lf.read_text(encoding="utf-8"), 7)
                kept, dropped = filter_boxes(boxes, DEFECT_DATASET_KEEP_IDS)
                dropped_vertical += dropped
                rec.boxes = kept
            if not rec.boxes:
                dropped_images += 1
                continue  # no usable defect left → drop the image entirely
            records.append(rec)
    return {
        "records": records,
        "dropped_vertical_lines": dropped_vertical,
        "dropped_empty_images": dropped_images,
    }


def assign_splits(
    records: List[ImageRecord],
    class_sizes: Dict[str, int],
    seed: int,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Assign every record to train/val/test by source group (greedy deficit)."""
    groups: Dict[str, List[ImageRecord]] = defaultdict(list)
    for rec in records:
        groups[rec.source_id].append(rec)

    rng = np.random.default_rng(seed)
    order = sorted(groups.keys())
    rng.shuffle(order)
    order = sorted(order, key=lambda g: -len(groups[g]))

    rare_codes = {c for c, n in class_sizes.items() if 0 < n < RARE_BOX_THRESHOLD}

    counts = {"train": 0, "val": 0, "test": 0}
    per_class_split: Dict[str, Counter] = defaultdict(Counter)
    group_split: Dict[str, str] = {}
    split_names = list(SPLIT_RATIOS)
    total_assigned = 0
    for i, gid in enumerate(order):
        recs = groups[gid]
        rare_code = next(
            (r.majority_code for r in recs if r.majority_code in rare_codes), None
        )
        rotated = split_names[i % 3:] + split_names[: i % 3]
        best, best_score = rotated[0], -1e9
        for s in rotated:
            target = SPLIT_RATIOS[s]
            frac = counts[s] / total_assigned if total_assigned else 0.0
            score = target - frac
            if s == "train" and rare_code is not None:
                pc = per_class_split[rare_code]
                if pc["val"] >= MIN_HOLDOUT and pc["test"] >= MIN_HOLDOUT:
                    score += 0.10
            if score > best_score:
                best, best_score = s, score
        for rec in recs:
            rec.split = best
            if rec.majority_code:
                per_class_split[rec.majority_code][best] += 1
        counts[best] += len(recs)
        group_split[gid] = best
        total_assigned += len(recs)

    return counts, group_split


def _write_label(path: Path, boxes: List[List[float]]) -> None:
    lines = [
        f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def place_originals(records: List[ImageRecord], out_root: Path, copy_mode: str) -> Dict[str, int]:
    """Copy/hardlink originals into split dirs; write normalised labels."""
    stats = Counter()
    for rec in records:
        img_dir = out_root / rec.split / "images"
        lbl_dir = out_root / rec.split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        dst_img = img_dir / rec.path.name
        if not dst_img.exists():
            linked = False
            if copy_mode in ("auto", "link"):
                try:
                    os.link(rec.path, dst_img)
                    linked = True
                except OSError:
                    if copy_mode == "link":
                        raise
                if linked:
                    stats["linked"] += 1
            if not linked:
                shutil.copy2(rec.path, dst_img)
                stats["copied"] += 1
        if rec.has_label_file:
            _write_label(lbl_dir / (rec.stem + ".txt"), rec.boxes)
        stats["images"] += 1
    return dict(stats)


def write_data_yaml(out_root: Path) -> None:
    lines = [
        "# Train data for 16-class defect detection (RAW pool, pre-augmentation)",
        "# Auto-generated by scripts/build_train_data_3_raw.py",
        f"# Date: {datetime.now().isoformat(timespec='seconds')}",
        "",
        f"path: {out_root.as_posix()}",
        "train: train/images",
        "val: val/images",
        "test: test/images",
        "",
        f"nc: {N_CLASSES}",
        "names:",
    ] + [f"  - {n}" for n in CLASS_NAMES]
    (out_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge raw Defect_dataset_2 + Defect_dataset into train_data_3_raw"
    )
    parser.add_argument(
        "--source-1", type=Path,
        default=Path("data/Defect_dataset_2/Defect_dataset"),
    )
    parser.add_argument(
        "--source-2", type=Path, default=Path("data/Defect_dataset"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/train_data_3_raw"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-mode", choices=["auto", "link", "copy"], default="auto",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # ── scan source 1 (Defect_dataset_2) ───────────────────────────────
    print(f"[scan-1] {args.source_1}")
    scan1 = scan_source_1(args.source_1)
    records1: List[ImageRecord] = scan1["records"]
    set_majority(records1)
    print(f"  images={len(records1)}  orphan_labels={len(scan1['orphan_labels'])}  "
          f"unlabelled={len(scan1['unlabelled_images'])}  "
          f"dropped_lines={scan1['dropped_lines']}  "
          f"stripped_excluded_lines={scan1['stripped_excluded_lines']}")

    # ── scan source 2 (Defect_dataset originals) ───────────────────────
    print(f"[scan-2] {args.source_2}")
    scan2 = scan_source_2(args.source_2)
    records2: List[ImageRecord] = scan2["records"]
    set_majority(records2)
    print(f"  images={len(records2)}  dropped_vertical_lines={scan2['dropped_vertical_lines']}  "
          f"dropped_empty_images={scan2['dropped_empty_images']}")

    # ── class sizes (source 1 only drives rare-class split bias) ───────
    class_boxes = Counter()
    for rec in records1:
        class_boxes.update(int(b[0]) for b in rec.boxes)
    class_sizes = {CLASS_NAMES[i]: n for i, n in class_boxes.items()}

    # ── split: source 1 re-split, source 2 preserves existing split ────
    counts1, group_split = assign_splits(records1, class_sizes, args.seed)

    dataset_records = records1 + records2

    split_counts = Counter(r.split for r in dataset_records)
    per_class_split: Dict[str, Counter] = defaultdict(Counter)
    for rec in dataset_records:
        if rec.majority_code:
            per_class_split[rec.majority_code][rec.split] += 1

    print(f"[split] source-1 groups: train={counts1['train']} val={counts1['val']} "
          f"test={counts1['test']}")
    print(f"[split] merged images: train={split_counts['train']} "
          f"val={split_counts['val']} test={split_counts['test']}")
    print("[split] per-class images (train/val/test):")
    for i, code in enumerate(CLASS_NAMES):
        pc = per_class_split.get(code)
        if not pc:
            continue
        print(f"    {i:>2} {code:8} {pc['train']:>5}/{pc['val']:>4}/{pc['test']:>4}")

    if args.dry_run:
        total = len(dataset_records)
        total_bytes = sum(r.path.stat().st_size for r in dataset_records)
        print(f"[dry-run] {total} raw images, {total_bytes / 1e9:.2f} GB — nothing written.")
        return

    # ── write dataset ─────────────────────────────────────────────────
    out_root = args.output
    if any((out_root / s / "images").exists() and any((out_root / s / "images").iterdir())
           for s in ("train", "val", "test")):
        print(f"ERROR: {out_root} already contains data. Remove it or choose --output.")
        sys.exit(1)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[write] placing {len(dataset_records)} originals → {out_root}")
    stats = place_originals(dataset_records, out_root, args.copy_mode)
    print(f"[write] originals: {stats}")

    (out_root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    write_data_yaml(out_root)

    # ── self-check + manifest ─────────────────────────────────────────
    print("[check] verifying output integrity ...")
    problems = 0
    final_stats: Dict[str, Dict] = {}
    for s in ("train", "val", "test"):
        imgs = {p.stem for p in (out_root / s / "images").glob("*")
                if p.suffix.lower() in _IMG_SUFFIXES}
        lbls = {p.stem for p in (out_root / s / "labels").glob("*.txt")}
        missing_lbl = imgs - lbls
        orphan_lbl = lbls - imgs
        problems += len(missing_lbl) + len(orphan_lbl)
        n_boxes = 0
        per_cls = Counter()
        for lf in (out_root / s / "labels").glob("*.txt"):
            boxes, _ = parse_label(lf.read_text(encoding="utf-8"), N_CLASSES)
            n_boxes += len(boxes)
            per_cls.update(int(b[0]) for b in boxes)
        final_stats[s] = {
            "images": len(imgs),
            "labels": len(lbls),
            "missing_labels": sorted(missing_lbl)[:10],
            "orphan_labels": sorted(orphan_lbl)[:10],
            "boxes": n_boxes,
            "boxes_per_class": {CLASS_NAMES[i]: per_cls.get(i, 0)
                                for i in range(N_CLASSES) if per_cls.get(i, 0)},
        }
        print(f"    {s:5}: {len(imgs)} images, {n_boxes} boxes"
              + (f"  PROBLEMS={len(missing_lbl) + len(orphan_lbl)}"
                 if missing_lbl or orphan_lbl else ""))

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "sources": {
            "source_1": str(args.source_1),
            "source_2": str(args.source_2),
        },
        "output": str(out_root),
        "seed": args.seed,
        "classes": CLASS_NAMES,
        "excluded_class_ids": sorted(EXCLUDED_CLASS_IDS),
        "source_1": {"images": len(records1), "split": dict(counts1)},
        "source_2": {"images": len(records2),
                     "dropped_vertical_lines": scan2["dropped_vertical_lines"],
                     "dropped_empty_images": scan2["dropped_empty_images"]},
        "group_assignments": group_split,
        "splits": final_stats,
    }
    mf = out_root / "manifest.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] manifest → {mf}")
    if problems:
        print(f"WARNING: {problems} integrity problems — inspect manifest.")
        sys.exit(2)


if __name__ == "__main__":
    main()
