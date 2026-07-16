#!/usr/bin/env python3
"""Step 3: Train/val split with source-image grouping to prevent data leakage.

Filenames like ``101813411_K44706_F1B04-542_1_22.jpg`` contain tile-coordinate
suffixes (``_1_21``, ``_1_22``). Tiles from the same source panorama share the
same prefix; we group by that prefix so all tiles from one source stay together
in either train or val.

Performance: Uses ThreadPoolExecutor for parallel file copy operations.

Usage:
    python scripts/split_dataset.py
    python scripts/split_dataset.py --ratio 0.8 --seed 42 --workers 8
"""

import argparse
import os
import random
import re
import shutil
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

SEED = 42
SPLIT_RATIO = 0.8

# Pattern: _<digits>_<digits> at end of stem, e.g. "_1_21" or "_2_15"
_TILE_SUFFIX_RE = re.compile(r"_\d+_\d+$")


def extract_source_prefix(stem: str) -> str:
    """Strip tile-coordinate suffix to recover the source-image key."""
    m = _TILE_SUFFIX_RE.search(stem)
    if m:
        return stem[: m.start()]
    return stem  # no recognised suffix — treat whole stem as source


def _copy_one(copy_args: Tuple[Path, Path]) -> None:
    """Copy a single file (image or label)."""
    src, dst = copy_args
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    _MAX_WORKERS = min(os.cpu_count() or 4, 8)

    parser = argparse.ArgumentParser(description="Train/val split with source grouping")
    parser.add_argument("--images_dir", default="data/Defect_dataset/images")
    parser.add_argument("--labels_dir", default="data/Defect_dataset/labels")
    parser.add_argument("--output_dir", default="data/Defect_dataset")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--ratio", type=float, default=SPLIT_RATIO)
    parser.add_argument(
        "--workers", type=int, default=_MAX_WORKERS,
        help=f"Number of I/O threads (default: {_MAX_WORKERS}, max: {_MAX_WORKERS})",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    output_dir = Path(args.output_dir)

    # ---- Group images by source prefix ----
    source_groups: dict[str, list[str]] = defaultdict(list)
    for img_path in sorted(images_dir.glob("*.jpg")):
        prefix = extract_source_prefix(img_path.stem)
        source_groups[prefix].append(img_path.stem)

    sources = list(source_groups.keys())
    random.shuffle(sources)

    split_idx = int(len(sources) * args.ratio)
    train_sources = set(sources[:split_idx])
    val_sources = set(sources[split_idx:])

    train_tiles = sum(len(source_groups[s]) for s in train_sources)
    val_tiles = sum(len(source_groups[s]) for s in val_sources)

    print(f"[Step 3] Total source images : {len(sources)}")
    print(f"         Train sources     : {len(train_sources)} → {train_tiles} tiles")
    print(f"         Val sources       : {len(val_sources)} → {val_tiles} tiles")
    print(f"         I/O workers       : {args.workers}")

    # Highlight multi-tile sources
    multi = {s: stems for s, stems in source_groups.items() if len(stems) > 1}
    if multi:
        print(f"         Multi-tile sources: {len(multi)} (kept intact within split)")
        for s, stems in list(multi.items())[:5]:
            print(f"           {s}: {stems}")

    # ---- Create output directories ----
    for split_name in ("train", "val"):
        (output_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

    # ---- Build copy task list ----
    copy_tasks: List[Tuple[Path, Path]] = []
    for split_name, source_set in (("train", train_sources), ("val", val_sources)):
        for source in source_set:
            for stem in source_groups[source]:
                src_img = images_dir / f"{stem}.jpg"
                dst_img = output_dir / "images" / split_name / f"{stem}.jpg"
                copy_tasks.append((src_img, dst_img))

                src_lbl = labels_dir / f"{stem}.txt"
                dst_lbl = output_dir / "labels" / split_name / f"{stem}.txt"
                if src_lbl.exists():
                    copy_tasks.append((src_lbl, dst_lbl))

    # ---- Parallel copy ----
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(_copy_one, copy_tasks))

    print(f"         Split complete ({len(copy_tasks)} files copied).")


if __name__ == "__main__":
    main()
