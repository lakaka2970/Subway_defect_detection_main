#!/usr/bin/env python3
"""Freeze source-level train/calibration/test splits for the 7.14 run.

The split unit is the original 5120 px source image (or a shared filename
prefix for the rare duplicated views).  Both labelled sources and unlabelled
normal sources are stratified, hashed and recorded before crop generation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "calibration", "test")
RATIOS = {"train": 0.70, "calibration": 0.15, "test": 0.15}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_id(stem: str) -> str:
    """Collapse duplicated view suffixes while preserving ordinary sources."""
    return re.sub(r"_\d+_\d+$", "", stem)


def read_boxes(path: Path) -> list[list[float]]:
    boxes: list[list[float]] = []
    if not path.exists():
        return boxes
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 columns")
        values = [float(v) for v in parts]
        cls = int(values[0])
        if cls < 0 or cls > 6 or any(v < 0 or v > 1 for v in values[1:]):
            raise ValueError(f"{path}:{line_no}: invalid YOLO annotation")
        boxes.append(values)
    return boxes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/Defect_dataset"))
    parser.add_argument("--output", type=Path, default=Path("data/eval_v20260714"))
    parser.add_argument("--seed", type=int, default=714)
    args = parser.parse_args()

    image_dir = args.dataset_root / "images"
    label_dir = args.dataset_root / "labels"
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not images:
        raise SystemExit(f"No images found in {image_dir}")

    records: dict[str, dict] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    for image in images:
        label = label_dir / f"{image.stem}.txt"
        boxes = read_boxes(label)
        records[image.stem] = {
            "image": image,
            "label": label if label.exists() else None,
            "boxes": boxes,
            "classes": Counter(int(b[0]) for b in boxes),
            "source_id": source_id(image.stem),
        }
        groups[source_id(image.stem)].append(image.stem)

    group_rows = []
    global_classes = Counter()
    for group_id, stems in groups.items():
        classes = Counter()
        for stem in stems:
            classes.update(records[stem]["classes"])
        global_classes.update(classes)
        group_rows.append({"id": group_id, "stems": stems, "classes": classes})

    rng = random.Random(args.seed)
    rng.shuffle(group_rows)
    # Rare labelled groups first; unlabelled groups are balanced by capacity.
    group_rows.sort(
        key=lambda g: (
            not bool(g["classes"]),
            min((global_classes[c] for c in g["classes"]), default=10**12),
            -sum(g["classes"].values()),
        )
    )

    target_groups = {s: len(group_rows) * RATIOS[s] for s in SPLITS}
    target_classes = {
        s: {c: global_classes[c] * RATIOS[s] for c in range(7)} for s in SPLITS
    }
    assigned_groups: dict[str, list[dict]] = {s: [] for s in SPLITS}
    assigned_classes: dict[str, Counter] = {s: Counter() for s in SPLITS}

    for group in group_rows:
        candidates = []
        for split in SPLITS:
            total_fill = (len(assigned_groups[split]) + 1) / max(target_groups[split], 1)
            class_cost = 0.0
            if group["classes"]:
                for cls, count in group["classes"].items():
                    after = assigned_classes[split][cls] + count
                    class_cost += after / max(target_classes[split][cls], 1)
                class_cost /= len(group["classes"])
            candidates.append((class_cost + total_fill * 0.35, total_fill, split))
        _, _, chosen = min(candidates)
        assigned_groups[chosen].append(group)
        assigned_classes[chosen].update(group["classes"])

    split_stems = {
        split: sorted(stem for group in assigned_groups[split] for stem in group["stems"])
        for split in SPLITS
    }
    if set.intersection(*(set(v) for v in split_stems.values())):
        raise RuntimeError("Source leakage across splits")
    if set().union(*(set(v) for v in split_stems.values())) != set(records):
        raise RuntimeError("Split assignment is incomplete")

    args.output.mkdir(parents=True, exist_ok=True)
    for split, stems in split_stems.items():
        (args.output / f"{split}_sources.txt").write_text("\n".join(stems) + "\n", encoding="utf-8")

    files = []
    split_summary = {}
    for split, stems in split_stems.items():
        counts = Counter()
        negatives = 0
        boxes = 0
        for stem in stems:
            row = records[stem]
            counts.update(row["classes"])
            boxes += len(row["boxes"])
            negatives += int(not row["boxes"])
            item = {
                "stem": stem,
                "source_id": row["source_id"],
                "split": split,
                "image": str(row["image"].relative_to(args.dataset_root)),
                "image_sha256": sha256(row["image"]),
                "label": str(row["label"].relative_to(args.dataset_root)) if row["label"] else None,
                "label_sha256": sha256(row["label"]) if row["label"] else None,
                "boxes": len(row["boxes"]),
                "class_boxes": dict(sorted(row["classes"].items())),
            }
            files.append(item)
        split_summary[split] = {
            "images": len(stems),
            "labelled_images": len(stems) - negatives,
            "unlabelled_negative_images": negatives,
            "boxes": boxes,
            "class_boxes": {str(c): counts[c] for c in range(7)},
            "source_groups": len(assigned_groups[split]),
        }

    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "dataset_root": str(args.dataset_root.resolve()),
        "split_ratios": RATIOS,
        "grouping": "filename with trailing _<view>_<camera> removed",
        "splits": split_stems,
        "summary": split_summary,
        "files": files,
        "leakage_source_groups": 0,
    }
    text = json.dumps(manifest, ensure_ascii=False, indent=2)
    (args.output / "manifest.json").write_text(text + "\n", encoding="utf-8")
    (args.output / "dataset_manifest.json").write_text(text + "\n", encoding="utf-8")
    print(json.dumps(split_summary, ensure_ascii=False, indent=2))
    print(f"Frozen manifest: {args.output / 'manifest.json'}")


if __name__ == "__main__":
    main()
