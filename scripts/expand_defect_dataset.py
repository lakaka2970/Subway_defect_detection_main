#!/usr/bin/env python3
"""Expand Defect_dataset_2 into a train-ready, class-balanced dataset.

Reads ``data/Defect_dataset_2/Defect_dataset`` (flat ``images/`` +
``labels/``, 16-class YOLO), splits it by source group into
train/val/test, and generates class-aware offline augmentation variants
(**train split only**) into ``data/train_data_2``:

    data/train_data_2/
    ├── train/images  train/labels   originals + augmented variants
    ├── val/images    val/labels     originals only (calibration)
    ├── test/images   test/labels    originals only (frozen hold-out)
    ├── classes.txt
    └── manifest.json

Design rules (per docs/plans/20260-08-09 plan):
* Source-group isolation — all frames of one capture second
  (``IMG_YYYYMMDD_HHMMSS[_N]``) go to the same split; the three splits
  have zero group overlap.
* Augmented variants never enter val/test (evaluation must stay free of
  near-duplicates), and never count toward independent-sample gates.
* Variant budget per source image is class-aware: low-sample and
  low-recall classes (VHBNM, CBHPM, INSD, SVHBNM) get more variants;
  false-positive-driven classes (DRPS, GWCNL) get fewer — their remedy
  is hard negatives, not more positives.
* The four zero-sample classes (RHTBNM/RHTBNL/GWCSBNM/GWCSBNL) have no
  images at all (orphan labels only) — nothing can be augmented; they
  need real collection.
* Empty-label images are kept as background/negative samples; images
  without any label file and orphan labels are excluded pending review.

Augmentation pool (bounding boxes unchanged by every transform):
    vibration / glare / night / tunnel / degrade (downscale→upscale,
    reduces effective defect pixel count) / bgblur (depth of field) /
    defocus / jpeg / bgswap (GrabCut + blend onto Normal_dataset pool) /
    whitebal / weather / sunlit

Usage:
    # Plan only — per-class variant counts and storage estimate
    python scripts/expand_defect_dataset.py --dry-run

    # Small validation run (first 24 source groups)
    python scripts/expand_defect_dataset.py --limit-groups 24

    # Full run
    python scripts/expand_defect_dataset.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from subway_defect.augmentations.background_replacement import BackgroundReplacer
from subway_defect.augmentations.degradation import (
    background_blur,
    defocus_blur,
    jpeg_compress,
    resolution_degrade,
)
from subway_defect.augmentations.scene import (
    glare_augment,
    night_augment,
    sunlitize,
    tunnelize,
    vibration_blur,
    weather_augment,
    white_balance_shift,
)

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}

# Augmentation pool for annotated images: (name, weight).  ``bgblur`` and
# ``bgswap`` are box-aware and dispatched separately.
AUG_POOL: List[Tuple[str, float]] = [
    ("vibration", 0.15),   # train-induced micro-jitter (FN: blurred frames)
    ("glare", 0.15),       # specular reflection on metal (FP: glare textures)
    ("night", 0.13),       # low-light / IR inspection
    ("tunnel", 0.12),      # tunnel lighting, most mileage underground
    ("degrade", 0.12),     # downscale→upscale: fewer effective defect pixels
    ("bgblur", 0.10),      # depth of field: sharp defect, blurred background
    ("defocus", 0.07),     # lens defocus
    ("jpeg", 0.05),        # transmission compression artifacts
    ("bgswap", 0.04),      # GrabCut + composite onto Normal_dataset pool
    ("whitebal", 0.03),    # colour-temperature shift
    ("weather", 0.02),     # fog / rain
    ("sunlit", 0.02),      # outdoor strong sunlight
]

# Variants per source training image, keyed by class code.  Rationale:
#   4 — low sample + low recall (VHBNM 84 boxes / R=0.17; CBHPM R 1.0→0.47)
#   3 — hard / unstable classes needing diverse difficult variants
#   2 — moderate reinforcement
#   1 — already sufficient or FP-driven (DRPS/GWCNL/SVHTNL); FP classes are
#       treated with hard negatives, not positive pile-ups.
CLASS_BUDGET: Dict[str, int] = {
    "VHBNM": 4, "CBHPM": 4,
    "INSD": 3, "SVHBNM": 3, "VHBNL": 3,
    "BSBM": 2, "GWCNM": 2, "SVHBNL": 2, "CBVPM": 2,
    "GWCNL": 1, "SVHTNL": 1, "DRPS": 1,
}
DEFAULT_BUDGET = 2
NEG_VARIANTS = 2          # scene variants per empty-label (background) image
RARE_BOX_THRESHOLD = 200  # classes below this get train-biased splitting
MIN_HOLDOUT = 8           # per-class val/test floor before train bias kicks in

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

_IMG_RE = re.compile(r"^(IMG_\d{8}_\d{6})(?:_\d+)?$")


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


def scan_source(src: Path, n_classes: int) -> Dict:
    """Scan flat images/ + labels/ into ImageRecords and stats."""
    img_dir, lbl_dir = src / "images", src / "labels"
    images = sorted(
        p for p in img_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES
    )
    label_files = {
        p.stem: p for p in lbl_dir.glob("*.txt") if p.name != "classes.txt"
    }

    records: List[ImageRecord] = []
    orphan_labels = sorted(set(label_files) - {p.stem for p in images})
    total_dropped_lines = 0
    for p in images:
        rec = ImageRecord(
            path=p, stem=p.stem, source_id=source_id_of(p.stem)
        )
        lf = label_files.get(p.stem)
        if lf is not None:
            rec.has_label_file = True
            rec.label_path = lf
            boxes, dropped = parse_label(
                lf.read_text(encoding="utf-8"), n_classes
            )
            rec.boxes = boxes
            total_dropped_lines += dropped
        records.append(rec)
    return {
        "records": records,
        "orphan_labels": orphan_labels,
        "dropped_lines": total_dropped_lines,
    }


def assign_splits(
    records: List[ImageRecord],
    class_sizes: Dict[str, int],
    seed: int,
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Assign every record to train/val/test by source group.

    Greedy deficit-balancing over groups (largest first, seeded shuffle
    for ties).  Groups dominated by a rare class are biased toward train,
    but only after that class has at least ``MIN_HOLDOUT`` images in both
    val and test — every class keeps an evaluation hold-out.

    Returns ``(images_per_split, group_id → split)``.
    """
    groups: Dict[str, List[ImageRecord]] = defaultdict(list)
    for rec in records:
        groups[rec.source_id].append(rec)

    rng = np.random.default_rng(seed)
    order = sorted(groups.keys())
    rng.shuffle(order)
    order = sorted(order, key=lambda g: -len(groups[g]))  # stable: size desc

    rare_codes = {c for c, n in class_sizes.items() if 0 < n < RARE_BOX_THRESHOLD}

    counts = {"train": 0, "val": 0, "test": 0}
    per_class_split: Dict[str, Counter] = defaultdict(Counter)
    group_split: Dict[str, str] = {}
    split_names = list(SPLIT_RATIOS)
    total_assigned = 0
    for i, gid in enumerate(order):
        recs = groups[gid]
        # Rare class carried by this group (if any) — for the train bias.
        rare_code = next(
            (r.majority_code for r in recs if r.majority_code in rare_codes),
            None,
        )
        # Rotate evaluation order so exact deficit ties do not always
        # favour the same split (keeps val and test balanced per class).
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


# ---------------------------------------------------------------------------
# Variant generation
# ---------------------------------------------------------------------------


class AugEngine:
    """Applies a named augmentation. Shared across worker threads."""

    def __init__(self, replacer: Optional[BackgroundReplacer]) -> None:
        self.replacer = replacer
        self._simple = {
            "vibration": vibration_blur,
            "glare": glare_augment,
            "night": night_augment,
            "tunnel": tunnelize,
            "degrade": resolution_degrade,
            "defocus": defocus_blur,
            "jpeg": jpeg_compress,
            "whitebal": white_balance_shift,
            "weather": weather_augment,
            "sunlit": sunlitize,
        }

    def apply(
        self, name: str, img: np.ndarray, boxes: List[List[float]]
    ) -> np.ndarray:
        if name == "bgblur":
            out, _ = background_blur(img, boxes)
            return out
        if name == "bgswap":
            if self.replacer is None:
                raise RuntimeError("bgswap requested without a background pool")
            out, _ = self.replacer.replace_background(img, boxes)
            return out
        return self._simple[name](img)


@dataclass
class VariantTask:
    src: ImageRecord
    aug_name: str
    variant_idx: int
    out_stem: str


def plan_variants(
    train_records: List[ImageRecord],
    seed: int,
    neg_variants: int,
) -> List[VariantTask]:
    """Deterministically choose one augmentation per variant slot."""
    names = [n for n, _ in AUG_POOL]
    weights = np.array([w for _, w in AUG_POOL], dtype=np.float64)
    weights /= weights.sum()
    names_no_box = [n for n in names if n not in ("bgblur", "bgswap")]
    weights_no_box = np.array(
        [w for n, w in AUG_POOL if n in set(names_no_box)], dtype=np.float64
    )
    weights_no_box /= weights_no_box.sum()

    tasks: List[VariantTask] = []
    for idx, rec in enumerate(train_records):
        if rec.is_negative:
            pool_names, pool_w, budget = names_no_box, weights_no_box, neg_variants
        elif rec.boxes:
            code = rec.majority_code or ""
            budget = CLASS_BUDGET.get(code, DEFAULT_BUDGET)
            pool_names, pool_w = names, weights
        else:
            continue  # no label file → excluded from the dataset entirely
        rng = np.random.default_rng(seed + idx * 7919)
        picks = rng.choice(len(pool_names), size=budget, p=pool_w)
        for vi, pi in enumerate(picks):
            aug = pool_names[int(pi)]
            out_stem = f"{rec.stem}_aug{vi}_{aug}"
            tasks.append(
                VariantTask(
                    src=rec,
                    aug_name=aug,
                    variant_idx=vi,
                    out_stem=out_stem,
                )
            )
    return tasks


def _write_label(path: Path, boxes: List[List[float]]) -> None:
    lines = [
        f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
        for b in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def place_originals(
    records: List[ImageRecord], out_root: Path, copy_mode: str
) -> Dict[str, int]:
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


def execute_variants(
    tasks: List[VariantTask],
    engine: AugEngine,
    out_root: Path,
    jpeg_quality: int,
    workers: int,
) -> Tuple[List[Dict], List[str]]:
    """Run variant tasks on a thread pool. Returns (records, errors)."""
    train_img = out_root / "train" / "images"
    train_lbl = out_root / "train" / "labels"
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)

    # Pre-write all variant labels (boxes are identical to the source).
    for t in tasks:
        _write_label(train_lbl / f"{t.out_stem}.txt", t.src.boxes)

    def run(task: VariantTask) -> Dict:
        out_path = train_img / f"{task.out_stem}.jpg"
        if out_path.exists():
            return {"out": task.out_stem, "aug": task.aug_name,
                    "src": task.src.stem, "status": "exists"}
        img = cv2.imread(str(task.src.path))
        if img is None:
            raise IOError(f"cannot read {task.src.path}")
        out = engine.apply(task.aug_name, img, task.src.boxes)
        if not cv2.imwrite(
            str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
        ):
            raise IOError(f"cannot write {out_path}")
        return {"out": task.out_stem, "aug": task.aug_name,
                "src": task.src.stem, "status": "ok"}

    records: List[Dict] = []
    errors: List[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, t): t for t in tasks}
        for fut in tqdm(
            as_completed(futures), total=len(futures),
            desc="Variants", unit="img",
        ):
            task = futures[fut]
            try:
                records.append(fut.result())
            except Exception as exc:  # noqa: BLE001 — keep the batch going
                errors.append(f"{task.src.stem} [{task.aug_name}]: {exc}")
    return records, errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expand Defect_dataset_2 into data/train_data_2"
    )
    parser.add_argument(
        "--source", type=Path,
        default=Path("data/Defect_dataset_2/Defect_dataset"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/train_data_2"))
    parser.add_argument(
        "--backgrounds", type=Path, default=Path("data/Normal_dataset/images"),
        help="Background pool for bgswap (scanned recursively).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument(
        "--copy-mode", choices=["auto", "link", "copy"], default="auto",
        help="auto = hardlink originals, fall back to copy (same volume).",
    )
    parser.add_argument(
        "--limit-groups", type=int, default=0,
        help="Process only the first N source groups (validation runs).",
    )
    parser.add_argument("--skip-variants", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = args.source
    classes = [
        ln.strip()
        for ln in (src / "classes.txt").read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    n_classes = len(classes)
    print(f"[scan] {src} — {n_classes} classes: {classes}")

    scan = scan_source(src, n_classes)
    records: List[ImageRecord] = scan["records"]
    for rec in records:
        if rec.boxes:
            cnt = Counter(int(b[0]) for b in rec.boxes)
            rec.majority_code = classes[cnt.most_common(1)[0][0]]

    annotated = [r for r in records if r.boxes]
    negatives = [r for r in records if r.is_negative]
    unlabelled = [r for r in records if not r.has_label_file]
    # Unlabelled images are pending human review (plan §3.2) and never
    # enter the dataset; orphan labels have no image at all.
    dataset_records = annotated + negatives

    class_boxes = Counter()
    for rec in annotated:
        class_boxes.update(int(b[0]) for b in rec.boxes)
    class_sizes = {classes[i]: n for i, n in class_boxes.items()}

    print(f"[scan] images={len(records)}  annotated={len(annotated)}  "
          f"background(empty label)={len(negatives)}  "
          f"unlabelled={len(unlabelled)}  orphan_labels={len(scan['orphan_labels'])}  "
          f"dropped_label_lines={scan['dropped_lines']}")
    print("[scan] per-class boxes:")
    for i, code in enumerate(classes):
        n = class_boxes.get(i, 0)
        flag = "  (rare)" if 0 < n < RARE_BOX_THRESHOLD else ""
        flag = flag or ("  (ZERO-SAMPLE — needs collection)" if n == 0 else "")
        print(f"    {i:>2} {code:8} {n:>5}{flag}")

    # ── split by source group ──────────────────────────────────────────
    _, group_split = assign_splits(dataset_records, class_sizes, args.seed)

    if args.limit_groups > 0:
        keep = set()
        seen = []
        for gid in sorted(group_split):
            seen.append(gid)
            if len(seen) >= args.limit_groups:
                break
        keep = set(seen)
        dataset_records = [r for r in dataset_records if r.source_id in keep]
        annotated = [r for r in dataset_records if r.boxes]
        negatives = [r for r in dataset_records if r.is_negative]
        group_split = {g: s for g, s in group_split.items() if g in keep}
        print(f"[limit] restricted to {len(keep)} source groups "
              f"({len(dataset_records)} images)")

    n_groups = Counter(group_split.values())
    split_counts = Counter(r.split for r in dataset_records)
    print(f"[split] source groups: train={n_groups['train']}  "
          f"val={n_groups['val']}  test={n_groups['test']}")
    print(f"[split] images: train={split_counts['train']}  "
          f"val={split_counts['val']}  test={split_counts['test']}")
    per_class_holdout = defaultdict(Counter)
    for rec in dataset_records:
        if rec.majority_code:
            per_class_holdout[rec.majority_code][rec.split] += 1
    print("[split] per-class images (train/val/test):")
    for code in classes:
        pc = per_class_holdout.get(code)
        if not pc:
            continue
        print(f"    {code:8} {pc['train']:>5}/{pc['val']:>4}/{pc['test']:>4}")

    # ── plan variants ──────────────────────────────────────────────────
    train_records = [r for r in dataset_records if r.split == "train"]
    tasks = [] if args.skip_variants else plan_variants(
        train_records, args.seed, NEG_VARIANTS
    )
    aug_hist = Counter(t.aug_name for t in tasks)

    per_class_plan = defaultdict(lambda: {"src_imgs": 0, "variants": 0})
    for rec in train_records:
        key = rec.majority_code or "background"
        per_class_plan[key]["src_imgs"] += 1
    for t in tasks:
        code = t.src.majority_code or "background"
        per_class_plan[code]["variants"] += 1

    print(f"[plan] variants to generate: {len(tasks)}")
    print(f"    {'class':10} {'train src':>9} {'variants':>8} {'budget':>6}")
    for code in classes + ["background"]:
        row = per_class_plan.get(code)
        if not row:
            continue
        budget = (f"×{CLASS_BUDGET.get(code, DEFAULT_BUDGET)}"
                  if code in classes else f"×{NEG_VARIANTS}(neg)")
        print(f"    {code:10} {row['src_imgs']:>9} {row['variants']:>8} {budget:>6}")
    print(f"[plan] augmentation mix: "
          f"{', '.join(f'{k}:{v}' for k, v in aug_hist.most_common())}")

    if args.dry_run:
        total_bytes = sum(r.path.stat().st_size for r in dataset_records)
        avg_mb = total_bytes / max(1, len(dataset_records)) / 1e6
        est_gb = len(tasks) * avg_mb * (args.jpeg_quality / 95.0) / 1024.0
        print(f"[dry-run] avg source size ≈ {avg_mb:.1f} MB  →  "
              f"variants ≈ {est_gb:.1f} GB + originals")
        print("[dry-run] nothing written.")
        return

    # ── write dataset ──────────────────────────────────────────────────
    out_root = args.output
    if any((out_root / s / "images").exists() and
           any((out_root / s / "images").iterdir())
           for s in ("train", "val", "test")):
        print(f"ERROR: {out_root} already contains data. "
              f"Remove it or choose another --output.")
        sys.exit(1)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"[write] placing originals → {out_root} (copy-mode={args.copy_mode})")
    place_stats = place_originals(dataset_records, out_root, args.copy_mode)
    print(f"[write] originals: {place_stats}")

    (out_root / "classes.txt").write_text(
        "\n".join(classes) + "\n", encoding="utf-8"
    )

    variant_records: List[Dict] = []
    errors: List[str] = []
    if tasks:
        replacer: Optional[BackgroundReplacer] = None
        if aug_hist.get("bgswap", 0) > 0:
            replacer = BackgroundReplacer(args.backgrounds, seed=args.seed)
            print(f"[bgswap] background pool: {len(replacer._bg_paths)} images "
                  f"from {args.backgrounds}")
        engine = AugEngine(replacer)
        print(f"[run] generating {len(tasks)} variants with {args.workers} workers")
        t0 = time.time()
        variant_records, errors = execute_variants(
            tasks, engine, out_root, args.jpeg_quality, args.workers
        )
        print(f"[run] done in {(time.time() - t0) / 60:.1f} min, "
              f"errors={len(errors)}")
        for e in errors[:20]:
            print(f"    ERROR {e}")

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
            boxes, _ = parse_label(lf.read_text(encoding="utf-8"), n_classes)
            n_boxes += len(boxes)
            per_cls.update(int(b[0]) for b in boxes)
        final_stats[s] = {
            "images": len(imgs),
            "labels": len(lbls),
            "missing_labels": sorted(missing_lbl)[:10],
            "orphan_labels": sorted(orphan_lbl)[:10],
            "boxes": n_boxes,
            "boxes_per_class": {classes[i]: per_cls.get(i, 0)
                                for i in range(n_classes) if per_cls.get(i, 0)},
        }
        print(f"    {s:5}: {len(imgs)} images, {n_boxes} boxes"
              + (f"  PROBLEMS={len(missing_lbl) + len(orphan_lbl)}"
                 if missing_lbl or orphan_lbl else ""))

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(src),
        "output": str(out_root),
        "seed": args.seed,
        "classes": classes,
        "config": {
            "class_budget": CLASS_BUDGET,
            "default_budget": DEFAULT_BUDGET,
            "neg_variants": NEG_VARIANTS,
            "aug_pool": AUG_POOL,
            "split_ratios": SPLIT_RATIOS,
            "rare_box_threshold": RARE_BOX_THRESHOLD,
            "min_holdout": MIN_HOLDOUT,
            "jpeg_quality": args.jpeg_quality,
            "copy_mode": args.copy_mode,
        },
        "source_groups": dict(n_groups),
        "group_assignments": group_split,
        "splits": final_stats,
        "excluded": {
            "orphan_labels": scan["orphan_labels"],
            "unlabelled_images": [r.path.name for r in unlabelled],
            "dropped_label_lines": scan["dropped_lines"],
        },
        "variant_errors": errors,
        "variants": variant_records,
    }
    mf = out_root / "manifest.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=1),
                  encoding="utf-8")
    print(f"[done] manifest → {mf}")
    if problems:
        print(f"WARNING: {problems} integrity problems — inspect manifest.")
        sys.exit(2)


if __name__ == "__main__":
    main()
