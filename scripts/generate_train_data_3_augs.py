#!/usr/bin/env python3
"""Generate augmented variants for the train_data_3 pipeline.

Reads the raw pool ``data/train_data_3_raw`` (already split train/val/test
by ``build_train_data_3_raw.py``) and produces the final training dataset
``data/train_data_3``:

    data/train_data_3/
    ├── train/images  train/labels   originals + augmented variants
    ├── val/images    val/labels     originals only (calibration)
    ├── test/images   test/labels    originals only (frozen hold-out)
    ├── classes.txt
    ├── data.yaml
    └── manifest.json

Key differences from the superseded ``expand_defect_dataset.py``:

* Fresh augmentation pool with **three new transforms** — CLAHE contrast,
  elastic deformation (label-aware), and defect-local shadow — plus
  defect-local glare (replacing random full-frame glare).
* **Increased dark proportion** (night 0.18 + tunnel 0.14) and a "dark
  guarantee" (≥1 dark variant per image with budget ≥2).
* **Refined per-class budget** — rare / poorly-recognised classes get MORE
  variants, good classes FEWER (inverse of the old philosophy).
* **Per-class augmentation-weight overrides** — weak classes sample a
  re-weighted pool favouring their targeted transforms.

Usage:
    python scripts/generate_train_data_3_augs.py --dry-run
    python scripts/generate_train_data_3_augs.py --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Make the repo root importable regardless of how the script is invoked
# (``python scripts/…`` puts ``scripts/`` — not the repo root — on sys.path).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
from subway_defect.augmentations.local_contrast import (
    clahe_contrast,
    defect_glare,
    defect_shadow,
    elastic_deform,
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

CLASS_NAMES = [
    "VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
    "RHTBNM", "RHTBNL", "GWCSBNM", "GWCSBNL", "GWCNM", "GWCNL",
    "BSBM", "INSD", "DRPS",
]
N_CLASSES = len(CLASS_NAMES)

# Fresh augmentation pool: (name, relative weight).  ``glare`` is
# defect-local (dispatched specially); ``bgblur``/``bgswap``/``elastic``/
# ``defect_shadow`` are box-aware.
AUG_POOL: List[Tuple[str, float]] = [
    ("night", 0.18),          # dark (increased)
    ("glare", 0.15),          # defect-local glare
    ("tunnel", 0.14),         # dark (increased)
    ("clahe", 0.10),          # NEW — local contrast
    ("elastic", 0.08),        # NEW — label-aware elastic warp
    ("vibration", 0.08),
    ("defocus", 0.07),
    ("degrade", 0.06),
    ("bgblur", 0.06),
    ("defect_shadow", 0.05),  # NEW — shadow near defect
    ("jpeg", 0.03),
    ("bgswap", 0.03),
    ("whitebal", 0.02),
    ("weather", 0.01),
    ("sunlit", 0.01),
]

# Variants per source training image.  Inverse of the old philosophy:
# rare / poorly-recognised classes get MORE variants, good classes FEWER.
CLASS_BUDGET: Dict[str, int] = {
    "VHBNL": 5,    # R=0.506 worst missed detection → most augmentation
    "GWCNM": 4,    # P=0.727 worst false positive
    "CBVPM": 4,    # P=0.765 high FP
    "SVHBNL": 4,   # R=0.798 subtle loose-nut signature
    "DRPS": 3,     # mAP50-95=0.515 poor localisation
    "VHBNM": 2,    # good but few samples (340)
    "CBHPM": 2,    # good, now ample after +313 merge
    "INSD": 2,
    "SVHBNM": 2,
    "BSBM": 2,
    "GWCNL": 1,    # P=1.000 excellent — keep few
    "SVHTNL": 1,   # P=1.000 excellent — keep few
}
DEFAULT_BUDGET = 2
NEG_VARIANTS = 2

# Per-class augmentation-weight multipliers (finer algorithmic measures).
CLASS_AUG_OVERRIDES: Dict[str, Dict[str, float]] = {
    # 松动类 (loose parts, visually subtle): more elastic + CLAHE.
    "VHBNL": {"elastic": 1.5, "clahe": 1.3},
    "SVHBNL": {"elastic": 1.5, "clahe": 1.3},
    # 高误检 (high FP): more background diversity + occlusion robustness.
    "GWCNM": {"bgswap": 2.0, "defect_shadow": 1.5, "glare": 1.3},
    "CBVPM": {"bgswap": 2.0, "defect_shadow": 1.5, "glare": 1.3},
    # 定位差 (poor localisation): more multi-scale + elastic.
    "DRPS": {"degrade": 1.8, "defocus": 1.8, "elastic": 1.5},
}

DARK_AUGS = {"night", "tunnel"}
BOX_AWARE = {"glare", "elastic", "defect_shadow", "bgblur", "bgswap"}


@dataclass
class ImageRecord:
    path: Path
    stem: str
    boxes: List[List[float]] = field(default_factory=list)
    has_label_file: bool = False
    majority_code: Optional[str] = None
    split: str = ""

    @property
    def is_negative(self) -> bool:
        return self.has_label_file and not self.boxes


class VariantTask:
    def __init__(self, src: ImageRecord, aug_name: str, variant_idx: int, out_stem: str):
        self.src = src
        self.aug_name = aug_name
        self.variant_idx = variant_idx
        self.out_stem = out_stem


def parse_label(text: str, n_classes: int) -> List[List[float]]:
    """Parse YOLO label text → valid boxes (malformed lines dropped)."""
    boxes: List[List[float]] = []
    for line in text.splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        try:
            cls = int(parts[0])
            coords = [float(v) for v in parts[1:]]
        except ValueError:
            continue
        if not (0 <= cls < n_classes) or any(not 0.0 <= v <= 1.0 for v in coords):
            continue
        if coords[2] <= 0 or coords[3] <= 0:
            continue
        boxes.append([cls] + coords)
    return boxes


def _write_label(path: Path, boxes: List[List[float]]) -> None:
    lines = [
        f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_records(split_root: Path, split: str) -> List[ImageRecord]:
    """Load image records (with labels) for one already-split raw dir."""
    img_dir = split_root / "images"
    lbl_dir = split_root / "labels"
    records: List[ImageRecord] = []
    for p in sorted(img_dir.iterdir()):
        if not p.is_file() or p.suffix.lower() not in _IMG_SUFFIXES:
            continue
        rec = ImageRecord(path=p, stem=p.stem, split=split)
        lf = lbl_dir / (p.stem + ".txt")
        if lf.exists():
            rec.has_label_file = True
            rec.boxes = parse_label(lf.read_text(encoding="utf-8"), N_CLASSES)
        records.append(rec)
    for rec in records:
        if rec.boxes:
            cnt = Counter(int(b[0]) for b in rec.boxes)
            rec.majority_code = CLASS_NAMES[cnt.most_common(1)[0][0]]
    return records


class AugEngine:
    """Applies a named augmentation, returning ``(img, boxes)`` uniformly."""

    def __init__(self, replacer: Optional[BackgroundReplacer]) -> None:
        self.replacer = replacer
        self._simple = {
            "night": night_augment,
            "tunnel": tunnelize,
            "clahe": clahe_contrast,
            "defocus": defocus_blur,
            "vibration": vibration_blur,
            "degrade": resolution_degrade,
            "jpeg": jpeg_compress,
            "whitebal": white_balance_shift,
            "weather": weather_augment,
            "sunlit": sunlitize,
        }

    def apply(self, name: str, img: np.ndarray, boxes: List[List[float]]) -> Tuple[np.ndarray, List[List[float]]]:
        boxes = list(boxes or [])
        if name == "elastic":
            out, new_boxes = elastic_deform(img, boxes)
            return out, new_boxes
        if name == "glare":
            if boxes:
                return defect_glare(img, boxes), boxes
            return glare_augment(img), boxes
        if name == "defect_shadow":
            return defect_shadow(img, boxes), boxes
        if name == "bgblur":
            out, _ = background_blur(img, boxes)
            return out, boxes
        if name == "bgswap":
            if self.replacer is None:
                raise RuntimeError("bgswap requested without a background pool")
            out, _ = self.replacer.replace_background(img, boxes)
            return out, boxes
        return self._simple[name](img), boxes


def plan_variants(
    train_records: List[ImageRecord],
    seed: int,
    neg_variants: int,
) -> List[VariantTask]:
    """Deterministically choose augmentations per variant slot."""
    names = [n for n, _ in AUG_POOL]
    base_w = np.array([w for _, w in AUG_POOL], dtype=np.float64)
    name_idx = {n: i for i, n in enumerate(names)}
    box_agnostic = [i for i, n in enumerate(names) if n not in BOX_AWARE]

    tasks: List[VariantTask] = []
    for idx, rec in enumerate(train_records):
        rng = np.random.default_rng(seed + idx * 7919)

        if rec.is_negative:
            budget = neg_variants
            pool_idx = box_agnostic
            w = base_w[box_agnostic].copy()
            code = None
        elif rec.boxes:
            code = rec.majority_code or ""
            budget = CLASS_BUDGET.get(code, DEFAULT_BUDGET)
            pool_idx = list(range(len(names)))
            w = base_w.copy()
            for aug, mult in CLASS_AUG_OVERRIDES.get(code, {}).items():
                w[name_idx[aug]] *= mult
        else:
            continue  # no label file → excluded entirely

        w /= w.sum()
        n = len(pool_idx)
        if budget <= n:
            picks_idx = rng.choice(n, size=budget, p=w, replace=False)
        else:
            picks_idx = rng.choice(n, size=budget, p=w, replace=True)
        picks = [names[pool_idx[i]] for i in picks_idx]

        # Dark guarantee: ensure ≥1 dark variant for budget ≥2.
        if budget >= 2 and not any(p in DARK_AUGS for p in picks):
            picks[-1] = str(rng.choice(sorted(DARK_AUGS)))

        for vi, aug in enumerate(picks):
            out_stem = f"{rec.stem}_aug{vi}_{aug}"
            tasks.append(VariantTask(rec, aug, vi, out_stem))
    return tasks


def place_originals(records: List[ImageRecord], out_root: Path, copy_mode: str) -> Dict[str, int]:
    """Hardlink/copy originals into split dirs; write normalised labels."""
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
    """Run variant tasks on a thread pool, writing (possibly transformed) labels."""
    train_img = out_root / "train" / "images"
    train_lbl = out_root / "train" / "labels"
    train_img.mkdir(parents=True, exist_ok=True)
    train_lbl.mkdir(parents=True, exist_ok=True)

    def run(task: VariantTask) -> Dict:
        out_path = train_img / f"{task.out_stem}.jpg"
        if out_path.exists():
            return {"out": task.out_stem, "aug": task.aug_name,
                    "src": task.src.stem, "status": "exists"}
        img = cv2.imread(str(task.src.path))
        if img is None:
            raise IOError(f"cannot read {task.src.path}")
        out, boxes = engine.apply(task.aug_name, img, task.src.boxes)
        _write_label(train_lbl / f"{task.out_stem}.txt", boxes)
        if not cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]):
            raise IOError(f"cannot write {out_path}")
        return {"out": task.out_stem, "aug": task.aug_name,
                "src": task.src.stem, "status": "ok"}

    records: List[Dict] = []
    errors: List[str] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run, t): t for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Variants", unit="img"):
            task = futures[fut]
            try:
                records.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{task.src.stem} [{task.aug_name}]: {exc}")
    return records, errors


def write_data_yaml(out_root: Path) -> None:
    lines = [
        "# Train data for 16-class defect detection (train_data_3, augmented)",
        "# Auto-generated by scripts/generate_train_data_3_augs.py",
        f"# Date: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "path: data/train_data_3",
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
        description="Generate train_data_3 augmentations from train_data_3_raw"
    )
    parser.add_argument("--raw", type=Path, default=Path("data/train_data_3_raw"))
    parser.add_argument("--output", type=Path, default=Path("data/train_data_3"))
    parser.add_argument(
        "--backgrounds", type=Path, default=Path("data/Normal_dataset/images"),
        help="Background pool for bgswap (scanned recursively).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--copy-mode", choices=["auto", "link", "copy"], default="auto")
    parser.add_argument("--skip-variants", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # ── load raw split records ────────────────────────────────────────
    all_records: Dict[str, List[ImageRecord]] = {}
    for split in ("train", "val", "test"):
        records = load_records(args.raw / split, split)
        all_records[split] = records
        n_boxed = sum(1 for r in records if r.boxes)
        n_neg = sum(1 for r in records if r.is_negative)
        print(f"[raw:{split}] images={len(records)}  boxed={n_boxed}  background={n_neg}")

    # ── plan variants (train only) ────────────────────────────────────
    train_records = all_records["train"]
    tasks = [] if args.skip_variants else plan_variants(train_records, args.seed, NEG_VARIANTS)
    aug_hist = Counter(t.aug_name for t in tasks)
    per_class_plan = defaultdict(lambda: {"src": 0, "variants": 0})
    for rec in train_records:
        per_class_plan[rec.majority_code or "background"]["src"] += 1
    for t in tasks:
        per_class_plan[t.src.majority_code or "background"]["variants"] += 1

    print(f"[plan] variants to generate: {len(tasks)}")
    print(f"    {'class':10} {'train src':>9} {'variants':>8} {'budget':>7}")
    for code in CLASS_NAMES + ["background"]:
        row = per_class_plan.get(code)
        if not row:
            continue
        budget = (f"×{CLASS_BUDGET.get(code, DEFAULT_BUDGET)}"
                  if code in CLASS_NAMES else f"×{NEG_VARIANTS}(neg)")
        print(f"    {code:10} {row['src']:>9} {row['variants']:>8} {budget:>7}")
    print(f"[plan] augmentation mix: {', '.join(f'{k}:{v}' for k, v in aug_hist.most_common())}")

    if args.dry_run:
        total_bytes = sum(r.path.stat().st_size for r in train_records)
        avg_mb = total_bytes / max(1, len(train_records)) / 1e6
        est_gb = len(tasks) * avg_mb * (args.jpeg_quality / 95.0) / 1024.0
        print(f"[dry-run] avg source size ≈ {avg_mb:.1f} MB → variants ≈ {est_gb:.1f} GB")
        print("[dry-run] nothing written.")
        return

    # ── write dataset ─────────────────────────────────────────────────
    out_root = args.output
    if any((out_root / s / "images").exists() and any((out_root / s / "images").iterdir())
           for s in ("train", "val", "test")):
        print(f"ERROR: {out_root} already contains data. Remove it or choose --output.")
        sys.exit(1)
    out_root.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val", "test"):
        stats = place_originals(all_records[split], out_root, args.copy_mode)
        print(f"[write] {split} originals: {stats}")

    (out_root / "classes.txt").write_text("\n".join(CLASS_NAMES) + "\n", encoding="utf-8")
    write_data_yaml(out_root)

    variant_records: List[Dict] = []
    errors: List[str] = []
    if tasks:
        replacer: Optional[BackgroundReplacer] = None
        if aug_hist.get("bgswap", 0) > 0:
            replacer = BackgroundReplacer(args.backgrounds, seed=args.seed)
            print(f"[bgswap] background pool: {len(replacer._bg_paths)} images from {args.backgrounds}")
        engine = AugEngine(replacer)
        print(f"[run] generating {len(tasks)} variants with {args.workers} workers")
        t0 = time.time()
        variant_records, errors = execute_variants(
            tasks, engine, out_root, args.jpeg_quality, args.workers
        )
        print(f"[run] done in {(time.time() - t0) / 60:.1f} min, errors={len(errors)}")
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
            boxes = parse_label(lf.read_text(encoding="utf-8"), N_CLASSES)
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
        "source": str(args.raw),
        "output": str(out_root),
        "seed": args.seed,
        "classes": CLASS_NAMES,
        "config": {
            "class_budget": CLASS_BUDGET,
            "default_budget": DEFAULT_BUDGET,
            "neg_variants": NEG_VARIANTS,
            "aug_pool": AUG_POOL,
            "class_aug_overrides": CLASS_AUG_OVERRIDES,
            "dark_augs": sorted(DARK_AUGS),
            "jpeg_quality": args.jpeg_quality,
            "copy_mode": args.copy_mode,
        },
        "splits": final_stats,
        "variant_errors": errors,
        "variants": variant_records,
    }
    mf = out_root / "manifest.json"
    mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] manifest → {mf}")
    if problems:
        print(f"WARNING: {problems} integrity problems — inspect manifest.")
        sys.exit(2)


if __name__ == "__main__":
    main()
