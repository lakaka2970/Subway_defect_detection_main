#!/usr/bin/env python3
"""Step 4: Generate offline scene-augmented training variants (parallel).

Applies tunnelize / sunlitize / motion_blur / weather_augment /
vibration_blur / white_balance_shift from
``subway_defect.augmentations.scene`` to training images.

Scene-level augmentations do NOT change bounding-box positions, so the
original label file is copied verbatim for each variant.

Augmentation weights reflect typical subway operating conditions:
    vibration   25% — train-induced high-frequency micro-jitter (NEW)
    tunnel      25% — most mileage is underground
    sunlit      15% — outdoor / elevated sections
    white_bal   15% — tunnel light colour temperature shifts (NEW)
    blur        10% — vehicle vibration on any section
    weather     10% — humidity fog in tunnels / rain outdoors

Supports both full-source-image mode (Defect_dataset) and crop-level mode
(subway_crops). In crop mode, augmentations are balanced per defect class
to prevent minority classes from being underrepresented.

Performance: Uses multiprocessing (default: all CPU cores) for parallel
image I/O + augmentation.

Usage:
    # Original full-image mode
    python scripts/generate_scene_augmentations.py

    # Crop-level mode (recommended for Stage 2+ training)
    python scripts/generate_scene_augmentations.py \\
        --train_images data/subway_crops/train/images \\
        --train_labels data/subway_crops/train/labels

    # More variants per image, more workers
    python scripts/generate_scene_augmentations.py --n_augs 5 --workers 8
"""

import argparse
import multiprocessing
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
from tqdm import tqdm

# Force "spawn" on Linux — "fork" deadlocks when OpenCV is built with CUDA
# (common on AutoDL / cloud GPU instances).  Must happen BEFORE any CUDA-
# adjacent library is imported.
if hasattr(multiprocessing, "set_start_method"):
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set by another component

# Import scene augmentations — try package import first
try:
    from subway_defect.augmentations.scene import (
        motion_blur,
        sunlitize,
        tunnelize,
        vibration_blur,
        weather_augment,
        white_balance_shift,
    )
    _USE_PACKAGE_IMPORT = True
except (ModuleNotFoundError, ImportError):
    import importlib.util
    _SCENE_PATH = Path(__file__).resolve().parent.parent / "subway_defect" / "augmentations" / "scene.py"
    _spec = importlib.util.spec_from_file_location("scene", str(_SCENE_PATH))
    _scene = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_scene)
    tunnelize = _scene.tunnelize
    sunlitize = _scene.sunlitize
    motion_blur = _scene.motion_blur
    vibration_blur = _scene.vibration_blur
    weather_augment = _scene.weather_augment
    white_balance_shift = _scene.white_balance_shift
    _USE_PACKAGE_IMPORT = False

# Cap workers to avoid overwhelming cloud instances (even spawn mode can
# saturate I/O / memory with too many parallel cv2.imread/cv2.imwrite).
_MAX_WORKERS = min(os.cpu_count() or 4, 8)

SEED = 42
N_AUGS_PER_IMAGE = 3

# Sampling weights (probabilities) — v2 extended pool
AUG_POOL = [
    ("vibration", vibration_blur, 0.25),       # NEW: high-frequency micro-jitter
    ("tunnel", tunnelize, 0.25),
    ("sunlit", sunlitize, 0.15),
    ("white_bal", white_balance_shift, 0.15),  # NEW: colour temperature shifts
    ("blur", motion_blur, 0.10),
    ("weather", weather_augment, 0.10),
]
_AUG_NAMES = [a[0] for a in AUG_POOL]
_AUG_FNS = [a[1] for a in AUG_POOL]
_AUG_WEIGHTS = [a[2] for a in AUG_POOL]


def _is_original(stem: str) -> bool:
    """Return True if *stem* looks like an original (not already augmented)."""
    return "_aug" not in stem and "_synth" not in stem


def _process_one_image(
    args: Tuple[Path, Path, int, int],
) -> Tuple[int, int]:
    """Process a single image: read → apply N augs → write. Returns (generated, class_id).

    Images with **empty or invalid** label files (no defect annotations) are skipped
    — augmentation would produce useless empty-label copies that fail dataset validation.

    For class-aware mode, returns the majority class_id in this image so the caller
    can track per-class augmentation counts.
    """
    img_path, labels_dir, n_augs, seed = args

    # Each worker needs its own random state
    local_random = random.Random(seed)

    img = cv2.imread(str(img_path))
    if img is None:
        return 0, -1

    label_path = labels_dir / f"{img_path.stem}.txt"
    if not label_path.exists():
        return 0, -1

    # Skip images without annotations — an empty label file means the image
    # contains no defects; augmenting it would only produce more empty-label
    # copies that trip dataset validation and waste training I/O.
    label_text = label_path.read_text(encoding="utf-8").strip()
    if not label_text:
        return 0, -1

    # Verify at least one line is a valid YOLO annotation (5 space-separated numbers)
    # and determine the majority class for class-aware balancing
    class_counts: Dict[int, int] = {}
    for line in label_text.splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                [float(p) for p in parts]
                cls_id = int(parts[0])
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
            except ValueError:
                continue
    if not class_counts:
        return 0, -1

    majority_cls = max(class_counts, key=class_counts.get)  # type: ignore[arg-type]

    generated = 0
    for aug_idx in range(n_augs):
        choices = local_random.choices(
            population=list(zip(_AUG_NAMES, _AUG_FNS)),
            weights=_AUG_WEIGHTS,
            k=1,
        )
        aug_name, aug_fn = choices[0]

        try:
            augmented = aug_fn(img)
        except Exception:
            continue

        out_stem = f"{img_path.stem}_aug{aug_idx}_{aug_name}"
        out_img = img_path.parent / f"{out_stem}.jpg"
        out_lbl = labels_dir / f"{out_stem}.txt"

        cv2.imwrite(str(out_img), augmented)
        shutil.copy2(label_path, out_lbl)
        generated += 1

    return generated, majority_cls


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline scene augmentations (parallel)")
    parser.add_argument(
        "--train_images", default="data/Defect_dataset/images/train",
    )
    parser.add_argument(
        "--train_labels", default="data/Defect_dataset/labels/train",
    )
    parser.add_argument("--n_augs", type=int, default=N_AUGS_PER_IMAGE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--workers", type=int, default=_MAX_WORKERS,
        help=f"Number of worker processes (default: {_MAX_WORKERS}, max: {_MAX_WORKERS})",
    )
    parser.add_argument(
        "--no-parallel", action="store_true",
        help="Force sequential mode (useful for debugging / resource-constrained instances)",
    )
    parser.add_argument(
        "--class-aware", action="store_true",
        help="Balance augmentations per defect class — minority classes get more variants. "
             "Best used with subway_crops where each crop has a dominant class.",
    )
    parser.add_argument(
        "--minority-multiplier", type=float, default=1.8,
        help="Extra augmentations multiplier for classes below median count (default: 1.8)",
    )
    args = parser.parse_args()

    # Clamp workers to safe range
    workers = min(args.workers, _MAX_WORKERS)

    random.seed(args.seed)

    images_dir = Path(args.train_images)
    labels_dir = Path(args.train_labels)

    if not images_dir.is_dir():
        print(f"ERROR: {images_dir} not found. Run split_dataset.py first.")
        return

    image_paths = sorted(images_dir.glob("*.jpg"))
    originals = [p for p in image_paths if _is_original(p.stem)]

    # ── Class-aware balancing logic ────────────────────────────────────
    if args.class_aware:
        # First pass: count images per class (by majority class in labels)
        per_class_count: Dict[int, int] = {}
        per_class_paths: Dict[int, List[Path]] = {}
        for p in originals:
            lbl_path = labels_dir / f"{p.stem}.txt"
            if not lbl_path.exists():
                continue
            cls_counts: Dict[int, int] = {}
            for line in lbl_path.read_text(encoding="utf-8").strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        cls_id = int(parts[0])
                        cls_counts[cls_id] = cls_counts.get(cls_id, 0) + 1
                    except ValueError:
                        continue
            if not cls_counts:
                continue
            maj_cls = max(cls_counts, key=cls_counts.get)
            per_class_count[maj_cls] = per_class_count.get(maj_cls, 0) + 1
            per_class_paths.setdefault(maj_cls, []).append(p)

        if per_class_count:
            median_count = sorted(per_class_count.values())[len(per_class_count) // 2]
            print(f"  Class-aware mode: {len(per_class_count)} classes detected")
            print(f"  Median images/class: {median_count}")
            for cls_id in sorted(per_class_count):
                cnt = per_class_count[cls_id]
                extra = ""
                if cnt < median_count and args.minority_multiplier > 1.0:
                    extra = f" → {int(cnt * args.minority_multiplier)} (×{args.minority_multiplier})"
                print(f"    Class {cls_id}: {cnt} images{extra}")

        # Build task list with per-class multipliers
        tasks: List[Tuple[Path, Path, int, int]] = []
        for cls_id, paths in per_class_paths.items():
            n_augs_for_class = args.n_augs
            if per_class_count[cls_id] < median_count:
                n_augs_for_class = max(args.n_augs,
                                       int(args.n_augs * args.minority_multiplier))
            for i, img_path in enumerate(paths):
                tasks.append((img_path, labels_dir, n_augs_for_class,
                             args.seed + i * 1000 + cls_id))
        total_tasks = sum(t[2] for t in tasks)
    else:
        tasks = [
            (img_path, labels_dir, args.n_augs, args.seed + i)
            for i, img_path in enumerate(originals)
        ]
        total_tasks = len(originals) * args.n_augs

    print(f"\n[Step 4] Found {len(originals)} original training images "
          f"(out of {len(image_paths)} total in {images_dir})")
    print(f"         Generating {args.n_augs} variants per image "
          f"(~{total_tasks} total, class_aware={args.class_aware})")
    print(f"         Workers: {workers}")
    print(f"         Augmentation pool: {[a[0] for a in AUG_POOL]}")

    generated = 0
    per_class_generated: Dict[int, int] = {}
    use_parallel = _USE_PACKAGE_IMPORT and workers > 1 and not args.no_parallel

    if use_parallel:
        # ── spawn-safe parallel mode ──────────────────────────────────
        print("         [parallel mode — spawn start method]\n")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_process_one_image, task): task for task in tasks}
            with tqdm(total=len(tasks), desc="Scene augmentations", unit="img") as pbar:
                for future in as_completed(futures):
                    try:
                        gen, cls_id = future.result()
                        generated += gen
                        if cls_id >= 0:
                            per_class_generated[cls_id] = \
                                per_class_generated.get(cls_id, 0) + gen
                    except Exception as e:
                        img_path = futures[future][0]
                        print(f"  WARNING: {img_path.name} failed: {e}")
                    pbar.update(1)
    else:
        # ── Sequential fallback ───────────────────────────────────────
        if not _USE_PACKAGE_IMPORT:
            print("         [sequential mode] Run 'pip install -e .' for parallel mode")
        elif args.no_parallel:
            print("         [sequential mode] --no-parallel flag set")
        for task in tqdm(tasks, desc="Scene augmentations", unit="img"):
            gen, cls_id = _process_one_image(task)
            generated += gen
            if cls_id >= 0:
                per_class_generated[cls_id] = per_class_generated.get(cls_id, 0) + gen

    print(f"\n         Generated {generated} augmented samples.")
    if per_class_generated:
        print(f"         Per-class distribution:")
        for cls_id in sorted(per_class_generated):
            print(f"           Class {cls_id}: {per_class_generated[cls_id]} augmented variants")


if __name__ == "__main__":
    main()
