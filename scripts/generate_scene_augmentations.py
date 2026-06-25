#!/usr/bin/env python3
"""Step 4: Generate offline scene-augmented training variants (parallel).

Applies tunnelize / sunlitize / motion_blur / weather_augment from
``subway_defect.augmentations.scene`` to every original training image.
Scene-level augmentations do NOT change bounding-box positions, so the
original label file is copied verbatim for each variant.

Augmentation weights reflect typical subway operating conditions:
    tunnel 40% — most mileage is underground
    sunlit 20% — outdoor / elevated sections
    blur   20% — vehicle vibration on any section
    weather 20% — humidity fog in tunnels / rain outdoors

Performance: Uses multiprocessing (default: all CPU cores) for parallel
image I/O + augmentation.  For 399 images × 3 augs on an 8-core machine
this reduces wall time from ~3 min to ~30 s.

Usage:
    python scripts/generate_scene_augmentations.py
    python scripts/generate_scene_augmentations.py --n_augs 5 --workers 8
"""

import argparse
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

import cv2
from tqdm import tqdm

# Import scene augmentations — try package import first (needed for multiprocessing)
try:
    from subway_defect.augmentations.scene import (
        motion_blur,
        sunlitize,
        tunnelize,
        weather_augment,
    )
    _USE_PACKAGE_IMPORT = True
except ModuleNotFoundError:
    import importlib.util
    _SCENE_PATH = Path(__file__).resolve().parent.parent / "subway_defect" / "augmentations" / "scene.py"
    _spec = importlib.util.spec_from_file_location("scene", str(_SCENE_PATH))
    _scene = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_scene)
    tunnelize = _scene.tunnelize
    sunlitize = _scene.sunlitize
    motion_blur = _scene.motion_blur
    weather_augment = _scene.weather_augment
    _USE_PACKAGE_IMPORT = False

SEED = 42
N_AUGS_PER_IMAGE = 3

# Sampling weights (probabilities)
AUG_POOL = [
    ("tunnel", tunnelize, 0.40),
    ("sunlit", sunlitize, 0.20),
    ("blur", motion_blur, 0.20),
    ("weather", weather_augment, 0.20),
]
_AUG_NAMES = [a[0] for a in AUG_POOL]
_AUG_FNS = [a[1] for a in AUG_POOL]
_AUG_WEIGHTS = [a[2] for a in AUG_POOL]


def _is_original(stem: str) -> bool:
    """Return True if *stem* looks like an original (not already augmented)."""
    return "_aug" not in stem and "_synth" not in stem


def _process_one_image(
    args: Tuple[Path, Path, int, int],
) -> int:
    """Process a single image: read → apply N augs → write. Returns count generated."""
    img_path, labels_dir, n_augs, seed = args

    # Each worker needs its own random state
    local_random = random.Random(seed)

    img = cv2.imread(str(img_path))
    if img is None:
        return 0

    label_path = labels_dir / f"{img_path.stem}.txt"
    if not label_path.exists():
        return 0

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

    return generated


def main() -> None:
    cpu_count = os.cpu_count() or 4

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
        "--workers", type=int, default=cpu_count,
        help=f"Number of worker processes (default: {cpu_count})",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    images_dir = Path(args.train_images)
    labels_dir = Path(args.train_labels)

    if not images_dir.is_dir():
        print(f"ERROR: {images_dir} not found. Run split_dataset.py first.")
        return

    image_paths = sorted(images_dir.glob("*.jpg"))
    originals = [p for p in image_paths if _is_original(p.stem)]
    total_tasks = len(originals) * args.n_augs

    print(f"[Step 4] Found {len(originals)} original training images "
          f"(out of {len(image_paths)} total in train/)")
    print(f"         Generating {args.n_augs} variants per image "
          f"(~{total_tasks} total)")
    print(f"         Workers: {args.workers} (CPU cores)")

    # Build task list — each worker gets a unique per-image seed for reproducibility
    tasks = [
        (img_path, labels_dir, args.n_augs, args.seed + i)
        for i, img_path in enumerate(originals)
    ]

    generated = 0

    if _USE_PACKAGE_IMPORT and args.workers > 1:
        # ── Parallel mode (Linux: fork, or Windows with installed package) ──
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_process_one_image, task): task for task in tasks}
            with tqdm(total=len(originals), desc="Scene augmentations", unit="img") as pbar:
                for future in as_completed(futures):
                    try:
                        generated += future.result()
                    except Exception as e:
                        img_path = futures[future][0]
                        print(f"  WARNING: {img_path.name} failed: {e}")
                    pbar.update(1)
    else:
        # ── Sequential fallback (Windows dev, or single worker) ──
        if not _USE_PACKAGE_IMPORT:
            print("         [sequential mode] Run 'pip install -e .' for parallel mode")
        for task in tqdm(tasks, desc="Scene augmentations", unit="img"):
            generated += _process_one_image(task)

    print(f"         Generated {generated} augmented samples.")


if __name__ == "__main__":
    main()
