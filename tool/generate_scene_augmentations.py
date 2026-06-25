#!/usr/bin/env python3
"""Step 4: Generate offline scene-augmented training variants.

Applies tunnelize / sunlitize / motion_blur / weather_augment from
``subway_defect.augmentations.scene`` to every original training image.
Scene-level augmentations do NOT change bounding-box positions, so the
original label file is copied verbatim for each variant.

Augmentation weights reflect typical subway operating conditions:
    tunnel 40% — most mileage is underground
    sunlit 20% — outdoor / elevated sections
    blur   20% — vehicle vibration on any section
    weather 20% — humidity fog in tunnels / rain outdoors

Usage:
    python tool/generate_scene_augmentations.py
    python tool/generate_scene_augmentations.py --n_augs 5 --seed 123
"""

import argparse
import importlib.util
import random
import shutil
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

# ---- Load scene.py without triggering the full package __init__ chain ----
_SCENE_PATH = Path("subway_defect/augmentations/scene.py")
_spec = importlib.util.spec_from_file_location("scene", str(_SCENE_PATH))
_scene = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scene)  # type: ignore[attr-defined]

tunnelize = _scene.tunnelize
sunlitize = _scene.sunlitize
motion_blur = _scene.motion_blur
weather_augment = _scene.weather_augment

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline scene augmentations")
    parser.add_argument(
        "--train_images", default="data/Defect_dataset/images/train",
    )
    parser.add_argument(
        "--train_labels", default="data/Defect_dataset/labels/train",
    )
    parser.add_argument("--n_augs", type=int, default=N_AUGS_PER_IMAGE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    random.seed(args.seed)

    images_dir = Path(args.train_images)
    labels_dir = Path(args.train_labels)

    if not images_dir.is_dir():
        print(f"ERROR: {images_dir} not found. Run split_dataset.py first.")
        sys.exit(1)

    image_paths = sorted(images_dir.glob("*.jpg"))
    originals = [p for p in image_paths if _is_original(p.stem)]
    print(f"[Step 4] Found {len(originals)} original training images "
          f"(out of {len(image_paths)} total in train/)")
    print(f"         Generating {args.n_augs} variants per image "
          f"(~{len(originals) * args.n_augs} total)")

    generated = 0
    for img_path in tqdm(originals, desc="Scene augmentations", unit="img"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARNING: cannot read {img_path.name}, skipping")
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"  WARNING: no label for {img_path.name}, skipping")
            continue

        for aug_idx in range(args.n_augs):
            # Weighted random choice
            choices = random.choices(
                population=list(zip(_AUG_NAMES, _AUG_FNS)),
                weights=_AUG_WEIGHTS,
                k=1,
            )
            aug_name, aug_fn = choices[0]

            try:
                augmented = aug_fn(img)
            except Exception:
                print(f"  WARNING: {aug_name} failed on {img_path.name}, skipping")
                continue

            out_stem = f"{img_path.stem}_aug{aug_idx}_{aug_name}"
            out_img = images_dir / f"{out_stem}.jpg"
            out_lbl = labels_dir / f"{out_stem}.txt"

            cv2.imwrite(str(out_img), augmented)
            shutil.copy2(label_path, out_lbl)
            generated += 1

    print(f"         Generated {generated} augmented samples.")


if __name__ == "__main__":
    main()
