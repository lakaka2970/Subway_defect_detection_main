#!/usr/bin/env python3
"""Step 4: Generate offline scene-augmented training variants (parallel).

Applies tunnelize / sunlitize / motion_blur / weather_augment /
vibration_blur / white_balance_shift from
``subway_defect.augmentations.scene`` to training images.

Scene-level augmentations do NOT change bounding-box positions, so the
original label file is copied verbatim for each variant.

Augmentation weights reflect typical subway operating conditions:
    vibration   20% — train-induced high-frequency micro-jitter
    tunnel      20% — most mileage is underground
    sunlit      12% — outdoor / elevated sections
    white_bal   12% — tunnel light colour temperature shifts
    glare       12% — reflective glare from metal / catenary wires (NEW v3)
    night       12% — night / low-light / IR inspection conditions (NEW v3)
    blur         6% — vehicle vibration on any section
    weather      6% — humidity fog in tunnels / rain outdoors

Supports both full-source-image mode (Defect_dataset) and crop-level mode
(subway_crops). In crop mode, augmentations are balanced per defect class
to prevent minority classes from being underrepresented.

Performance: Uses thread-pool parallelism (default: 8 workers) — OpenCV
releases the GIL during imread/imwrite and scene augmentations never touch
CUDA, so threads give the same throughput as processes without the
per-process memory overhead that causes OOM on cloud instances.

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
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

# Import scene augmentations — try package import first
try:
    from subway_defect.augmentations.scene import (
        glare_augment,
        motion_blur,
        night_augment,
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
    glare_augment = _scene.glare_augment
    night_augment = _scene.night_augment
    _USE_PACKAGE_IMPORT = False

# Cap workers to avoid overwhelming cloud instances (even spawn mode can
# saturate I/O / memory with too many parallel cv2.imread/cv2.imwrite).
_MAX_WORKERS = min(os.cpu_count() or 4, 8)

SEED = 42
N_AUGS_PER_IMAGE = 3

# Sampling weights (probabilities) — v3 extended pool with glare + night
AUG_POOL = [
    ("vibration", vibration_blur, 0.20),       # high-frequency micro-jitter
    ("tunnel", tunnelize, 0.20),
    ("sunlit", sunlitize, 0.12),
    ("white_bal", white_balance_shift, 0.12),  # colour temperature shifts
    ("glare", glare_augment, 0.12),            # NEW: reflective glare / specular
    ("night", night_augment, 0.12),            # NEW: night / low-light / IR
    ("blur", motion_blur, 0.06),
    ("weather", weather_augment, 0.06),
]
_AUG_NAMES = [a[0] for a in AUG_POOL]
_AUG_FNS = [a[1] for a in AUG_POOL]
_AUG_WEIGHTS = [a[2] for a in AUG_POOL]

_HN_AUG_SCHEDULES = {
    "medium": [("tunnelize", .30), ("sunlitize", .25), ("white_balance", .25),
               ("vibration_blur", .15), ("motion_blur", .05)],
    "high": [("tunnelize", .30), ("sunlitize", .25), ("white_balance", .20),
             ("vibration_blur", .15), ("motion_blur", .10)],
    "extreme": [("tunnelize", .25), ("sunlitize", .25), ("white_balance", .20),
                ("vibration_blur", .15), ("motion_blur", .15)],
}


def _motion_kernel(length: int, angle: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2
    radians = np.deg2rad(angle)
    dx, dy = np.cos(radians), np.sin(radians)
    for i in np.linspace(-center, center, length * 2):
        x = int(round(center + i * dx))
        y = int(round(center + i * dy))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1
    return kernel / max(kernel.sum(), 1)


def _apply_hard_normal_aug(img: np.ndarray, intensity: str, seed: int) -> np.ndarray:
    """Apply one deterministic aggressive augmentation to a true normal crop."""
    rng = random.Random(seed)
    schedule = _HN_AUG_SCHEDULES[intensity]
    choice = rng.choices([x[0] for x in schedule], [x[1] for x in schedule], k=1)[0]
    if choice == "tunnelize":
        return np.clip(img.astype(np.float32) * rng.uniform(.2, .5), 0, 255).astype(np.uint8)
    if choice == "sunlitize":
        contrast = rng.uniform(1.8, 3.0)
        return np.clip((img.astype(np.float32) - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
    if choice == "white_balance":
        gains = np.array([rng.uniform(.6, 1.5), 1.0, rng.uniform(.7, 1.4)],
                         dtype=np.float32).reshape(1, 1, 3)
        return np.clip(img.astype(np.float32) * gains, 0, 255).astype(np.uint8)
    if choice == "vibration_blur":
        sigma = rng.uniform(1.5, 3.0)
        kernel = rng.choice([5, 7, 9])
        return cv2.GaussianBlur(img, (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    length = rng.choice([9, 11, 13, 15])
    return cv2.filter2D(img, -1, _motion_kernel(length, rng.uniform(0, 360)))


def _find_normal_images(images_dir: Path, labels_dir: Path, max_samples: int = 0) -> List[Path]:
    normal = [
        p for p in sorted(images_dir.iterdir())
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and not (labels_dir / f"{p.stem}.txt").exists()
    ]
    if max_samples and len(normal) > max_samples:
        normal = sorted(random.Random(SEED).sample(normal, max_samples))
    return normal


def generate_hard_normals(
    normal_images: List[Path], output_dir: Path, intensity: str,
    n_per_region: int, workers: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = [(path, variant, SEED + i * 1009 + variant)
             for i, path in enumerate(normal_images) for variant in range(n_per_region)]

    def process(task: Tuple[Path, int, int]) -> Optional[Path]:
        path, variant, seed = task
        image = cv2.imread(str(path))
        if image is None or image.size == 0:
            return None
        result = _apply_hard_normal_aug(image, intensity, seed)
        output = output_dir / f"hn_{path.stem}_v{variant}.jpg"
        if not cv2.imwrite(str(output), result, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            return None
        return output

    generated = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Hard Normal", unit="img"):
            generated += int(future.result() is not None)
    return generated


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
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete all previously-generated _aug*.jpg / _aug*.txt files before "
             "generating new variants.  Use this when earlier runs were killed "
             "mid-flight and left thousands of stale images on disk.",
    )
    parser.add_argument(
        "--hard-normal", action="store_true",
        help="Aggressively augment unlabelled normal crops and create no labels.",
    )
    parser.add_argument(
        "--hn-aug-intensity", choices=["medium", "high", "extreme"], default="high",
    )
    parser.add_argument("--hn-per-region", type=int, default=3)
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Maximum source normal crops for hard-normal generation (0 = all).",
    )
    parser.add_argument(
        "--hn-output", type=Path, default=None,
        help="Hard-normal image output (default: <train_images>/hard_normals).",
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

    if args.hard_normal:
        normal_images = _find_normal_images(images_dir, labels_dir, args.max_samples)
        if not normal_images:
            raise SystemExit(f"No unlabelled normal images found in {images_dir}")
        output = args.hn_output or (images_dir / "hard_normals")
        generated = generate_hard_normals(
            normal_images, output, args.hn_aug_intensity, args.hn_per_region, workers,
        )
        print(f"Generated {generated}/{len(normal_images) * args.hn_per_region} "
              f"hard-normal images in {output}; no labels were created.")
        if generated < 2000:
            raise SystemExit("Hard Normal acceptance failed: fewer than 2,000 images")
        return

    # ── Cleanup of stale augmentations from previous killed runs ─────────
    if args.clean:
        _AUG_GLOB = ("*_aug*.*",)
        stale_imgs = list(images_dir.glob("*_aug*.jpg"))
        stale_lbls = list(labels_dir.glob("*_aug*.txt")) if labels_dir.is_dir() else []
        stale_total = len(stale_imgs) + len(stale_lbls)
        if stale_total > 0:
            print(f"  [clean] Removing {len(stale_imgs)} stale augmented images "
                  f"and {len(stale_lbls)} stale labels...")
            for f in stale_imgs:
                f.unlink()
            for f in stale_lbls:
                f.unlink()
            print(f"  [clean] Done — {stale_total} files removed.\n")

    image_paths = sorted(images_dir.glob("*.jpg"))
    originals = [p for p in image_paths if _is_original(p.stem)]
    n_stale = len(image_paths) - len(originals)
    if n_stale > len(originals):
        print(f"  [WARN] {n_stale} stale augmented files detected "
              f"({len(originals)} originals).  Re-run with --clean to remove them.\n")

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
        # ── Thread-parallel mode ─────────────────────────────────────────
        # Threads are safe here: scene augmentations never touch CUDA, and
        # OpenCV's imread/imwrite release the GIL.  ThreadPoolExecutor uses
        # shared memory → no per-worker Python-interpreter overhead that
        # causes OOM with ProcessPoolExecutor on small cloud instances.
        print(f"         [thread-pool mode — {workers} workers]\n")

        # Submit tasks in batches to keep the memory of in-flight futures
        # bounded (1134 futures × pickled args would also spike RAM).
        _BATCH = max(workers * 8, 64)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            with tqdm(total=len(tasks), desc="Scene augmentations", unit="img") as pbar:
                for i in range(0, len(tasks), _BATCH):
                    batch = tasks[i:i + _BATCH]
                    futures = {executor.submit(_process_one_image, t): t for t in batch}
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
