#!/usr/bin/env python3
"""Step 5: Generate synthetic missing-defect samples via inpainting.

Uses ``generate_missing_defect`` from ``subway_defect.synthetic.defect_synthesis``
to inpaint-remove the largest instance of an under-represented class, creating a
synthetic "missing component" training example.

Focuses on the five classes with the fewest training images (class 3 first).
Classes 0 and 2 (265 images each) are already well-represented and are skipped.

Usage:
    python tool/generate_synthetic_defects.py
    python tool/generate_synthetic_defects.py --target_class 3 --limit 20
"""

import argparse
import importlib.util
import sys
from pathlib import Path

from tqdm import tqdm

# ---- Load defect_synthesis.py without triggering the full package __init__ ----
_SYNTH_PATH = Path("subway_defect/synthetic/defect_synthesis.py")
_spec = importlib.util.spec_from_file_location("defect_synthesis", str(_SYNTH_PATH))
_synth = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_synth)  # type: ignore[attr-defined]

generate_missing_defect = _synth.generate_missing_defect

# Classes ordered by representation scarcity (fewest images first).
# Class 0 (VHBNM = 265) and class 2 (SVHBNM = 265) are skipped.
TARGET_CLASSES = [3, 6, 1, 5, 4]

# How many synthetic samples to generate per class (0 = generate for ALL images
# in train/ that contain the class).
SYNTH_LIMITS: dict[int, int] = {3: 100, 6: 50, 1: 50, 5: 50, 4: 50}


def _is_original(stem: str) -> bool:
    """Return True if *stem* looks like an original (not already augmented)."""
    return "_aug" not in stem and "_synth" not in stem


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic missing-defect samples",
    )
    parser.add_argument(
        "--train_images", default="data/Defect_dataset/images/train",
    )
    parser.add_argument(
        "--train_labels", default="data/Defect_dataset/labels/train",
    )
    parser.add_argument(
        "--target_classes", type=int, nargs="*", default=None,
        help="Classes to process (default: under-represented 3,6,1,5,4)",
    )
    parser.add_argument(
        "--limit_per_class", type=int, default=0,
        help="Max synthetic per class (0 = use built-in limits, -1 = unlimited)",
    )
    args = parser.parse_args()

    images_dir = Path(args.train_images)
    labels_dir = Path(args.train_labels)

    if not images_dir.is_dir():
        print(f"ERROR: {images_dir} not found. Run split_dataset.py first.")
        sys.exit(1)

    target_classes = args.target_classes or TARGET_CLASSES

    # Gather original images only
    all_images = sorted(images_dir.glob("*.jpg"))
    originals = [p for p in all_images if _is_original(p.stem)]

    total_generated = 0
    for cls_id in target_classes:
        limit = args.limit_per_class or SYNTH_LIMITS.get(cls_id, 50)
        if limit == -1:
            limit = len(originals)  # unlimited

        print(f"\n[Step 5] Class {cls_id}: generating up to {limit} synthetic samples")

        generated = 0
        for img_path in tqdm(originals, desc=f"  Class {cls_id}", unit="img"):
            if generated >= limit:
                break

            lbl_path = labels_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            # Quick pre-check: does this image contain the target class?
            has_target = False
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == cls_id:
                        has_target = True
                        break
            if not has_target:
                continue

            result = generate_missing_defect(
                image_path=img_path,
                label_path=lbl_path,
                output_img_dir=images_dir,
                output_label_dir=labels_dir,
                target_class=cls_id,
                suffix=f"_synth_missing_{cls_id}",
            )
            if result is not None:
                generated += 1

        print(f"         Class {cls_id}: {generated} synthetic samples generated")
        total_generated += generated

    print(f"\n         Total synthetic samples: {total_generated}")


if __name__ == "__main__":
    main()
