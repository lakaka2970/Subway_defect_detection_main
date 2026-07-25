#!/usr/bin/env python3
"""Generate additional hard negatives from empty-label images to reach ~40% ratio."""
from pathlib import Path
import random
import cv2
import numpy as np
from subway_defect.augmentations.scene import night_augment, glare_augment, tunnelize

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

img_dir = Path("data/subway_crops/train/images")
lbl_dir = Path("data/subway_crops/train/labels")

# Find all empty-label images (negatives) as source
sources = []
for lbl in lbl_dir.glob("*.txt"):
    if not lbl.read_text(encoding="utf-8").strip():
        img_path = img_dir / (lbl.stem + ".jpg")
        if img_path.exists() and "_hn2" not in lbl.stem:
            sources.append(img_path)

print(f"Source negatives: {len(sources)}")

# Target: ~40% negative ratio
# Current: 28755 pos, 13860 neg = 42615 total (32.5%)
# Need: neg/(pos+neg) = 0.40 → neg = 0.40 * pos / 0.60 = 19170
# Additional: 19170 - 13860 = 5310
TARGET_ADD = 5310

# Sample sources and generate 1 variant each
sample = random.sample(sources, min(TARGET_ADD, len(sources)))
aug_fns = [night_augment, glare_augment, tunnelize]

generated = 0
for i, src in enumerate(sample):
    img = cv2.imread(str(src))
    if img is None:
        continue
    fn = random.choice(aug_fns)
    try:
        aug = fn(img)
    except Exception:
        continue
    out_name = f"{src.stem}_hn2_{i}.jpg"
    cv2.imwrite(str(img_dir / out_name), aug, [cv2.IMWRITE_JPEG_QUALITY, 95])
    (lbl_dir / f"{src.stem}_hn2_{i}.txt").write_text("", encoding="utf-8")
    generated += 1
    if (generated + 1) % 1000 == 0:
        print(f"  Generated {generated + 1}/{TARGET_ADD}")

print(f"Generated {generated} additional negatives")

# Final count
pos = neg = 0
for l in lbl_dir.glob("*.txt"):
    if l.read_text(encoding="utf-8").strip():
        pos += 1
    else:
        neg += 1
total = pos + neg
print(f"Final: {total} images, {pos} pos ({pos/total*100:.1f}%), {neg} neg ({neg/total*100:.1f}%)")