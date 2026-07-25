#!/usr/bin/env python3
"""Move hard negatives to training set and report final composition."""
from pathlib import Path
import shutil

hn_dir = Path("data/subway_crops/train/images/hard_normals")
img_dir = Path("data/subway_crops/train/images")
lbl_dir = Path("data/subway_crops/train/labels")

moved = 0
for f in hn_dir.glob("*.jpg"):
    shutil.move(str(f), str(img_dir / f.name))
    (lbl_dir / (f.stem + ".txt")).write_text("", encoding="utf-8")
    moved += 1

try:
    hn_dir.rmdir()
except OSError:
    pass

print(f"Moved {moved} hard negatives to training set with empty labels")

pos = neg = 0
for l in lbl_dir.glob("*.txt"):
    if l.read_text(encoding="utf-8").strip():
        pos += 1
    else:
        neg += 1
total = pos + neg
print(f"Final: {total} images, {pos} pos ({pos/total*100:.1f}%), {neg} neg ({neg/total*100:.1f}%)")