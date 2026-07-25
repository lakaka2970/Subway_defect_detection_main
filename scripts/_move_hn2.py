#!/usr/bin/env python3
"""Move hard negatives batch 2 and report final composition."""
from pathlib import Path
import shutil

hn = Path("data/subway_crops/train/images/hard_normals2")
img = Path("data/subway_crops/train/images")
lbl = Path("data/subway_crops/train/labels")

moved = 0
if hn.exists():
    for f in hn.glob("*.jpg"):
        shutil.move(str(f), str(img / f.name))
        (lbl / (f.stem + ".txt")).write_text("", encoding="utf-8")
        moved += 1
    try:
        hn.rmdir()
    except OSError:
        pass
print(f"Moved {moved} hard negatives (batch 2)")

pos = neg = 0
for l in lbl.glob("*.txt"):
    if l.read_text(encoding="utf-8").strip():
        pos += 1
    else:
        neg += 1
total = pos + neg
print(f"Final: {total} images")
print(f"  Positive: {pos} ({pos/total*100:.1f}%)")
print(f"  Negative: {neg} ({neg/total*100:.1f}%)")