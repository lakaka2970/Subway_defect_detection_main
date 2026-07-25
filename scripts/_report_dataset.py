#!/usr/bin/env python3
"""Report dataset composition."""
from pathlib import Path
from collections import Counter

d = Path("data/subway_crops/train/images")
files = list(d.glob("*.jpg"))
cats = Counter()
for f in files:
    s = f.stem
    if "_aug" in s: cats["场景增强"] += 1
    elif "_cp" in s: cats["Copy-Paste"] += 1
    elif "_hn" in s: cats["A2-HN"] += 1
    elif "_fn" in s: cats["A2-FN"] += 1
    elif "_rev" in s: cats["A2-Rev"] += 1
    elif "_a2fp" in s: cats["A2-FP"] += 1
    elif "_a2neg" in s: cats["A2-Neg"] += 1
    else: cats["原始"] += 1

print(f"Total images: {len(files)}")
for k, v in sorted(cats.items(), key=lambda x: -x[1]):
    print(f"  {k:12s}: {v:>6} ({v/len(files)*100:5.1f}%)")

ld = Path("data/subway_crops/train/labels")
pos = neg = 0
for l in ld.glob("*.txt"):
    if l.read_text(encoding="utf-8").strip():
        pos += 1
    else:
        neg += 1
total_lbl = pos + neg
print(f"\nPositive: {pos} ({pos/total_lbl*100:.1f}%)")
print(f"Negative: {neg} ({neg/total_lbl*100:.1f}%)")

c = Counter()
for l in ld.glob("*.txt"):
    for line in l.read_text(encoding="utf-8").strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            c[int(parts[0])] += 1
print("\nClass distribution (annotations):")
names = ["VHBNM","VHBNL","SVHBNM","SVHBNL","SVHTNL","CBHPM","CBVPM"]
for k in sorted(c.keys()):
    name = names[k] if k < len(names) else f"class_{k}"
    print(f"  {k} ({name:8s}): {c[k]:>5}")
print(f"  Total annotations: {sum(c.values())}")