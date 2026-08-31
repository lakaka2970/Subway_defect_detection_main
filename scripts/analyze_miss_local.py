# -*- coding: utf-8 -*-
"""四折漏检分解（本地运行，复用已回传的 predictions）。
用法：python analyze_miss_local.py <bench_dir> <fold_csv> <predictions.csv> <fold>
"""
import csv
import json
import os
import sys
from collections import Counter

NAMES16 = ["VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
           "RHTBNM", "RHTBNL", "GWCSBNM", "GWCSBNL", "GWCNM", "GWCNL",
           "BSBM", "INSD", "DRPS"]


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) +
                    (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main():
    bench, foldcsv, pred, fold = sys.argv[1], sys.argv[2], sys.argv[3], \
        int(sys.argv[4])
    from PIL import Image
    fold_of = {r["image"]: int(r["fold"]) for r in
               csv.DictReader(open(foldcsv, encoding="utf-8-sig"))}
    names = [n for n, f in fold_of.items() if f == fold]
    gts = {}
    for name in names:
        W, H = Image.open(os.path.join(bench, "images", name)).size
        boxes = []
        lp = os.path.join(bench, "labels", os.path.splitext(name)[0] + ".txt")
        if os.path.exists(lp):
            for ln in open(lp, encoding="utf-8"):
                p = ln.split()
                if len(p) == 5:
                    c, cx, cy, w, h = int(p[0]), *map(float, p[1:])
                    boxes.append((c, [(cx - w / 2) * W, (cy - h / 2) * H,
                                      (cx + w / 2) * W, (cy + h / 2) * H]))
        gts[name] = boxes
    preds = {}
    for r in csv.DictReader(open(pred, encoding="utf-8-sig")):
        preds.setdefault(r["image"], []).append(
            (int(r["class_id"]), float(r["confidence"]),
             [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
    cat = Counter()
    by_cls = {}
    for name in names:
        pl = preds.get(name, [])
        for c, box in gts[name]:
            same = [cf for pc, cf, pb in pl if pc == c and iou(box, pb) >= 0.5]
            anyc = [cf for pc, cf, pb in pl if pc == c]
            mx = max(same) if same else 0.0
            if mx >= 0.25:
                k = "TP(conf>=.25)"
            elif max(anyc or [0]) >= 0.25:
                k = "C定位差"
            elif mx >= 0.05:
                k = "B信心不足"
            elif max(anyc or [0]) >= 0.05:
                k = "C2同类低conf且IoU<0.5"
            else:
                k = "A看不见"
            cat[k] += 1
            by_cls.setdefault(NAMES16[c], Counter())[k] += 1
    print("fold%d 漏检分解: %s" % (fold, json.dumps(dict(cat),
                                                     ensure_ascii=False)))
    for c, d in sorted(by_cls.items()):
        print("  %-6s %s" % (c, json.dumps(dict(d), ensure_ascii=False)))


if __name__ == "__main__":
    main()
