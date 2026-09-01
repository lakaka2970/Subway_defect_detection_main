# -*- coding: utf-8 -*-
"""逐类召回与误报构成分析（在统一口径下，四折合并）。"""
import csv
import os
from collections import defaultdict, Counter

BASE = r"E:\Work\Subway_defect_detection_main\docs\plans\9.01阶段2产物"
BENCH = os.path.join(BASE, "bench")
PRED = os.path.join(BASE, "predictions")

NAMES = ['VHBNM', 'VHBNL', 'SVHBNM', 'SVHBNL', 'SVHTNL', 'CBHPM', 'CBVPM',
         'RHTBNM', 'RHTBNL', 'GWCSBNM', 'GWCSBNL', 'GWCNM', 'GWCNL', 'BSBM',
         'INSD', 'DRPS']
N2I = {n: i for i, n in enumerate(NAMES)}
IMW = IMH = 5120

FOLD_OF = {}
for r in csv.DictReader(open(os.path.join(BENCH, "fold_assignments.csv"),
                             encoding="utf-8-sig")):
    FOLD_OF[r["image"]] = int(r["fold"])

GT = {}
for f in os.listdir(os.path.join(BENCH, "labels")):
    name = os.path.splitext(f)[0] + ".jpg"
    boxes = []
    for ln in open(os.path.join(BENCH, "labels", f), encoding="utf-8"):
        t = ln.split()
        if len(t) != 5:
            continue
        c = int(t[0]) if t[0].isdigit() else N2I[t[0]]
        cx, cy, w, h = float(t[1]), float(t[2]), float(t[3]), float(t[4])
        boxes.append([c, (cx - w / 2) * IMW, (cy - h / 2) * IMH,
                      (cx + w / 2) * IMW, (cy + h / 2) * IMH])
    GT[name] = boxes


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) +
                    (b[2] - b[0]) * (b[3] - b[1]) - inter)


def load(fn, fold):
    by = defaultdict(list)
    with open(os.path.join(PRED, fn), encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            nm = r["image"]
            if fold is not None and FOLD_OF.get(nm) != fold:
                continue
            by[nm].append((int(r["class_id"]), float(r["confidence"]),
                           float(r["x1"]), float(r["y1"]),
                           float(r["x2"]), float(r["y2"])))
    for k in by:
        by[k].sort(key=lambda a: -a[1])
    return by


GROUPS = {
    "r1": [("pred_eval_dgv2a_f%d_predictions.csv" % k, k) for k in range(4)],
    "r2": [("pred_eval_dgv2a_r2_f%d_predictions.csv" % k) for k in range(4)],
    "r2": [("pred_eval_dgv2a_r2_f%d_predictions.csv" % k, k) for k in range(4)],
}

for tag, files in GROUPS.items():
    merged = defaultdict(list)
    for fn, k in files:
        d = load(fn, k)
        for nm, v in d.items():
            merged[nm].extend(v)
    for nm in merged:
        merged[nm].sort(key=lambda a: -a[1])

    print("=" * 92)
    print("%s 四折合并：逐类召回 / 误报构成" % tag)
    print("=" * 92)
    for conf in [0.25, 0.10]:
        gt_cls = Counter()
        tp_cls = Counter()
        fp_cls = Counter()
        for nm in GT:
            gt = GT[nm]
            pr = [p for p in merged.get(nm, []) if p[1] >= conf]
            matched = set()
            for c, s, x0, y0, x1, y1 in pr:
                best, bi = 0.0, -1
                for gi, g in enumerate(gt):
                    if gi in matched or g[0] != c:
                        continue
                    v = iou((x0, y0, x1, y1), g[1:])
                    if v > best:
                        best, bi = v, gi
                if best >= 0.5 and bi >= 0:
                    matched.add(bi)
                    tp_cls[c] += 1
                else:
                    fp_cls[c] += 1
            for g in gt:
                gt_cls[g[0]] += 1
        print("  [conf=%.2f]" % conf)
        print("   %-8s %6s %6s %8s" % ("类别", "GT", "TP", "Recall"))
        for c in sorted(gt_cls):
            print("   %-8s %6d %6d %8.3f" % (
                NAMES[c], gt_cls[c], tp_cls.get(c, 0),
                tp_cls.get(c, 0) / gt_cls[c]))
        print("   --- 误报(FP)按预测类别构成 ---")
        tot_fp = sum(fp_cls.values())
        for c, v in fp_cls.most_common():
            print("   %-8s %6d  (%.1f%%)" % (NAMES[c], v, 100.0 * v / tot_fp))
        print("   合计 FP = %d" % tot_fp)
        print()

print("PERCLASS_DONE")
