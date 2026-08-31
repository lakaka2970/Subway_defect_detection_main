# -*- coding: utf-8 -*-
"""硬负样本挖掘第 1 轮：消费各折评估的 predictions.csv（conf>=0.001），
在运营阈值 0.25 下提取与任意 GT IoU<0.1 的纯背景误报框，按类聚类，
导出空标签 tile 清单（供第 2 轮训练回灌）与统计报告。
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--preds", nargs="+", required=True,
                    help="各折 predictions.csv（仅含该折测试图）")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    gts = {}
    for name in os.listdir(os.path.join(args.bench, "images")):
        boxes = []
        lp = os.path.join(args.bench, "labels",
                          os.path.splitext(name)[0] + ".txt")
        if os.path.exists(lp):
            for ln in open(lp, encoding="utf-8"):
                f = ln.split()
                if len(f) == 5:
                    boxes.append([float(x) for x in f[1:]])
        gts[name] = boxes

    fp_by_cls = Counter()
    fp_tiles = defaultdict(list)
    n_fp = 0
    for pp in args.preds:
        for r in csv.DictReader(open(pp, encoding="utf-8-sig")):
            if float(r["confidence"]) < args.conf:
                continue
            name = r["image"]
            box = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
            if any(iou(box, g) >= 0.1 for g in gts.get(name, [])):
                continue
            n_fp += 1
            fp_by_cls[r["class_name"]] += 1
            stem = os.path.splitext(name)[0]
            fp_tiles[r["class_name"]].append(
                {"image": name, "box": box, "conf": float(r["confidence"])})

    with open(os.path.join(args.out, "hn_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump({"n_fp": n_fp, "by_class": dict(fp_by_cls.most_common()),
                   "n_images_with_fp": len({t["image"] for v in
                                            fp_tiles.values() for t in v})},
                  f, ensure_ascii=False, indent=1)
    with open(os.path.join(args.out, "hn_boxes.jsonl"), "w",
              encoding="utf-8") as f:
        for c, rows in fp_tiles.items():
            for t in rows:
                f.write(json.dumps({"cls": c, **t}, ensure_ascii=False) + "\n")
    print("硬负样本: %d 个误报框" % n_fp)
    for c, n in fp_by_cls.most_common(8):
        print("  %-8s %d" % (c, n))
    print("MINE_DONE")


if __name__ == "__main__":
    main()
