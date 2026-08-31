# -*- coding: utf-8 -*-
"""阈值校准管线实验（P0-2，防泄漏）。

每折 k：
  1) 用 fold{k}_best.pt 在其“监控图”（val_fold{k}.txt 中出现过的真实线 tile 的原图，
     这些图从未进梯度）上做与评估协议一致的滑窗推理；
  2) 在监控图上扫描阈值，选 t* = 满足每图FP<=2.0 时 Recall 最大的阈值；
  3) 将 t* 应用到该折测试折的 predictions.csv（协议运营点原为 0.25），
     重算 TP/FP/FN/Recall/每图FP —— 报告“校准后运营点”的真实收益。
输出 out/calib/calib_results.json + 打印汇总。
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import torch
from torchvision.ops import nms

TILE, STRIDE = 1280, 960
MATCH_IOU, NMS_IOU, MAX_DET = 0.5, 0.5, 300
GRID = [0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
FP_BUDGET = 2.0


def offsets(size):
    if size <= TILE:
        return [0]
    out, i = [], 0
    while True:
        o = min(i * STRIDE, size - TILE)
        if not out or o != out[-1]:
            out.append(o)
        if o == size - TILE:
            break
        i += 1
    return out


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) +
                    (b[2] - b[0]) * (b[3] - b[1]) - inter)


def load_gt(bench, names):
    from PIL import Image
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
    return gts


def metrics_at(preds_by_img, gts, names, th):
    tp = fp = ngt = 0
    for name in names:
        gt = gts[name]
        ngt += len(gt)
        used = set()
        for c, cf, box in sorted(
                [p for p in preds_by_img.get(name, []) if p[1] >= th],
                key=lambda x: -x[1]):
            best, bi = 0.0, -1
            for gi, (gc, gb) in enumerate(gt):
                if gi in used or gc != c:
                    continue
                v = iou(box, gb)
                if v > best:
                    best, bi = v, gi
            if best >= MATCH_IOU and bi >= 0:
                used.add(bi)
                tp += 1
            else:
                fp += 1
    return {"tp": tp, "fp": fp, "fn": ngt - tp,
            "recall": tp / max(1, ngt), "fp_per_img": fp / max(1, len(names))}


def slide_infer(model, bench, names):
    preds = defaultdict(list)
    from PIL import Image
    for name in names:
        im = np.asarray(Image.open(
            os.path.join(bench, "images", name)).convert("RGB"))
        H, W = im.shape[:2]
        tasks = []
        for ty in offsets(H):
            for tx in offsets(W):
                tasks.append((tx, ty, im[ty:ty + TILE, tx:tx + TILE]))
        raw = []
        for i in range(0, len(tasks), 8):
            res = model.predict([t[2] for t in tasks[i:i + 8]], conf=0.001,
                                iou=NMS_IOU, max_det=MAX_DET, verbose=False,
                                device=0)
            for (tx, ty, _), r in zip(tasks[i:i + 8], res):
                b = r.boxes
                if b is None or len(b) == 0:
                    continue
                xyxy = b.xyxy.cpu().numpy()
                cls = b.cls.cpu().numpy().astype(int)
                cf = b.conf.cpu().numpy()
                for (x0, y0, x1, y1), c, s in zip(xyxy, cls, cf):
                    raw.append((int(c), float(s),
                                [x0 + tx, y0 + ty, x1 + tx, y1 + ty]))
        # 全图逐类 NMS
        for c in range(16):
            cb = [p for p in raw if p[0] == c]
            if not cb:
                continue
            t = torch.tensor([p[2] for p in cb])
            s = torch.tensor([p[1] for p in cb])
            idx = nms(t, s, NMS_IOU).tolist()
            preds[name] += [cb[i] for i in idx]
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="/root/autodl-tmp/subway/data/rebuilt")
    ap.add_argument("--modeldir", default="/root/autodl-tmp/subway/out/dgv2a")
    ap.add_argument("--idx", default="/root/autodl-tmp/subway/data/tiles/index")
    ap.add_argument("--evalroot", default="/root/autodl-tmp/subway/out")
    ap.add_argument("--out", default="/root/autodl-tmp/subway/out/calib")
    ap.add_argument("--folds", default="0,1,2,3")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from ultralytics import YOLO
    fold_of = {r["image"]: int(r["fold"]) for r in
               csv.DictReader(open(os.path.join(args.bench,
                                                "fold_assignments.csv"),
                                   encoding="utf-8-sig"))}
    results = {}
    for k in [int(x) for x in args.folds.split(",")]:
        # 监控图 = val 清单中真实线 tile 对应的原图
        mon = set()
        for ln in open(os.path.join(args.idx, "val_fold%d.txt" % k),
                       encoding="utf-8"):
            p = ln.strip()
            if "/real/images/" in p:
                stem = os.path.splitext(os.path.basename(p))[0]
                img = stem[2:].rsplit("_t", 1)[0] + ".jpg"
                mon.add(img)
        mon = sorted(mon)
        gts = load_gt(args.bench, mon)
        model = YOLO(os.path.join(args.modeldir, "fold%d_best.pt" % k))
        mp = slide_infer(model, args.bench, mon)
        scan = {("%.2f" % th): metrics_at(mp, gts, mon, th) for th in GRID}
        ok = [(th, m["recall"]) for th, m in
              ((t, metrics_at(mp, gts, mon, t)) for t in GRID)
              if m["fp_per_img"] <= FP_BUDGET]
        tstar = max(ok, key=lambda x: x[1])[0] if ok else 0.25
        # 应用到测试折
        tp_path = os.path.join(args.evalroot, "eval_dgv2a_f%d" % k,
                               "predictions.csv")
        names = [n for n, f in fold_of.items() if f == k]
        tp_by_img = defaultdict(list)
        for r in csv.DictReader(open(tp_path, encoding="utf-8-sig")):
            tp_by_img[r["image"]].append(
                (int(r["class_id"]), float(r["confidence"]),
                 [float(r[c]) for c in ("x1", "y1", "x2", "y2")]))
        tgts = load_gt(args.bench, names)
        base = metrics_at(tp_by_img, tgts, names, 0.25)
        cal = metrics_at(tp_by_img, tgts, names, tstar)
        results["fold%d" % k] = {
            "n_monitor_imgs": len(mon), "t_star": tstar,
            "monitor_scan": scan, "test_at_0.25": base,
            "test_at_t_star": cal}
        print("fold%d: t*=%.2f 测试点 0.25 R=%.3f FP/img=%.2f -> "
              "t* R=%.3f FP/img=%.2f" %
              (k, tstar, base["recall"], base["fp_per_img"],
               cal["recall"], cal["fp_per_img"]))
    with open(os.path.join(args.out, "calib_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("CALIB_DONE")


if __name__ == "__main__":
    main()
