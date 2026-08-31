# -*- coding: utf-8 -*-
"""冻结版 LOSO 评估（阶段1唯一判分尺子）。

协议（写死，改动需另起脚本版本）：
  滑窗 1280x1280 / stride 960；跨 tile 全图逐类 NMS IoU 0.5；max_det 300；
  运营阈值 0.25；匹配 IoU 0.5；低阈值 0.001 全量落盘（供硬负样本挖掘）。
指标：
  每类 AP50、GT 类宏平均 AP50/Recall、每图平均 FP、GT=0 图误报率；
  双口径：仅 GT 类 / 全 16 类（无 GT 类的预测一律计 FP）；
  按折 + 总体；bootstrap 1000 次 95% CI（图像级重采样）。
用法：
  python remote_eval_loso.py --model <pt> --bench <rebuilt_dir> --out <dir> [--limit N]
"""
import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
from torchvision.ops import nms

NAMES16 = ["VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
           "RHTBNM", "RHTBNL", "GWCSBNM", "GWCSBNL", "GWCNM", "GWCNL",
           "BSBM", "INSD", "DRPS"]
TILE, STRIDE = 1280, 960
CONF_SAVE, CONF_OP = 0.001, 0.25
NMS_IOU, MATCH_IOU, MAX_DET = 0.5, 0.5, 300


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)


def ap50(tp_flags, n_gt):
    if n_gt == 0:
        return float("nan")
    tp = np.cumsum(tp_flags)
    fp = np.cumsum(1 - tp_flags)
    rec = tp / n_gt
    prec = tp / (tp + fp)
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-fold", type=int, default=-1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    h0 = sha256(args.model)
    print("权重 SHA256(前):", h0)

    from ultralytics import YOLO
    model = YOLO(args.model)

    images = sorted(os.listdir(os.path.join(args.bench, "images")))
    if args.limit:
        images = images[:args.limit]
    fold_of = {}
    for r in csv.DictReader(open(os.path.join(args.bench,
                                              "fold_assignments.csv"),
                                 encoding="utf-8-sig")):
        fold_of[r["image"]] = int(r["fold"])

    if args.only_fold >= 0:
        images = [n for n in images if fold_of.get(n) == args.only_fold]
        print("仅评估折 %d：%d 张" % (args.only_fold, len(images)))

    # 读 GT
    from PIL import Image
    gts = {}
    sizes = {}
    for name in images:
        im = Image.open(os.path.join(args.bench, "images", name))
        W, H = im.size
        sizes[name] = (W, H)
        boxes = []
        lp = os.path.join(args.bench, "labels",
                          os.path.splitext(name)[0] + ".txt")
        if os.path.exists(lp):
            for ln in open(lp, encoding="utf-8"):
                f = ln.split()
                if len(f) == 5:
                    c, cx, cy, w, h = int(f[0]), float(f[1]), float(f[2]), \
                        float(f[3]), float(f[4])
                    boxes.append([c, (cx - w / 2) * W, (cy - h / 2) * H,
                                  (cx + w / 2) * W, (cy + h / 2) * H])
        gts[name] = boxes

    # 滑窗推理
    tasks = []
    for name in images:
        im = np.asarray(Image.open(
            os.path.join(args.bench, "images", name)).convert("RGB"))
        H, W = im.shape[:2]
        for ty in offsets(H):
            for tx in offsets(W):
                tasks.append((name, tx, ty, im[ty:ty + TILE, tx:tx + TILE]))
    print("tile 总数: %d" % len(tasks))
    preds_raw = defaultdict(list)
    for i in range(0, len(tasks), args.batch):
        batch = tasks[i:i + args.batch]
        res = model.predict([t[3] for t in batch], conf=CONF_SAVE,
                            iou=NMS_IOU, max_det=MAX_DET, verbose=False,
                            device=0)
        for (name, tx, ty, _), r in zip(batch, res):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            cls = b.cls.cpu().numpy().astype(int)
            cf = b.conf.cpu().numpy()
            for (x0, y0, x1, y1), c, s in zip(xyxy, cls, cf):
                preds_raw[name].append(
                    [int(c), float(s), x0 + tx, y0 + ty, x1 + tx, y1 + ty])
        if (i // args.batch) % 50 == 0:
            print("  %d/%d 批" % (i // args.batch,
                                  (len(tasks) + args.batch - 1) // args.batch))

    # 跨 tile 全图逐类 NMS + 落盘
    preds = {}
    rows = []
    for name in images:
        arr = preds_raw.get(name, [])
        kept = []
        for c in range(16):
            cb = [a for a in arr if a[0] == c]
            if not cb:
                continue
            t = torch.tensor([[a[2], a[3], a[4], a[5]] for a in cb])
            s = torch.tensor([a[1] for a in cb])
            idx = nms(t, s, NMS_IOU).tolist()
            kept += [cb[i] for i in idx]
        kept.sort(key=lambda a: -a[1])
        kept = kept[:MAX_DET]
        preds[name] = kept
        for a in kept:
            rows.append([name, a[0], NAMES16[a[0]], a[1]] + a[2:])
    with open(os.path.join(args.out, "predictions.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "class_id", "class_name", "confidence",
                    "x1", "y1", "x2", "y2"])
        w.writerows(rows)

    # 匹配与指标
    def evaluate(img_list):
        per_cls = defaultdict(lambda: {"tp": [], "ngt": 0})
        fp_op, n_op = 0, 0
        gt0_fp = [0, 0]
        for name in img_list:
            gt = gts[name]
            pr = [p for p in preds.get(name, []) if p[1] >= CONF_OP]
            matched = set()
            for c, s, x0, y0, x1, y1 in pr:
                best, bi = 0.0, -1
                for gi, g in enumerate(gt):
                    if gi in matched or g[0] != c:
                        continue
                    v = iou((x0, y0, x1, y1), g[1:])
                    if v > best:
                        best, bi = v, gi
                if best >= MATCH_IOU and bi >= 0:
                    matched.add(bi)
                    per_cls[c]["tp"].append(1)
                else:
                    per_cls[c]["tp"].append(0)
                    fp_op += 1
                n_op += 1
            for g in gt:
                per_cls[g[0]]["ngt"] += 1
            if not gt:
                gt0_fp[1] += 1
                if pr:
                    gt0_fp[0] += 1
        aps = {}
        for c, d in per_cls.items():
            aps[c] = ap50(np.array(d["tp"]), d["ngt"])
        gt_classes = sorted({g[0] for g in sum(gts.values(), [])})
        m = {}
        vals = [aps[c] for c in gt_classes if c in aps and
                not np.isnan(aps[c])]
        m["macro_ap50_gt_classes"] = float(np.mean(vals)) if vals else 0.0
        tot_gt = sum(len(g) for g in [gts[n] for n in img_list])
        tot_tp = sum(1 for d in per_cls.values() for t in d["tp"] if t)
        m["recall_op"] = tot_tp / tot_gt if tot_gt else 0.0
        m["fp_per_image"] = fp_op / len(img_list) if img_list else 0.0
        m["gt0_fp_rate"] = gt0_fp[0] / gt0_fp[1] if gt0_fp[1] else 0.0
        m["per_class_ap50"] = {NAMES16[c]: (round(v, 4) if not np.isnan(v)
                                            else None)
                               for c, v in sorted(aps.items())}
        m["n_images"] = len(img_list)
        m["tp"], m["fp"], m["fn"] = tot_tp, fp_op, tot_gt - tot_tp
        return m

    overall = evaluate(images)
    folds = {}
    for f in sorted(set(fold_of.get(n, 0) for n in images)):
        fl = [n for n in images if fold_of.get(n) == f]
        folds[f] = evaluate(fl)

    # bootstrap CI（总体，宏AP/Recall/每图FP）
    rnd = random.Random(args.seed)
    boots = []
    for _ in range(1000):
        sample = [images[rnd.randrange(len(images))]
                  for _ in range(len(images))]
        m = evaluate(sample)
        boots.append((m["macro_ap50_gt_classes"], m["recall_op"],
                      m["fp_per_image"]))
    boots = np.array(boots)
    ci = {k: [float(np.percentile(boots[:, i], 2.5)),
              float(np.percentile(boots[:, i], 97.5))]
          for i, k in enumerate(("macro_ap50", "recall_op", "fp_per_image"))}

    out = {"protocol": {"tile": TILE, "stride": STRIDE, "nms": NMS_IOU,
                        "match_iou": MATCH_IOU, "max_det": MAX_DET,
                        "conf_op": CONF_OP, "conf_save": CONF_SAVE},
           "weight_sha256": h0,
           "weight_sha256_after": sha256(args.model),
           "overall": overall, "folds": {str(k): v for k, v in folds.items()},
           "bootstrap_ci95": ci}
    with open(os.path.join(args.out, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n=== 总体 ===")
    print("宏AP50(GT类) %.4f  Recall %.4f  每图FP %.2f  GT0误报率 %.2f"
          % (overall["macro_ap50_gt_classes"], overall["recall_op"],
             overall["fp_per_image"], overall["gt0_fp_rate"]))
    print("TP/FP/FN: %d/%d/%d" % (overall["tp"], overall["fp"], overall["fn"]))
    print("CI95:", json.dumps(ci))
    for f in sorted(folds):
        v = folds[f]
        print("折%d: 图%d 宏AP %.4f Recall %.4f 每图FP %.2f"
              % (f, v["n_images"], v["macro_ap50_gt_classes"],
                 v["recall_op"], v["fp_per_image"]))
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
