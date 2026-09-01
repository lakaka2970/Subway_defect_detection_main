# -*- coding: utf-8 -*-
"""离线复算 LOSO 指标 + 等误报率（iso-FP）对比。

用途：
1. 用本地下载的 predictions.csv + rebuilt 基准标签，完整复现 remote_eval_loso.py 的
   匹配与指标口径，验证已下载 metrics.json 无误。
2. 在统一 conf 网格上扫描，得到 Recall–FP 曲线，从而做 r1 / r2 / 消融A 的
   **等误报率公平对比**（这是只在 conf=0.25 单点比较时看不到的结论）。

匹配口径（与 remote_eval_loso.py 严格一致）：
  - 预测按置信度降序，逐类贪婪匹配，IoU >= 0.5，同类才可匹配，每个 GT 仅匹配一次
  - macro_ap50 只对"有 GT 的类"取均值
  - fp_per_image = 未匹配上的预测数 / 图片数
"""
import csv
import json
import os
from collections import defaultdict

BASE = r"E:\Work\Subway_defect_detection_main\docs\plans\9.01阶段2产物"
BENCH = os.path.join(BASE, "bench")
PRED = os.path.join(BASE, "predictions")
MET = os.path.join(BASE, "metrics")

NAMES = ['VHBNM', 'VHBNL', 'SVHBNM', 'SVHBNL', 'SVHTNL', 'CBHPM', 'CBVPM',
         'RHTBNM', 'RHTBNL', 'GWCSBNM', 'GWCSBNL', 'GWCNM', 'GWCNL', 'BSBM',
         'INSD', 'DRPS']
N2I = {n: i for i, n in enumerate(NAMES)}
MATCH_IOU = 0.5
IMW = IMH = 5120

# 运行名 -> 预测文件名
RUNS = [
    ("stage4(基线)", "pred_eval_stage4_predictions.csv", None),
    ("r1 f0", "pred_eval_dgv2a_f0_predictions.csv", 0),
    ("r1 f1", "pred_eval_dgv2a_f1_predictions.csv", 1),
    ("r1 f2", "pred_eval_dgv2a_f2_predictions.csv", 2),
    ("r1 f3", "pred_eval_dgv2a_f3_predictions.csv", 3),
    ("r2 f0", "pred_eval_dgv2a_r2_f0_predictions.csv", 0),
    ("r2 f1", "pred_eval_dgv2a_r2_f1_predictions.csv", 1),
    ("r2 f2", "pred_eval_dgv2a_r2_f2_predictions.csv", 2),
    ("r2 f3", "pred_eval_dgv2a_r2_f3_predictions.csv", 3),
    ("abA f0", "pred_eval_abA_f0_predictions.csv", 0),
    ("abA f2", "pred_eval_abA_f2_predictions.csv", 2),
]


def load_gt():
    gts, sizes = {}, {}
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
        gts[name] = boxes
        sizes[name] = (IMW, IMH)
    return gts, sizes


FOLD_OF = {}
for r in csv.DictReader(open(os.path.join(BENCH, "fold_assignments.csv"),
                             encoding="utf-8-sig")):
    FOLD_OF[r["image"]] = int(r["fold"])


def load_pred(fn, fold):
    """返回 {image: [(cls, conf, x1,y1,x2,y2), ...]}，按 conf 降序。"""
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


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = ix1 - ix0, iy1 - iy0
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) +
                    (b[2] - b[0]) * (b[3] - b[1]) - inter)


def ap50(tp_flags, n_gt):
    if n_gt == 0:
        return float("nan")
    tp = [0]
    s = 0
    for v in tp_flags:
        s += v
        tp.append(s)
    tp = tp[1:]
    fp = [i + 1 - tp[i] for i in range(len(tp))]
    rec = [tp[i] / n_gt for i in range(len(tp))]
    prec = [tp[i] / (tp[i] + fp[i]) for i in range(len(tp))]
    mrec = [0.0] + rec + [1.0]
    mpre = [0.0] + prec + [0.0]
    # 反向累积最大值
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]
               for i in range(len(mrec) - 1))


def evaluate(gts, preds, imgs, conf):
    per_cls = defaultdict(lambda: {"tp": [], "ngt": 0})
    fp_op = 0
    gt0_fp = [0, 0]
    for name in imgs:
        gt = gts.get(name, [])
        pr = [p for p in preds.get(name, []) if p[1] >= conf]
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
        for g in gt:
            per_cls[g[0]]["ngt"] += 1
        if not gt:
            gt0_fp[1] += 1
            if pr:
                gt0_fp[0] += 1
    aps = {c: ap50(d["tp"], d["ngt"]) for c, d in per_cls.items()}
    gt_classes = sorted({g[0] for n in imgs for g in gts.get(n, [])})
    vals = [aps[c] for c in gt_classes
            if c in aps and aps[c] == aps[c]]  # 非 nan
    tot_gt = sum(len(gts.get(n, [])) for n in imgs)
    tot_tp = sum(1 for d in per_cls.values() for t in d["tp"] if t)
    return {
        "macro_ap50": sum(vals) / len(vals) if vals else 0.0,
        "recall": tot_tp / tot_gt if tot_gt else 0.0,
        "fp_per_image": fp_op / len(imgs) if imgs else 0.0,
        "gt0_fp_rate": gt0_fp[0] / gt0_fp[1] if gt0_fp[1] else 0.0,
        "tp": tot_tp, "fp": fp_op, "fn": tot_gt - tot_tp,
        "n_images": len(imgs),
        "per_class_ap50": {NAMES[c]: (None if v != v else round(v, 4))
                           for c, v in sorted(aps.items())},
        "per_class_ngt": {NAMES[c]: per_cls[c]["ngt"]
                          for c in sorted(per_cls) if per_cls[c]["ngt"]},
    }


def main():
    gts, _ = load_gt()
    imgs_all = sorted(gts.keys())
    print("基准：%d 张，GT 类别 = %s" % (len(imgs_all), "全部(基线时)"))

    print("\n" + "=" * 100)
    print("一、复算校验：conf=0.25 与远端 metrics.json 对照")
    print("=" * 100)
    print("%-14s %10s %10s %10s %10s %10s" % (
        "运行", "宏AP50", "Recall", "每图FP", "TP/FP/FN", "图数"))
    print("-" * 100)

    store = {}
    for label, fn, fold in RUNS:
        preds = load_pred(fn, fold)
        imgs = [n for n in imgs_all
                if fold is None or FOLD_OF.get(n) == fold]
        m = evaluate(gts, preds, imgs, 0.25)
        store[label] = (preds, imgs)
        print("%-14s %10.4f %10.3f %10.3f %5d/%5d/%5d %10d" % (
            label, m["macro_ap50"], m["recall"], m["fp_per_image"],
            m["tp"], m["fp"], m["fn"], m["n_images"]))

    # ---- 等误报率对比 ----
    print("\n" + "=" * 100)
    print("二、等误报率（iso-FP）对比：r1 vs r2，四折合并（534 张 / 376 GT）")
    print("=" * 100)
    grid = [0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25,
            0.30, 0.40, 0.50]

    def pooled(run_labels, conf):
        tp = fp = fn = 0
        aps = []
        ngt_total = 0
        for lb in run_labels:
            preds, imgs = store[lb]
            m = evaluate(gts, preds, imgs, conf)
            tp += m["tp"]; fp += m["fp"]; fn += m["fn"]
            aps.append(m["macro_ap50"])
            ngt_total += m["tp"] + m["fn"]
        return tp, fp, fn, sum(aps) / len(aps), ngt_total

    r1 = ["r1 f0", "r1 f1", "r1 f2", "r1 f3"]
    r2 = ["r2 f0", "r2 f1", "r2 f2", "r2 f3"]
    print("%8s | %-24s | %-24s" % ("conf", "r1 (R / FP每图 / 宏AP)",
                                   "r2 (R / FP每图 / 宏AP)"))
    print("-" * 100)
    curve1, curve2 = [], []
    for c in grid:
        t1, f1_, n1, a1, g1 = pooled(r1, c)
        t2, f2_, n2, a2, g2 = pooled(r2, c)
        r_1 = t1 / g1 if g1 else 0
        r_2 = t2 / g2 if g2 else 0
        curve1.append((c, r_1, f1_ / 534, a1))
        curve2.append((c, r_2, f2_ / 534, a2))
        print("%8.2f | R=%.3f FP=%.2f AP=%.4f  | R=%.3f FP=%.2f AP=%.4f" % (
            c, r_1, f1_ / 534, a1, r_2, f2_ / 534, a2))

    print("\n--- 在若干 FP 预算下，各自能达到的最高 Recall（插值上界）---")
    print("%12s | %-16s | %-16s | %s" % ("FP每图预算", "r1 最高R", "r2 最高R",
                                         "优势方"))
    for budget in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
        b1 = max([r for _, r, f, _ in curve1 if f <= budget] or [0])
        b2 = max([r for _, r, f, _ in curve2 if f <= budget] or [0])
        win = "r1" if b1 > b2 + 1e-9 else ("r2" if b2 > b1 + 1e-9 else "持平")
        print("%12.2f | %16.3f | %16.3f | %s" % (budget, b1, b2, win))

    # ---- 消融 A 同折对比（等 FP）----
    print("\n" + "=" * 100)
    print("三、消融A（低照度变体）等误报率对比 vs 同折 r1 基线")
    print("=" * 100)
    for fold, base_lb, a_lb in [(0, "r1 f0", "abA f0"), (2, "r1 f2", "abA f2")]:
        print("[fold%d]" % fold)
        print("  %8s | %-22s | %-22s" % ("conf", "r1 基线 (R/FP)", "消融A (R/FP)"))
        pb, ib = store[base_lb]
        pa, ia = store[a_lb]
        for c in [0.07, 0.10, 0.15, 0.20, 0.25, 0.30]:
            mb = evaluate(gts, pb, ib, c)
            ma = evaluate(gts, pa, ia, c)
            print("  %8.2f | R=%.3f FP=%.2f        | R=%.3f FP=%.2f" % (
                c, mb["recall"], mb["fp_per_image"],
                ma["recall"], ma["fp_per_image"]))

    print("\nRESCORE_DONE")


if __name__ == "__main__":
    main()
