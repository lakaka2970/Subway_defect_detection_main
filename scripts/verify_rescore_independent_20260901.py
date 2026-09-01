#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立复算（2026-09-01）：不复用 scripts/rescore_loso_offline_20260901.py，
从 9.01阶段2产物/predictions/*.csv + data/Defect_dataset_16_rebuilt/labels 重算全部 LOSO 指标。
用于独立验证 8.31/9.01 报告 §13 的核心数字。
"""
import csv, json
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image

ROOT = Path(r"E:\Work\Subway_defect_detection_main")
BENCH = ROOT / "data" / "Defect_dataset_16_rebuilt"
PRED = ROOT / "docs" / "plans" / "9.01阶段2产物" / "predictions"
OUT = ROOT / "docs" / "plans" / "9.01全量数据盘点"
OUT.mkdir(parents=True, exist_ok=True)

CLS16 = ["VHBNM","VHBNL","SVHBNM","SVHBNL","SVHTNL","CBHPM","CBVPM","RHTBNM",
         "RHTBNL","GWCSBNM","GWCSBNL","GWCNM","GWCNL","BSBM","INSD","DRPS"]

# ---------- 载入 GT
gt = {}          # stem -> [(cls, x1,y1,x2,y2)]
sizes = {}
for f in sorted((BENCH / "labels").glob("*.txt")):
    stem = f.stem
    img = None
    for ext in (".jpg", ".png", ".jpeg"):
        p = BENCH / "images" / (stem + ext)
        if p.exists():
            img = p
            break
    if img is None:
        continue
    with Image.open(img) as im:
        W, H = im.size
    sizes[stem] = (W, H)
    boxes = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        p = line.split()
        if len(p) < 5:
            continue
        c = int(float(p[0])); cx, cy, bw, bh = map(float, p[1:5])
        boxes.append((c, (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H))
    gt[stem] = boxes

# ---------- 折分配
fold_of = {}
with (BENCH / "fold_assignments.csv").open(encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
fk = [k for k in rows[0] if "fold" in k.lower()][0]
ik = [k for k in rows[0] if k.lower() in ("image", "stem", "name", "file", "img")][0]
for r in rows:
    fold_of[Path(r[ik]).stem] = str(r[fk])

def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0

def ap50(pairs, n_gt):
    """pairs: [(conf, is_tp)] 已按 conf 降序；all-point 插值"""
    if n_gt == 0:
        return None
    pairs = sorted(pairs, key=lambda x: -x[0])
    tp = fp = 0
    pr, rec = [], []
    for _, t in pairs:
        tp += 1 if t else 0
        fp += 0 if t else 1
        pr.append(tp / (tp + fp))
        rec.append(tp / n_gt)
    # all-point 插值
    ap, prev = 0.0, 0.0
    for r in [i / 100 for i in range(101)]:
        p = max([pr[i] for i in range(len(rec)) if rec[i] >= r], default=0.0)
        ap += p * (r - prev)
        prev = r
    return ap

def evaluate(pred_files, image_subset=None, conf_thr=0.25):
    """pred_files: list of csv paths; image_subset: set of stems or None(全部)"""
    preds = defaultdict(list)
    for pf in pred_files:
        with pf.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                stem = Path(r["image"]).stem
                if image_subset is not None and stem not in image_subset:
                    continue
                preds[stem].append((int(r["class_id"]), float(r["confidence"]),
                                    float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])))
    imgs = image_subset if image_subset is not None else set(gt)
    per_cls_pairs = defaultdict(list)
    n_gt_cls = Counter()
    tp_tot = fp_tot = 0
    gt0_imgs = gt0_with_fp = 0
    for stem in imgs:
        g = gt.get(stem, [])
        p = sorted(preds.get(stem, []), key=lambda x: -x[1])
        used = [False] * len(g)
        n_fp_img = 0
        for (c, conf, x1, y1, x2, y2) in p:
            if conf < conf_thr:
                continue
            best, bi = 0.0, -1
            for i, gb in enumerate(g):
                if used[i] or gb[0] != c:
                    continue
                v = iou((x1, y1, x2, y2), (gb[1], gb[2], gb[3], gb[4]))
                if v > best:
                    best, bi = v, i
            if best >= 0.5:
                used[bi] = True
                tp_tot += 1
                per_cls_pairs[c].append((conf, True))
            else:
                fp_tot += 1
                n_fp_img += 1
                per_cls_pairs[c].append((conf, False))
        for gb in g:
            n_gt_cls[gb[0]] += 1
        for i, gb in enumerate(g):
            if not used[i]:
                per_cls_pairs[gb[0]].append((-1.0, False))  # FN 占位，不参与 AP（用 n_gt 归一）
        if len(g) == 0:
            gt0_imgs += 1
            if n_fp_img > 0:
                gt0_with_fp += 1
    # 剔除 FN 占位后算 AP
    aps = {}
    for c in sorted(n_gt_cls):
        pairs = [(cf, t) for (cf, t) in per_cls_pairs[c] if cf >= 0]
        aps[c] = ap50(pairs, n_gt_cls[c])
    macro = sum(aps.values()) / len(aps) if aps else 0.0
    n_gt_total = sum(n_gt_cls.values())
    return {
        "n_images": len(imgs),
        "n_gt": n_gt_total,
        "gt_classes": sorted(n_gt_cls),
        "per_class_ap50": {f"{c} {CLS16[c]}": round(aps[c], 4) for c in sorted(aps)},
        "macro_ap50": round(macro, 4),
        "recall": round(tp_tot / n_gt_total, 4) if n_gt_total else 0.0,
        "fp_per_image": round(fp_tot / len(imgs), 4) if imgs else 0.0,
        "gt0_fpr": round(gt0_with_fp / gt0_imgs, 4) if gt0_imgs else 0.0,
        "tp": tp_tot, "fp": fp_tot, "fn": n_gt_total - tp_tot,
    }

# ---------- 运行清单
runs = {
    "stage4_baseline": ["pred_eval_stage4_predictions.csv"],
    "r1_f0": ["pred_eval_dgv2a_f0_predictions.csv"],
    "r1_f1": ["pred_eval_dgv2a_f1_predictions.csv"],
    "r1_f2": ["pred_eval_dgv2a_f2_predictions.csv"],
    "r1_f3": ["pred_eval_dgv2a_f3_predictions.csv"],
    "r2_f0": ["pred_eval_dgv2a_r2_f0_predictions.csv"],
    "r2_f1": ["pred_eval_dgv2a_r2_f1_predictions.csv"],
    "r2_f2": ["pred_eval_dgv2a_r2_f2_predictions.csv"],
    "r2_f3": ["pred_eval_dgv2a_r2_f3_predictions.csv"],
    "abA_f0": ["pred_eval_abA_f0_predictions.csv"],
    "abA_f2": ["pred_eval_abA_f2_predictions.csv"],
}
fold_imgs = defaultdict(set)
for s, f in fold_of.items():
    fold_imgs[str(f)].add(s)

CONF = 0.25
res = {}
print("=" * 92)
print(f"独立复算 conf={CONF}（全类计 FP，逐类贪心匹配 IoU>=0.5）")
print("=" * 92)
print(f"{'运行':<18}{'图':>5}{'GT':>5}{'宏AP50':>9}{'Recall':>9}{'FP/图':>8}{'GT0误报':>9}{'TP/FP/FN':>16}")
print("-" * 92)
for name, files in runs.items():
    paths = [PRED / f for f in files]
    if not all(p.exists() for p in paths):
        print(f"{name:<18} 缺文件")
        continue
    m = evaluate(paths, conf_thr=CONF)
    res[name] = m
    print(f"{name:<18}{m['n_images']:>5}{m['n_gt']:>5}{m['macro_ap50']:>9.4f}{m['recall']:>9.3f}"
          f"{m['fp_per_image']:>8.3f}{m['gt0_fpr']:>9.3f}   {m['tp']}/{m['fp']}/{m['fn']}")

# ---------- 合并四折 + conf 扫描
print("\n" + "=" * 92)
print("四折合并 conf 扫描（独立复算）")
print("=" * 92)
merged = {"r1": ["pred_eval_dgv2a_f0_predictions.csv","pred_eval_dgv2a_f1_predictions.csv",
                 "pred_eval_dgv2a_f2_predictions.csv","pred_eval_dgv2a_f3_predictions.csv"],
          "r2": ["pred_eval_dgv2a_r2_f0_predictions.csv","pred_eval_dgv2a_r2_f1_predictions.csv",
                 "pred_eval_dgv2a_r2_f2_predictions.csv","pred_eval_dgv2a_r2_f3_predictions.csv"]}
grid = [0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
sweep = {}
for tag, files in merged.items():
    paths = [PRED / f for f in files]
    sweep[tag] = {}
    for c in grid:
        sweep[tag][str(c)] = evaluate(paths, conf_thr=c)
print(f"{'conf':>6} | " + " | ".join(f"{t}: R / FP每图 / 宏AP" for t in merged))
print("-" * 92)
for c in grid:
    row = f"{c:>6.2f} | "
    parts = []
    for t in merged:
        m = sweep[t][str(c)]
        parts.append(f"{m['recall']:.3f} / {m['fp_per_image']:.2f} / {m['macro_ap50']:.4f}")
    print(row + " | ".join(parts))

# 等误报率：给预算找最高 Recall
print("\n" + "=" * 92)
print("等误报率对比（给定 FP/图预算，各模型能达到的最高 Recall）")
print("=" * 92)
budgets = [0.25, 0.50, 1.00, 1.50, 2.00, 3.00]
iso = {}
print(f"{'FP预算':>8} | {'r1 Recall':>10}(conf) | {'r2 Recall':>10}(conf) | 优势方")
print("-" * 92)
for b in budgets:
    best = {}
    for t in merged:
        cand = [(m["recall"], float(c)) for c, m in sweep[t].items() if m["fp_per_image"] <= b]
        best[t] = max(cand) if cand else (0.0, None)
    win = "r2" if best["r2"][0] > best["r1"][0] else ("r1" if best["r1"][0] > best["r2"][0] else "平")
    iso[str(b)] = {"r1": best["r1"], "r2": best["r2"], "winner": win}
    print(f"{b:>8.2f} | {best['r1'][0]:>10.3f}({best['r1'][1]}) | {best['r2'][0]:>10.3f}({best['r2'][1]}) | {win}")

# 逐类召回
print("\n" + "=" * 92)
print("逐类 TP/GT（合并四折）")
print("=" * 92)
perclass = {}
for t in merged:
    paths = [PRED / f for f in merged[t]]
    for c in [0.10, 0.25]:
        # 复用 evaluate 但需逐类 TP：这里单独算
        preds = defaultdict(list)
        for pf in paths:
            with pf.open(encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    preds[Path(r["image"]).stem].append(
                        (int(r["class_id"]), float(r["confidence"]),
                         float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])))
        n_gt = Counter(); tp = Counter()
        for stem, g in gt.items():
            for gb in g:
                n_gt[gb[0]] += 1
            used = [False] * len(g)
            for (pc, conf, x1, y1, x2, y2) in sorted(preds.get(stem, []), key=lambda x: -x[1]):
                if conf < c:
                    continue
                best, bi = 0.0, -1
                for i, gb in enumerate(g):
                    if used[i] or gb[0] != pc:
                        continue
                    v = iou((x1,y1,x2,y2), (gb[1],gb[2],gb[3],gb[4]))
                    if v > best:
                        best, bi = v, i
                if best >= 0.5:
                    used[bi] = True
                    tp[pc] += 1
        perclass[f"{t}@conf{c}"] = {f"{k} {CLS16[k]}": f"{tp[k]}/{n_gt[k]} = {tp[k]/n_gt[k]:.3f}"
                                    for k in sorted(n_gt)}
for k, v in perclass.items():
    print(f"  {k:<14} {v}")

# 误报类别构成
print("\n" + "=" * 92)
print("误报类别构成（合并四折，conf=0.10）")
print("=" * 92)
for t in merged:
    paths = [PRED / f for f in merged[t]]
    preds = defaultdict(list)
    for pf in paths:
        with pf.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                preds[Path(r["image"]).stem].append(
                    (int(r["class_id"]), float(r["confidence"]),
                     float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])))
    fpcls = Counter(); tot = 0
    for stem, g in gt.items():
        used = [False] * len(g)
        for (pc, conf, x1, y1, x2, y2) in sorted(preds.get(stem, []), key=lambda x: -x[1]):
            if conf < 0.10:
                continue
            best, bi = 0.0, -1
            for i, gb in enumerate(g):
                if used[i] or gb[0] != pc:
                    continue
                v = iou((x1,y1,x2,y2), (gb[1],gb[2],gb[3],gb[4]))
                if v > best:
                    best, bi = v, i
            if best >= 0.5:
                used[bi] = True
            else:
                fpcls[pc] += 1; tot += 1
    top = {f"{k} {CLS16[k]}": f"{v} ({v/tot*100:.1f}%)" for k, v in fpcls.most_common(8)}
    print(f"  {t}: 共 {tot} 个误报 | {top}")

# 保存
(OUT / "verify_rescore.json").write_text(
    json.dumps({"per_run": res, "sweep": sweep, "iso_fp": iso, "per_class": perclass},
               ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n已保存: {OUT/'verify_rescore.json'}")
