#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
独立复算 第二阶段（2026-09-01）：
  F1. 按折子集修正每折指标（上一版误用全部 534 图作分母）
  F2. 配对 bootstrap：r1 vs r2 的召回差异是否显著？（决定 r2 该不该采纳）
  F3. 单一全局阈值 vs 逐折标定阈值：差多少？（决定能否用一个数落地）
  F4. 折的统计功效：fold3 只有 58 张图，它的"失败"有多可信？
  F5. fold3 几何：F1B02(5张) + F1B05(53张)，拆开看各是多少
"""
import csv, json, random
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

# ---------- GT
gt, fold_of, seg_of_map = {}, {}, {}
for f in sorted((BENCH / "labels").glob("*.txt")):
    stem = f.stem
    img = None
    for ext in (".jpg", ".png", ".jpeg"):
        p = BENCH / "images" / (stem + ext)
        if p.exists():
            img = p; break
    if img is None:
        continue
    with Image.open(img) as im:
        W, H = im.size
    boxes = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        p = line.split()
        if len(p) < 5: continue
        c = int(float(p[0])); cx, cy, bw, bh = map(float, p[1:5])
        boxes.append((c, (cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H))
    gt[stem] = boxes
    seg_of_map[stem] = next((x.split("-")[0] for x in stem.split("_") if x.startswith("F1") and "-" in x), "?")

with (BENCH / "fold_assignments.csv").open(encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
fk = [k for k in rows[0] if "fold" in k.lower()][0]
ik = [k for k in rows[0] if k.lower() in ("image","stem","name","file","img")][0]
for r in rows:
    fold_of[Path(r[ik]).stem] = str(r[fk])

def iou(a, b):
    ix1, iy1 = max(a[0],b[0]), max(a[1],b[1]); ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
    iw, ih = max(0.0, ix2-ix1), max(0.0, iy2-iy1); inter = iw*ih
    if inter <= 0: return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0

def load_preds(files, subset=None):
    preds = defaultdict(list)
    for fn in files:
        with (PRED / fn).open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                s = Path(r["image"]).stem
                if subset is not None and s not in subset: continue
                preds[s].append((int(r["class_id"]), float(r["confidence"]),
                                 float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"])))
    return preds

def match(preds, stems, conf):
    """返回 {stem: (n_gt, n_tp, n_fp)}"""
    out = {}
    for s in stems:
        g = gt.get(s, [])
        used = [False]*len(g); tp = fp = 0
        for (c, cf, x1, y1, x2, y2) in sorted(preds.get(s, []), key=lambda x: -x[1]):
            if cf < conf: continue
            best, bi = 0.0, -1
            for i, gb in enumerate(g):
                if used[i] or gb[0] != c: continue
                v = iou((x1,y1,x2,y2), (gb[1],gb[2],gb[3],gb[4]))
                if v > best: best, bi = v, i
            if best >= 0.5:
                used[bi] = True; tp += 1
            else:
                fp += 1
        out[s] = (len(g), tp, fp)
    return out

FOLD_STEMS = defaultdict(set)
for s, f in fold_of.items():
    FOLD_STEMS[str(f)].add(s)
ALL = set(gt)

R1 = ["pred_eval_dgv2a_f0_predictions.csv","pred_eval_dgv2a_f1_predictions.csv",
      "pred_eval_dgv2a_f2_predictions.csv","pred_eval_dgv2a_f3_predictions.csv"]
R2 = ["pred_eval_dgv2a_r2_f0_predictions.csv","pred_eval_dgv2a_r2_f1_predictions.csv",
      "pred_eval_dgv2a_r2_f2_predictions.csv","pred_eval_dgv2a_r2_f3_predictions.csv"]
S4 = ["pred_eval_stage4_predictions.csv"]

# ================= F1 按折子集修正
print("=" * 100)
print("F1. 按折子集修正后的每折指标（报告 §13.1 复核）")
print("=" * 100)
print(f"{'运行':<10}{'折':<5}{'图':>5}{'GT':>5}{'Recall':>9}{'FP/图':>8}{'GT0误报':>9}{'TP/FP/FN':>14}")
print("-" * 100)
fold_metrics = {}
for tag, files in [("r1", R1), ("r2", R2)]:
    for k in sorted(FOLD_STEMS):
        sub = FOLD_STEMS[k]
        preds = load_preds([files[int(k)]], sub)
        m = match(preds, sub, 0.25)
        ng = sum(v[0] for v in m.values()); tp = sum(v[1] for v in m.values()); fp = sum(v[2] for v in m.values())
        gt0 = [s for s in sub if len(gt.get(s, [])) == 0]
        gt0f = sum(1 for s in gt0 if m[s][2] > 0)
        fold_metrics[f"{tag}_f{k}"] = {"n": len(sub), "gt": ng, "tp": tp, "fp": fp,
                                        "recall": tp/ng if ng else 0, "fpi": fp/len(sub),
                                        "gt0fpr": gt0f/len(gt0) if gt0 else 0}
        print(f"{tag:<10}{k:<5}{len(sub):>5}{ng:>5}{tp/ng:>9.3f}{fp/len(sub):>8.3f}"
              f"{gt0f/len(gt0) if gt0 else 0:>9.3f}   {tp}/{fp}/{ng-tp}")
preds = load_preds(S4, ALL); m = match(preds, ALL, 0.25)
tp = sum(v[1] for v in m.values()); fp = sum(v[2] for v in m.values())
gt0 = [s for s in ALL if len(gt.get(s, [])) == 0]; gt0f = sum(1 for s in gt0 if m[s][2] > 0)
fold_metrics["stage4"] = {"n": len(ALL), "gt": 376, "tp": tp, "fp": fp, "recall": tp/376,
                          "fpi": fp/len(ALL), "gt0fpr": gt0f/len(gt0)}
print(f"{'stage4':<10}{'全部':<5}{len(ALL):>5}{376:>5}{tp/376:>9.3f}{fp/len(ALL):>8.3f}{gt0f/len(gt0):>9.3f}   {tp}/{fp}/{376-tp}")

# ================= F2 配对 bootstrap
print("\n" + "=" * 100)
print("F2. 配对 bootstrap（1000 次，按图像重采样）：r2 − r1 的召回差是否显著？")
print("=" * 100)
p1 = load_preds(R1, ALL); p2 = load_preds(R2, ALL)
CONFS = [0.10, 0.15, 0.25]
m1 = {c: match(p1, ALL, c) for c in CONFS}
m2 = {c: match(p2, ALL, c) for c in CONFS}
random.seed(42)
imgs = sorted(ALL)
boot = {}
for c in CONFS:
    diffs, r1s, r2s = [], [], []
    for _ in range(1000):
        samp = [random.choice(imgs) for _ in imgs]
        a = sum(m1[c][s][1] for s in samp); b = sum(m2[c][s][1] for s in samp)
        n = sum(m1[c][s][0] for s in samp)
        if n == 0: continue
        r1s.append(a/n); r2s.append(b/n); diffs.append(b/n - a/n)
    diffs.sort()
    boot[str(c)] = {
        "r1_recall": sum(r1s)/len(r1s), "r2_recall": sum(r2s)/len(r2s),
        "mean_diff": sum(diffs)/len(diffs),
        "ci95": [diffs[int(0.025*len(diffs))], diffs[int(0.975*len(diffs))]],
        "p_r2_better": sum(1 for d in diffs if d > 0)/len(diffs),
    }
    b = boot[str(c)]
    sig = "显著" if b["ci95"][0] > 0 else ("显著为负" if b["ci95"][1] < 0 else "不显著")
    print(f"  conf={c:<5} r1={b['r1_recall']:.4f}  r2={b['r2_recall']:.4f}  "
          f"Δ={b['mean_diff']:+.4f}  95%CI=[{b['ci95'][0]:+.4f}, {b['ci95'][1]:+.4f}]  "
          f"P(r2更好)={b['p_r2_better']:.3f}  → {sig}")

# ================= F3 全局阈值 vs 逐折标定
print("\n" + "=" * 100)
print("F3. 单一全局阈值 vs 逐折标定阈值（校准管线是否可简化为一个数）")
print("=" * 100)
report_tstar = {"r1": {"0": 0.10, "1": 0.10, "2": 0.10, "3": 0.07},
                "r2": {"0": 0.15, "1": 0.10, "2": 0.10, "3": 0.07}}
print(f"{'轮次':<6}{'折':<4}{'逐折t*  R / FP每图':<24}{'全局t=0.10  R / FP每图':<26}{'差值'}")
print("-" * 100)
gt_fixed = {}
for tag, files in [("r1", R1), ("r2", R2)]:
    tot_tstar = tot_fixed = tot_fp_t = tot_fp_f = tot_gt = 0
    for k in sorted(FOLD_STEMS):
        sub = FOLD_STEMS[k]
        pr = load_preds([files[int(k)]], sub)
        ts = report_tstar[tag][k]
        mt = match(pr, sub, ts); mf = match(pr, sub, 0.10)
        rt = sum(v[1] for v in mt.values()); ft = sum(v[2] for v in mt.values())
        rf = sum(v[1] for v in mf.values()); ff = sum(v[2] for v in mf.values())
        ng = sum(v[0] for v in mt.values())
        tot_tstar += rt; tot_fixed += rf; tot_fp_t += ft; tot_fp_f += ff; tot_gt += ng
        print(f"{tag:<6}{k:<4}{rt/ng:>8.3f} / {ft/len(sub):>6.2f}        "
              f"{rf/ng:>10.3f} / {ff/len(sub):>6.2f}        {rf/ng-rt/ng:+.3f}")
    print(f"{tag:<6}{'合并':<4}{tot_tstar/tot_gt:>8.3f} / {tot_fp_t/534:>6.2f}        "
          f"{tot_fixed/tot_gt:>10.3f} / {tot_fp_f/534:>6.2f}        {tot_fixed/tot_gt-tot_tstar/tot_gt:+.3f}")
    print("-" * 100)

# ================= F4 折的统计功效
print("\n" + "=" * 100)
print("F4. 各折统计功效（r2，conf=0.25，图像级 bootstrap 95%CI）")
print("=" * 100)
for k in sorted(FOLD_STEMS):
    sub = sorted(FOLD_STEMS[k])
    pr = load_preds([R2[int(k)]], set(sub))
    mm = match(pr, set(sub), 0.25)
    rnd = []
    for _ in range(1000):
        samp = [random.choice(sub) for _ in sub]
        a = sum(mm[s][1] for s in samp); n = sum(mm[s][0] for s in samp)
        if n: rnd.append(a/n)
    rnd.sort()
    ng = sum(mm[s][0] for s in sub); tp = sum(mm[s][1] for s in sub)
    print(f"  fold{k}: {len(sub):>3} 图 / {ng:>3} GT | Recall={tp/ng:.3f} "
          f"95%CI=[{rnd[int(0.025*len(rnd))]:.3f}, {rnd[int(0.975*len(rnd))]:.3f}] "
          f"区间宽度={rnd[int(0.975*len(rnd))]-rnd[int(0.025*len(rnd))]:.3f}")

# ================= F5 fold3 拆开看
print("\n" + "=" * 100)
print("F5. fold3 拆解：F1B02(5图) vs F1B05(53图) 分别是什么水平")
print("=" * 100)
sub3 = FOLD_STEMS["3"]
by_seg = defaultdict(list)
for s in sub3:
    by_seg[seg_of_map[s]].append(s)
for tag, files in [("r1", R1), ("r2", R2)]:
    pr = load_preds([files[3]], set(sub3))
    for conf in [0.10, 0.25]:
        mm = match(pr, set(sub3), conf)
        for sg in sorted(by_seg):
            ss = by_seg[sg]
            ng = sum(mm[s][0] for s in ss); tp = sum(mm[s][1] for s in ss); fp = sum(mm[s][2] for s in ss)
            print(f"  {tag} conf={conf}: {sg} {len(ss):>2}图 GT={ng:>2} TP={tp:>2} FP={fp:>3} "
                  f"Recall={tp/ng if ng else 0:.3f} FP/图={fp/len(ss):.2f}")

(OUT / "verify_significance.json").write_text(json.dumps(
    {"fold_metrics": fold_metrics, "bootstrap_r1_vs_r2": boot}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n已保存: {OUT/'verify_significance.json'}")
