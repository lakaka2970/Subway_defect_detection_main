# -*- coding: utf-8 -*-
"""阶段2 结果汇总分析：r1 / r2 / 消融A / 阈值校准 / 交付模型 的一致性核对与结论提取。

输出到 stdout，供撰写报告 §13/§14 引用。
"""
import json
import os
import glob

import numpy as np

BASE = r"E:\Work\Subway_defect_detection_main\docs\plans\9.01阶段2产物"
M = os.path.join(BASE, "metrics")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def ov(d):
    return d.get("overall", d)


def fmt(x, n=4):
    return ("%%.%df" % n) % x if isinstance(x, (int, float)) else str(x)


runs = {}
for d in sorted(os.listdir(M)):
    p = os.path.join(M, d, "metrics.json")
    if os.path.isfile(p):
        runs[d] = load(p)

print("=" * 100)
print("一、留一区段（LOSO）评估结果总表")
print("=" * 100)
hdr = "%-16s %8s %8s %9s %10s %10s %6s" % (
    "运行", "宏AP50", "Recall", "每图FP", "GT0误报率", "TP/FP/FN", "图数")
print(hdr)
print("-" * 100)
for k in sorted(runs):
    o = ov(runs[k])
    print("%-16s %8s %8s %9s %10s %10s %6s" % (
        k,
        fmt(o.get("macro_ap50_gt_classes")),
        fmt(o.get("recall_op"), 3),
        fmt(o.get("fp_per_image"), 3),
        fmt(o.get("gt0_fp_rate"), 3),
        "%s/%s/%s" % (o.get("tp"), o.get("fp"), o.get("fn")),
        o.get("n_images")))


def macro(keys):
    a = np.array([ov(runs[k])["macro_ap50_gt_classes"] for k in keys])
    r = np.array([ov(runs[k])["recall_op"] for k in keys])
    f = np.array([ov(runs[k])["fp_per_image"] for k in keys])
    g = np.array([ov(runs[k])["gt0_fp_rate"] for k in keys])
    tp = sum(ov(runs[k])["tp"] for k in keys)
    fp = sum(ov(runs[k])["fp"] for k in keys)
    fn = sum(ov(runs[k])["fn"] for k in keys)
    return a.mean(), r.mean(), f.mean(), g.mean(), tp, fp, fn


print("\n" + "=" * 100)
print("二、四折宏平均对比")
print("=" * 100)
groups = {
    "stage4 基线": [k for k in runs if k.startswith("eval_stage4")],
    "DG-v2a 第1轮": ["eval_dgv2a_f0", "eval_dgv2a_f1", "eval_dgv2a_f2", "eval_dgv2a_f3"],
    "DG-v2a 第2轮": ["eval_dgv2a_r2_f0", "eval_dgv2a_r2_f1", "eval_dgv2a_r2_f2", "eval_dgv2a_r2_f3"],
}
print("%-14s %9s %9s %9s %10s %12s %9s" % (
    "组", "宏AP50", "Recall", "每图FP", "GT0误报率", "合并TP/FP/FN", "合并Recall"))
print("-" * 100)
for name, keys in groups.items():
    if not keys:
        continue
    a, r, f, g, tp, fp, fn = macro(keys)
    print("%-14s %9s %9s %9s %10s %12s %9s" % (
        name, fmt(a), fmt(r, 3), fmt(f, 3), fmt(g, 3),
        "%d/%d/%d" % (tp, fp, fn), fmt(tp / max(1, tp + fn), 3)))

print("\n" + "=" * 100)
print("三、消融 A（追加低照度变体 night/tunnel/degrade）— 同折对照")
print("=" * 100)
print("%-10s %-16s %9s %9s %9s %10s %10s" % (
    "折", "配方", "宏AP50", "Recall", "每图FP", "GT0误报", "TP/FP/FN"))
print("-" * 100)
for k in ("0", "2"):
    base = "eval_dgv2a_f" + k
    a_ = "eval_abA_f" + k
    for tag, key in (("基线条", base), ("+低照度变体", a_)):
        if key not in runs:
            continue
        o = ov(runs[key])
        print("%-10s %-16s %9s %9s %9s %10s %10s" % (
            "fold" + k, tag,
            fmt(o["macro_ap50_gt_classes"]), fmt(o["recall_op"], 3),
            fmt(o["fp_per_image"], 3), fmt(o["gt0_fp_rate"], 3),
            "%s/%s/%s" % (o["tp"], o["fp"], o["fn"])))
    if base in runs and a_ in runs:
        b, x = ov(runs[base]), ov(runs[a_])
        da = x["macro_ap50_gt_classes"] - b["macro_ap50_gt_classes"]
        dr = x["recall_op"] - b["recall_op"]
        dfp = x["fp_per_image"] - b["fp_per_image"]
        print("%-10s %-16s %9s %9s %9s" % (
            "", "Δ (变体-基线)", ("%+.4f" % da), ("%+.3f" % dr), ("%+.3f" % dfp)))
    print("-" * 100)

print("\n" + "=" * 100)
print("四、阈值校准（monitor 集上选 t*，在留出折上报告）")
print("=" * 100)
for tag in ("calib", "calib_r2"):
    p = os.path.join(M, "calib_._%s_calib_results.json" % tag)
    if not os.path.exists(p):
        continue
    d = load(p)
    print("\n[%s]" % ("第1轮" if tag == "calib" else "第2轮"))
    print("%-8s %8s %10s %22s %22s" % (
        "折", "t*", "monitor图", "协议点 conf0.25", "校准点 t*"))
    print("-" * 80)
    for k in sorted(d):
        v = d[k]
        sc = v.get("monitor_scan", {})
        at = sc.get("0.25", {})
        ts = sc.get(str(v.get("t_star")), {}) or sc.get("%s" % v.get("t_star"), {})
        print("%-8s %8s %10s %22s %22s" % (
            k, v.get("t_star"), v.get("n_monitor_imgs"),
            "R=%.3f FP=%.2f" % (at.get("recall", 0), at.get("fp_per_img", 0)),
            "R=%.3f FP=%.2f" % (ts.get("recall", 0), ts.get("fp_per_img", 0))))

print("\n" + "=" * 100)
print("五、逐类 AP50（第2轮四折，GT 存在的类）")
print("=" * 100)
allcls = {}
for k in ["eval_dgv2a_r2_f%d" % i for i in range(4)]:
    if k not in runs:
        continue
    for c, v in ov(runs[k]).get("per_class_ap50", {}).items():
        if v is None:
            continue
        allcls.setdefault(c, []).append(v)
print("%-10s %10s %10s %8s" % ("类别", "均值AP50", "各折", "折数"))
print("-" * 60)
for c in sorted(allcls, key=lambda x: -np.mean(allcls[x])):
    v = allcls[c]
    print("%-10s %10s %-10s %8d" % (
        c, fmt(np.mean(v)), " ".join(fmt(x, 3) for x in v), len(v)))

print("\n" + "=" * 100)
print("六、交付模型（全量训练，自测仅作收敛 sanity）")
print("=" * 100)
rp = os.path.join(BASE, "results", "results_._deploy_deploy_results.csv")
if os.path.exists(rp):
    import csv
    rows = list(csv.DictReader(open(rp, encoding="utf-8")))
    keys = [k for k in rows[0].keys() if "mAP50" in k or "metrics/precision" in k
            or "metrics/recall" in k]
    print("epochs: %d" % len(rows))
    last = rows[-1]
    for k in keys:
        print("  %-32s %s" % (k, last[k]))
print("\nANALYZE_DONE")
