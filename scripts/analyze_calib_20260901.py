# -*- coding: utf-8 -*-
"""阈值校准结果汇总：r1 / r2 各折 t*、test@0.25 与 test@t* 的 Recall / FP 对照。"""
import json
import os

BASE = r"E:\Work\Subway_defect_detection_main\docs\plans\9.01阶段2产物\remain_pkg\calib"
OUT = []


def p(s=""):
    print(s)
    OUT.append(s)


p("=" * 96)
p("阈值校准总表（monitor 集选 t*，held-out 折上报，无泄漏）")
p("=" * 96)
p("%-8s %-6s %8s | %-22s | %-22s | %8s" % (
    "轮次", "折", "t*", "conf=0.25 (R / FP每图)", "conf=t* (R / FP每图)", "ΔRecall"))
p("-" * 96)

rows = []
for tag, fn in [("r1", "calib_r1.json"), ("r2", "calib_r2.json")]:
    d = json.load(open(os.path.join(BASE, fn), encoding="utf-8"))
    for f in ["fold0", "fold1", "fold2", "fold3"]:
        e = d[f]
        ts = e["t_star"]
        a = e["test_at_0.25"]
        b = e["test_at_t_star"]
        dr = b["recall"] - a["recall"]
        rows.append((tag, f, ts, a, b, dr))
        p("%-8s %-6s %8.2f | R=%.3f  FP=%.2f        | R=%.3f  FP=%.2f        | %+7.3f" % (
            tag, f, ts, a["recall"], a["fp_per_img"], b["recall"], b["fp_per_img"], dr))

p("-" * 96)
for tag in ["r1", "r2"]:
    sub = [r for r in rows if r[0] == tag]
    m_a = sum(x[3]["recall"] for x in sub) / len(sub)
    m_b = sum(x[4]["recall"] for x in sub) / len(sub)
    m_fa = sum(x[3]["fp_per_img"] for x in sub) / len(sub)
    m_fb = sum(x[4]["fp_per_img"] for x in sub) / len(sub)
    tp_a = sum(x[3]["tp"] for x in sub)
    tp_b = sum(x[4]["tp"] for x in sub)
    fp_a = sum(x[3]["fp"] for x in sub)
    fp_b = sum(x[4]["fp"] for x in sub)
    fn_a = sum(x[3]["fn"] for x in sub)
    fn_b = sum(x[4]["fn"] for x in sub)
    p("%s 四折宏均：conf=0.25 → R=%.3f FP=%.2f (合并 TP/FP/FN = %d/%d/%d, 合并R=%.3f)" % (
        tag, m_a, m_fa, tp_a, fp_a, fn_a, tp_a / (tp_a + fn_a)))
    p("%s 四折宏均：conf=t*   → R=%.3f FP=%.2f (合并 TP/FP/FN = %d/%d/%d, 合并R=%.3f)" % (
        tag, m_b, m_fb, tp_b, fp_b, fn_b, tp_b / (tp_b + fn_b)))
    p("%s t* 取值：%s" % (tag, ", ".join("%s=%.2f" % (x[1], x[2]) for x in sub)))
    p("")

p("=" * 96)
p("Monitor 集扫描曲线（每折 tp/fp/fn 随 conf 变化）—— 用于判断「看得见 vs 看不见」")
p("=" * 96)
for tag, fn in [("r1", "calib_r1.json"), ("r2", "calib_r2.json")]:
    d = json.load(open(os.path.join(BASE, fn), encoding="utf-8"))
    for f in ["fold0", "fold1", "fold2", "fold3"]:
        e = d[f]
        p("[%s %s] monitor=%d张, t*=%.2f" % (tag, f, e["n_monitor_imgs"], e["t_star"]))
        keys = sorted(e["monitor_scan"].keys(), key=float)
        line = []
        for k in keys:
            v = e["monitor_scan"][k]
            line.append("%s:R%.2f/F%.1f" % (k, v["recall"], v["fp_per_img"]))
        p("   " + "  ".join(line))
p("")
p("CALIB_DONE")
