# -*- coding: utf-8 -*-
"""
域探针有效性对照实验（2026-08-29）

目的：验证 probe_domain_align_20260829.py 里的域探针确实在测"域差异"，
      而不是无论喂什么都能得到 1.0000（那会推翻全部结论）。

对照：
  A. 车间 vs 检测车          期望 ≈ 1.0   （真域差异）
  B. 车间前半 vs 车间后半    期望 ≈ 0.5   （同域，应不可分）
  C. 检测车前半 vs 检测车后半 期望 ≈ 0.5   （同域，应不可分）
  D. 退化后车间 vs 检测车    复现主实验
"""
import os, sys, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_domain_align_20260829 import (extract, prep, sample_dir, probe_acc,
                                         hist_match, degrade, WS_DIR, ACT_DIR,
                                         STYLE_KEYS, STRUCT_KEYS, ANALYZE)

random.seed(42)
np.random.seed(42)

KEYS = None


def main():
    n_ws, n_act = 150, 200
    print("载入 %d 车间 + %d 检测车 ..." % (n_ws, n_act))
    ws = [prep(p) for p in sample_dir(WS_DIR, n_ws)]
    act = [prep(p) for p in sample_dir(ACT_DIR, n_act)]
    ws = [x for x in ws if x is not None]
    act = [x for x in act if x is not None]
    print("  车间 %d / 检测车 %d" % (len(ws), len(act)))

    print("抽特征 ...")
    f_ws = [extract(x) for x in ws]
    f_act = [extract(x) for x in act]
    global KEYS
    KEYS = list(f_ws[0].keys())

    # 参考累积分布 + 复用主实验标定出的参数
    ref_luts = []
    for im in act[:150]:
        for c in range(3):
            h = np.histogram(im[:, :, c], bins=256, range=(0, 256))[0].astype(np.float64)
            ref_luts.append(h.cumsum() / (h.sum() + 1e-9))
    ref_cdf = np.median(np.array(ref_luts), axis=0)
    ref_cdf = ref_cdf / ref_cdf[-1]
    BLUR, NOISE = 0.60, 2.00

    print("生成退化版车间图 ...")
    f_ws_d = [extract(degrade(ws[i], ref_cdf, BLUR, NOISE)) for i in range(len(ws))]

    hw = len(f_ws) // 2
    ha = len(f_act) // 2

    tests = [
        ("A  车间 vs 检测车（真域差异）", f_ws, f_act),
        ("B  车间前半 vs 车间后半（同域对照）", f_ws[:hw], f_ws[hw:]),
        ("C  检测车前半 vs 检测车后半（同域对照）", f_act[:ha], f_act[ha:]),
        ("D  退化后车间 vs 检测车", f_ws_d, f_act),
    ]

    print()
    print("=" * 74)
    print("%-38s %8s %8s %8s" % ("对照", "全部", "风格", "结构"))
    print("=" * 74)
    for name, a, b in tests:
        r = (probe_acc(a, b, KEYS), probe_acc(a, b, STYLE_KEYS), probe_acc(a, b, STRUCT_KEYS))
        print("%-38s %8.4f %8.4f %8.4f" % (name, r[0], r[1], r[2]))
    print("=" * 74)
    print()
    print("判读：B、C 需接近 0.50 才说明探针有效；A 接近 1.00 说明两域确实可分；")
    print("      D 与 A 的差距 = 离线退化管线实际买到的域对齐量。")


if __name__ == "__main__":
    main()
