# -*- coding: utf-8 -*-
"""
验证 data/Normal_dataset（1000 张现场无缺陷图）是否与部署域同域。

若它与 Defect_dataset（检测车实拍，含缺陷）在像素特征上不可分（探针 ≈ 0.5），
则它可以直接作为"目标域无缺陷底图"用于：
  1. 直接作为负样本进训练（治 P(缺陷|部件) 先验错位）
  2. 作为缺陷合成的背景池（用户提议的"AI 半自动 PS"的底图）

对照：
  A  车间 vs 检测车           期望 ≈ 1.0   （已知：两域断裂）
  B  Normal vs 检测车         期望 ≈ 0.5   （若成立，说明是同一域）
  C  检测车前半 vs 检测车后半   期望 ≈ 0.5   （同域对照，验证探针）
"""
import os, sys, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_domain_align_20260829 import (extract, prep, sample_dir, probe_acc,
                                         WS_DIR, ACT_DIR, STYLE_KEYS, STRUCT_KEYS)

random.seed(42)
np.random.seed(42)

NORM_DIR = os.path.join("data", "Normal_dataset", "images", "train")


def main():
    root = r"E:\Work\Subway_defect_detection_main"
    norm_dir = os.path.join(root, NORM_DIR)
    n_norm, n_act, n_ws = 200, 200, 200

    print("载入 Normal_dataset %d / 检测车 %d / 车间 %d ..." % (n_norm, n_act, n_ws))
    norm = [prep(p) for p in sample_dir(norm_dir, n_norm)]
    act = [prep(p) for p in sample_dir(ACT_DIR, n_act)]
    ws = [prep(p) for p in sample_dir(WS_DIR, n_ws)]
    norm = [x for x in norm if x is not None]
    act = [x for x in act if x is not None]
    ws = [x for x in ws if x is not None]
    print("  实际: Normal %d / 检测车 %d / 车间 %d" % (len(norm), len(act), len(ws)))

    print("抽特征 ...")
    f_norm = [extract(x) for x in norm]
    f_act = [extract(x) for x in act]
    f_ws = [extract(x) for x in ws]
    KEYS = list(f_norm[0].keys())

    h = len(f_act) // 2
    tests = [
        ("A  车间 vs 检测车（已知断裂）", f_ws, f_act),
        ("B  Normal vs 检测车（关键）", f_norm, f_act),
        ("C  检测车前半 vs 后半（对照）", f_act[:h], f_act[h:]),
        ("D  Normal 前半 vs 后半（对照）", f_norm[:len(f_norm) // 2], f_norm[len(f_norm) // 2:]),
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
    print("判读：若 B 接近 0.5，说明 Normal_dataset 与部署域同域，可直接用作底图与负样本；")
    print("      若 B 接近 1.0，说明它虽然无缺陷但成像条件与部署域不同，需先做域校正。")

    # 关键统计量对照
    print()
    print("关键统计量中位数")
    print("%-22s %12s %12s %12s" % ("feature", "车间", "Normal", "检测车(有缺陷)"))
    for k in ["luma_mean", "luma_std", "sat_mean", "colorfulness", "edge_density",
              "laplacian_var", "entropy", "fft_high_ratio"]:
        a = float(np.median([f[k] for f in f_ws]))
        b = float(np.median([f[k] for f in f_norm]))
        c = float(np.median([f[k] for f in f_act]))
        print("%-22s %12.4f %12.4f %12.4f" % (k, a, b, c))


if __name__ == "__main__":
    main()
