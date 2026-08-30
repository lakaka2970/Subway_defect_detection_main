# -*- coding: utf-8 -*-
"""统计 Defect_dataset_2 的缺陷实例：类别分布、bbox 尺寸分布。
用于规划 SAM 实例库构建的裁剪策略与上传体积估算。
"""
import glob
import os
import collections

import numpy as np

ROOT = r"E:\Work\Subway_defect_detection_main"
IMG_DIR = os.path.join(ROOT, "data", "Defect_dataset_2", "Defect_dataset", "images")
LBL_DIR = os.path.join(ROOT, "data", "Defect_dataset_2", "Defect_dataset", "labels")

names = [l.strip() for l in open(os.path.join(ROOT, "data", "train_data_2", "classes.txt"),
                                 encoding="utf-8") if l.strip()]
n2i = {n: i for i, n in enumerate(names)}


def parse_label(path):
    """返回 [(cls, xc, yc, w, h), ...]，自动兼容数字 ID 与类名，忽略畸形行。"""
    out = []
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            try:
                c = int(p[0]) if p[0].isdigit() else n2i[p[0]]
                v = [float(x) for x in p[1:5]]
            except Exception:
                continue
            out.append((c, v[0], v[1], v[2], v[3]))
    return out


def main():
    lfs = sorted(glob.glob(os.path.join(LBL_DIR, "*.txt")))
    cnt = collections.Counter()
    per_img = collections.Counter()
    areas = []
    sides_w, sides_h = [], []
    tot = 0
    nimg = 0
    sizes = {}

    for lf in lfs:
        rows = parse_label(lf)
        if rows:
            nimg += 1
        per_img[len(rows)] += 1
        for c, xc, yc, w, h in rows:
            tot += 1
            cnt[c] += 1
            areas.append(w * h)
        # 记录图片尺寸（抽样，避免读全量 31GB）
        if len(sizes) < 400:
            stem = os.path.splitext(os.path.basename(lf))[0]
            ip = os.path.join(IMG_DIR, stem + ".jpg")
            if os.path.exists(ip):
                sizes[stem] = os.path.getsize(ip)

    areas = np.array(areas)
    print("标签文件 %d，其中有标注 %d 张，总实例 %d" % (len(lfs), nimg, tot))
    print()
    print("每张图的实例数分布：", dict(sorted(per_img.items())[:8]))
    print()
    print("%-4s %-9s %7s %8s" % ("id", "name", "count", "占比%"))
    for c in range(16):
        n = cnt.get(c, 0)
        print("%-4d %-9s %7d %8.2f" % (c, names[c], n, 100.0 * n / max(1, tot)))
    print()

    # 参考分辨率（用抽样均值）
    avg_px = np.mean([np.sqrt(a) for a in areas]) if len(areas) else 0
    print("bbox 面积占全图比例 %%: 中位 %.5f  均值 %.5f  p90 %.4f  p99 %.3f"
          % (np.median(areas) * 100, areas.mean() * 100,
             np.percentile(areas, 90) * 100, np.percentile(areas, 99) * 100))

    # 以 8192 宽为参考，换算像素边长
    W = 8192.0
    wpx = np.sqrt(areas) * W
    print("折算到 8192×6144 原图的 bbox 边长(px)：中位 %.0f  p10 %.0f  p90 %.0f"
          % (np.median(wpx), np.percentile(wpx, 10), np.percentile(wpx, 90)))

    # 估算裁剪体积：2.5x padding，cap 1024
    crop_px = np.clip(np.sqrt(areas) * W * 2.5, 64, 1024)
    est_bytes = (crop_px ** 2 * 3 * 0.25).sum()  # jpeg q95 粗略 0.25 B/px
    print()
    print("实例裁剪估算（2.5x pad, 最长边 cap 1024, JPEG q95）：")
    print("  平均裁剪边长 %.0f px，总体积约 %.0f MB" % (crop_px.mean(), est_bytes / 1e6))
    if sizes:
        print("  单张原图平均 %.1f MB（抽样 %d 张）" % (np.mean(list(sizes.values())) / 1e6, len(sizes)))


if __name__ == "__main__":
    main()
