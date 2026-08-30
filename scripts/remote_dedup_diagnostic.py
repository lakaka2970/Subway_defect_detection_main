#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断 pHash 去重是否误杀：统计被判重复的图像对之间的公里标差。

若大量被判重复的图公里标相差很远（不同位置），说明 pHash 阈值过松，需要收紧或弃用。
"""
import os, re, glob, itertools, random
from collections import Counter
import cv2
import numpy as np

NORMAL = "/root/autodl-tmp/subway/data/Normal_dataset"
TEST = "/root/autodl-tmp/subway/scripts/test770.csv"


def phash(im, size=32):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    d = cv2.dct(g.astype(np.float32))[:8, :8].flatten()
    return (d[1:] > np.median(d[1:])).astype(np.uint8)


def km(name):
    m = re.match(r"^\d+_K(\d+)_(F\w+)-", name)
    return (m.group(2), int(m.group(1))) if m else ("UNK", -1)


def main():
    ext = set()
    if os.path.exists(TEST):
        ext = set(open(TEST).read().split()[1:])
    files = []
    for sub in ["train", "val"]:
        files += sorted(glob.glob(os.path.join(NORMAL, "images", sub, "*.jpg")))
    files = [f for f in files if os.path.basename(f) not in ext]
    print("参与诊断 %d 张" % len(files))

    random.seed(0)
    sample = files if len(files) <= 400 else random.sample(files, 400)
    meta, hs = [], []
    for f in sample:
        im = cv2.imread(f, cv2.IMREAD_REDUCED_COLOR_8)
        if im is None:
            continue
        meta.append((os.path.basename(f), km(os.path.basename(f))))
        hs.append(phash(im))
    hs = np.vstack(hs)
    print("有效 %d 张" % len(meta))

    # 两两汉明距离
    n = len(hs)
    d = np.zeros((n, n), np.uint8)
    for i in range(n):
        d[i] = np.count_nonzero(hs != hs[i], axis=1)

    print()
    print("汉明距离分布（重复判定阈值敏感性）")
    print("%6s %10s %10s %12s %12s" % ("阈值", "判定重复对", "占全部对", "同位置占比", "异区段占比"))
    total = n * (n - 1) // 2
    for th in [0, 1, 2, 3, 4, 6, 8]:
        iu = np.triu_indices(n, 1)
        mask = d[iu] <= th
        idx = np.array(iu).T[mask]
        if len(idx) == 0:
            print("%6d %10d %10s %12s %12s" % (th, 0, "0.00%", "-", "-"))
            continue
        same_pos, diff_seg = 0, 0
        dks = []
        for a, b in idx:
            (sa, ka), (sb, kb) = meta[a][1], meta[b][1]
            if sa == sb and ka == kb:
                same_pos += 1
            elif sa != sb:
                diff_seg += 1
            dks.append(abs(ka - kb))
        print("%6d %10d %9.2f%% %11.1f%% %11.1f%%"
              % (th, len(idx), 100.0 * len(idx) / total,
                 100.0 * same_pos / len(idx), 100.0 * diff_seg / len(idx)))
        print("        公里标差中位数 %s" % int(np.median(dks)))

    # 按"唯一公里标"去重后还剩多少
    keys = set((km(os.path.basename(f)) for f in files))
    print()
    print("按 (区段,公里标) 主键去重: %d 张 -> %d 个唯一位置" % (len(files), len(keys)))


if __name__ == "__main__":
    main()
