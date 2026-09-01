# -*- coding: utf-8 -*-
"""消融A 准备：night/tunnel/degrade 低照度变体 tile 清单 + tar 打包。

A 组配方 = 第1轮 r1 完全相同（stage4 初始化、12ep、同 seed），唯一变量：
训练清单追加低照度变体源图的 tile（车间源图，无真实线路内容，不涉 LOSO 泄漏）。
对照折：fold0 与 fold2（一弱一强，看变体对不同区段的作用方向）。
"""
import glob
import json
import os
import tarfile
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDX = os.path.join(ROOT, "data", "tiles_index")
TW = os.path.join(ROOT, "data", "tiles_workshop")
RROOT = "/root/autodl-tmp/subway/data/tiles"
AUGS = {"night", "tunnel", "degrade"}


def main():
    man = json.load(open(os.path.join(ROOT, "data", "train_data_2",
                                      "manifest.json"), encoding="utf-8"))
    aug_out = {v["out"] for v in man["variants"] if v["aug"] in AUGS}
    print("变体源图:", len(aug_out), dict(Counter(
        v["aug"] for v in man["variants"] if v["aug"] in AUGS)))

    a_tiles = []
    for p in glob.glob(os.path.join(TW, "images", "W_*.jpg")):
        stem = os.path.splitext(os.path.basename(p))[0]
        src = stem[2:].rsplit("_t", 1)[0]
        if src in aug_out:
            a_tiles.append(stem)
    print("A 组 tile:", len(a_tiles))

    for k in (0, 2):
        base = open(os.path.join(IDX, "train_fold%d.txt" % k),
                    encoding="utf-8").read().splitlines()
        extra = ["%s/workshop/images/%s.jpg" % (RROOT, s) for s in
                 sorted(a_tiles)]
        with open(os.path.join(IDX, "train_a_fold%d.txt" % k), "w",
                  encoding="utf-8", newline="\n") as fp:
            fp.write("\n".join(base + extra))
        print("train_a_fold%d: %d + %d = %d" %
              (k, len(base), len(extra), len(base) + len(extra)))

    out = os.path.join(ROOT, "data", "_upload", "ablationA_tiles.tar")
    n = 0
    with tarfile.open(out, "w") as t:
        for s in sorted(a_tiles):
            for ext, kind in (("jpg", "images"), ("txt", "labels")):
                p = os.path.join(TW, kind, s + "." + ext)
                if os.path.exists(p):
                    t.add(p, arcname="tiles/workshop/%s/%s.%s"
                          % (kind, s, ext))
                    n += 1
    print("TAR_OK %d 文件 %dMB" % (n, os.path.getsize(out) // 1048576))


if __name__ == "__main__":
    main()
