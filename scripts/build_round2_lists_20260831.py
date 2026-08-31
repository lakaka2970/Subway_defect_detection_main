# -*- coding: utf-8 -*-
"""P1-3a：构建硬负样本第 2 轮训练清单（跨折防泄漏）。

输入：docs/plans/8.31阶段1产物/hn_boxes_round1.jsonl（四折测试集挖出的 207 个 FP，
      全图坐标）+ 本地 tiles_index/train_fold{k}.txt。
逻辑：
  1) FP 框中心 → 原图 5×5 网格 tile 名 R_{stem}_t{iy*5+ix}（与生成协议一致）；
  2) 折 k 的过采样集 = 属于折 j!=k 图像的 FP tile（折 k 测试图的 FP 绝不进
     折 k 训练清单——防 LOSO 泄漏）；每 tile 过采样 x3；
  3) 输出 train2_fold{k}.txt = 原清单 + 过采样行；上传远端 tiles/index/。
"""
import csv
import json
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HN = os.path.join(ROOT, "docs", "plans", "8.31阶段1产物",
                  "hn_boxes_round1.jsonl")
IDX = os.path.join(ROOT, "data", "tiles_index")
RROOT = "/root/autodl-tmp/subway/data/tiles"
OVERSAMPLE = 3


def main():
    fold_of = {}
    for r in csv.DictReader(open(os.path.join(
            ROOT, "data", "Defect_dataset_16_rebuilt",
            "fold_assignments.csv"), encoding="utf-8-sig")):
        fold_of[r["image"]] = int(r["fold"])

    # FP -> tile
    tiles_by_fold_img = defaultdict(set)   # fold of source image -> tile paths
    n_fp = n_tile = 0
    for ln in open(HN, encoding="utf-8"):
        b = json.loads(ln)
        n_fp += 1
        img = b["image"]
        f = fold_of.get(img)
        if f is None:
            continue
        stem = os.path.splitext(img)[0]
        cx = (b["box"][0] + b["box"][2]) / 2
        cy = (b["box"][1] + b["box"][3]) / 2
        ix = min(int(cx / 960), 4)
        iy = min(int(cy / 960), 4)
        tn = "R_%s_t%d" % (stem, iy * 5 + ix)
        tiles_by_fold_img[f].add("%s/real/images/%s.jpg" % (RROOT, tn))
        n_tile += 1
    print("FP %d 个 -> tile %d 个（去重后 %s）" %
          (n_fp, n_tile, {k: len(v) for k, v in sorted(tiles_by_fold_img.items())}))

    for k in range(4):
        base = open(os.path.join(IDX, "train_fold%d.txt" % k),
                    encoding="utf-8").read().splitlines()
        extra = []
        for j, tset in tiles_by_fold_img.items():
            if j == k:
                continue
            allow = set(base)
            extra += [t for t in sorted(tset) for _ in range(OVERSAMPLE)
                      if t in allow]
        out = os.path.join(IDX, "train2_fold%d.txt" % k)
        with open(out, "w", encoding="utf-8", newline="\n") as fp:
            fp.write("\n".join(base + extra))
        print("fold%d: base %d + 过采样 %d = %d" %
              (k, len(base), len(extra), len(base) + len(extra)))
    print("R2LISTS_DONE")


if __name__ == "__main__":
    main()
