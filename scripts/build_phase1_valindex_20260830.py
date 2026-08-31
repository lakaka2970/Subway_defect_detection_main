# -*- coding: utf-8 -*-
"""车间 val 切片 + 生成远端训练/验证清单（折索引）。

- 车间 val 551 张按 train 同协议切片（前缀 V_）入 data/tiles_workshop_val；
- 真实线每折内按支柱号抽 ~15% 作该折验证图（早停监控，绝不进测试）；
- 清单（远端绝对路径）：
    train_fold{k}.txt = W 全 + N 全 + R(折!=k 且非 val 图)
    val_fold{k}.txt   = V 全 + R(折!=k 的 val 图)
种子 42。
"""
import csv
import glob
import os
import random
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RROOT = "/root/autodl-tmp/subway/data/tiles"
OUT_V = os.path.join(ROOT, "data", "tiles_workshop_val")
IDX = os.path.join(ROOT, "data", "tiles_index")
SEED = 42

import importlib.util
import sys
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import build_phase1_tiles_20260830 as bt


def tile_val():
    os.makedirs(os.path.join(OUT_V, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_V, "labels"), exist_ok=True)
    img, lab = {}, {}
    for p in glob.glob(os.path.join(ROOT, "data", "train_data_2", "val",
                                    "images", "**", "*.jpg"), recursive=True):
        img[os.path.splitext(os.path.basename(p))[0]] = p
    for p in glob.glob(os.path.join(ROOT, "data", "train_data_2", "val",
                                    "labels", "**", "*.txt"), recursive=True):
        lab[os.path.splitext(os.path.basename(p))[0]] = p
    jobs = [(img[s], lab.get(s, ""), s, False) for s in sorted(img)]
    with Pool(8) as pool:
        res = pool.map(bt.tile_workshop, [(ip, lp, s, w, "V", OUT_V)
                                          for ip, lp, s, w in jobs])
    print("车间 val tile: %d" % sum(r[1] for r in res))


def main():
    if not glob.glob(os.path.join(OUT_V, "images", "*.jpg")):
        tile_val()
    rnd = random.Random(SEED)

    # 折与真实线 val 划分
    fold_of, kpost = {}, {}
    for r in csv.DictReader(open(os.path.join(
            ROOT, "data", "Defect_dataset_16_rebuilt",
            "fold_assignments.csv"), encoding="utf-8-sig")):
        fold_of[r["image"]] = int(r["fold"])
        kpost[r["image"]] = int(r["kpost"]) if r["kpost"] else 0
    val_imgs = set()
    for f in range(4):
        members = sorted((n for n, ff in fold_of.items() if ff == f),
                         key=lambda n: kpost[n])
        # 按公里标等距抽 15%（保持空间分散）
        step = max(1, round(1 / 0.15))
        pick = [n for i, n in enumerate(members) if i % step == 3]
        val_imgs.update(pick)
        print("折%d: %d 图, val %d" % (f, len(members), len(pick)))

    def rpath(sub, stem):
        return "%s/%s/images/%s.jpg" % (RROOT, sub, stem)

    os.makedirs(IDX, exist_ok=True)
    import json
    man = json.load(open(os.path.join(ROOT, "data", "train_data_2",
                                      "manifest.json"), encoding="utf-8"))
    aug_out = {v["out"] for v in man["variants"]}
    w_all = sorted(os.path.splitext(os.path.basename(p))[0] for p in
                   glob.glob(os.path.join(ROOT, "data", "tiles_workshop",
                                          "images", "*.jpg")))
    # W_<src>_tN：仅保留 base 源图（数据纪律：排除 5211 旧离线增强）
    w_tiles = [s for s in w_all if s[2:].rsplit("_t", 1)[0] not in aug_out]
    print("W tile 全 %d, base 过滤后 %d (排除增强源)" % (len(w_all), len(w_tiles)))
    v_tiles = sorted(os.path.splitext(os.path.basename(p))[0] for p in
                     glob.glob(os.path.join(OUT_V, "images", "*.jpg")))
    n_tiles = sorted(os.path.splitext(os.path.basename(p))[0] for p in
                     glob.glob(os.path.join(ROOT, "data", "tiles_normal",
                                            "images", "*.jpg")))
    r_tiles = {}
    for p in glob.glob(os.path.join(ROOT, "data", "tiles_real",
                                    "images", "*.jpg")):
        stem = os.path.splitext(os.path.basename(p))[0]   # R_<imgstem>_tN
        imgname = stem[2:].rsplit("_t", 1)[0] + ".jpg"
        r_tiles.setdefault(imgname, []).append(stem)
    print("W %d / V %d / N %d / R图 %d" %
          (len(w_tiles), len(v_tiles), len(n_tiles), len(r_tiles)))

    for f in range(4):
        tr, va = [], []
        tr += ["%s/workshop/images/%s.jpg" % (RROOT, s) for s in w_tiles]
        tr += ["%s/normal/images/%s.jpg" % (RROOT, s) for s in n_tiles]
        for imgname, stems in r_tiles.items():
            ff = fold_of.get(imgname)
            if ff is None or ff == f:
                continue
            if imgname in val_imgs:
                va += ["%s/real/images/%s.jpg" % (RROOT, s) for s in stems]
            else:
                tr += ["%s/real/images/%s.jpg" % (RROOT, s) for s in stems]
        va += ["%s/workshop_val/images/%s.jpg" % (RROOT, s) for s in v_tiles]
        with open(os.path.join(IDX, "train_fold%d.txt" % f), "w",
                  encoding="utf-8", newline="\n") as fp:
            fp.write("\n".join(tr))
        with open(os.path.join(IDX, "val_fold%d.txt" % f), "w",
                  encoding="utf-8", newline="\n") as fp:
            fp.write("\n".join(va))
        print("fold%d: train %d, val %d" % (f, len(tr), len(va)))
    print("INDEX_DONE")


if __name__ == "__main__":
    main()
