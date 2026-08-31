# -*- coding: utf-8 -*-
"""生成阶段1混合训练集的三路切片（本地 CPU，多进程）。

A 车间协议对齐切片：train_data_2/train 原图先 3.3x 下采样，再 1280/stride640 滑窗；
  标签同步换算；保留条件：裁剪面积比>=0.4 且裁剪短边>=20px。
  另保留 20% 整图（letterbox 对照分支）+ 全部 val 整图（监控用，不进训练列表）。
B 真实线切片：基准 534 张按 1280/stride960 切 25 tile（含 GT=0 空标签 tile）。
D Normal 负样本切片：931 张可用中分层抽 220 张，同 B 协议。
种子 42。输出 data/tiles_* 与 tiles_index/ 清单。
"""
import glob
import json
import os
import random
from collections import defaultdict
from multiprocessing import Pool

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_W = os.path.join(ROOT, "data", "tiles_workshop")
OUT_R = os.path.join(ROOT, "data", "tiles_real")
OUT_N = os.path.join(ROOT, "data", "tiles_normal")
IDX = os.path.join(ROOT, "data", "tiles_index")
SEED = 42
TILE, STRIDE_W, STRIDE_R = 1280, 640, 960
DS = 3.3
JPEG_Q = 88


def offsets(n, stride, size):
    if size <= TILE:
        return [0]
    out, i = [], 0
    while True:
        o = min(i * stride, size - TILE)
        if not out or o != out[-1]:
            out.append(o)
        if o == size - TILE:
            break
        i += 1
    return out


def read_label(p):
    boxes = []
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8"):
            f = ln.split()
            if len(f) == 5:
                boxes.append([int(f[0])] + [float(x) for x in f[1:]])
    return boxes


def write_tile(img, box_list, out_img, out_lab):
    img.save(out_img, "JPEG", quality=JPEG_Q)
    with open(out_lab, "w", encoding="utf-8") as fp:
        for b in box_list:
            fp.write("%d %.6f %.6f %.6f %.6f\n" % tuple(b))


def tile_workshop(args):
    ip, lp, stem, whole = args[:4]
    prefix = args[4] if len(args) > 4 else "W"
    out = args[5] if len(args) > 5 else OUT_W
    im = Image.open(ip).convert("RGB")
    W, H = im.size
    nw, nh = int(round(W / DS)), int(round(H / DS))
    small = im.resize((nw, nh), Image.LANCZOS)
    boxes = read_label(lp)
    px = [(b[0], b[1] * nw, b[2] * nh, b[3] * nw, b[4] * nh) for b in boxes]
    xs, ys = offsets(nw, STRIDE_W, nw), offsets(nh, STRIDE_W, nh)
    n = 0
    for yi, ty in enumerate(ys):
        for xi, tx in enumerate(xs):
            crop = small.crop((tx, ty, min(tx + TILE, nw), min(ty + TILE, nh)))
            if crop.size[0] < TILE or crop.size[1] < TILE:
                pad = Image.new("RGB", (TILE, TILE), (114, 114, 114))
                pad.paste(crop, (0, 0))
                crop = pad
            kept = []
            for c, cx, cy, w, h in px:
                x0, y0 = cx - w / 2, cy - h / 2
                x1, y1 = cx + w / 2, cy + h / 2
                ix0, iy0 = max(x0, tx), max(y0, ty)
                ix1, iy1 = min(x1, tx + TILE), min(y1, ty + TILE)
                iw, ih = ix1 - ix0, iy1 - iy0
                if iw <= 0 or ih <= 0:
                    continue
                area_ratio = (iw * ih) / max(1.0, w * h)
                if area_ratio < 0.4 or min(iw, ih) < 20:
                    continue
                kept.append([c, (ix0 + iw / 2 - tx) / TILE,
                             (iy0 + ih / 2 - ty) / TILE, iw / TILE, ih / TILE])
            tn = "%s_%s_t%d" % (prefix, stem, n)
            write_tile(crop, kept,
                       os.path.join(out, "images", tn + ".jpg"),
                       os.path.join(out, "labels", tn + ".txt"))
            n += 1
    return stem, n, whole


def tile_full(args):
    ip, lp, stem, prefix, out = args
    im = Image.open(ip).convert("RGB")
    W, H = im.size
    boxes = read_label(lp)
    px = [(b[0], b[1] * W, b[2] * H, b[3] * W, b[4] * H) for b in boxes]
    xs, ys = offsets(W, STRIDE_R, W), offsets(H, STRIDE_R, H)
    n = 0
    for ty in ys:
        for tx in xs:
            crop = im.crop((tx, ty, min(tx + TILE, W), min(ty + TILE, H)))
            if crop.size[0] < TILE or crop.size[1] < TILE:
                pad = Image.new("RGB", (TILE, TILE), (114, 114, 114))
                pad.paste(crop, (0, 0))
                crop = pad
            kept = []
            for c, cx, cy, w, h in px:
                x0, y0 = cx - w / 2, cy - h / 2
                ix0, iy0 = max(x0, tx), max(y0, ty)
                ix1, iy1 = min(x1 := cx + w / 2, tx + TILE), \
                    min(y1 := cy + h / 2, ty + TILE)
                iw, ih = ix1 - ix0, iy1 - iy0
                if iw <= 0 or ih <= 0:
                    continue
                if (iw * ih) / max(1.0, w * h) < 0.4 or min(iw, ih) < 16:
                    continue
                kept.append([c, (ix0 + iw / 2 - tx) / TILE,
                             (iy0 + ih / 2 - ty) / TILE, iw / TILE, ih / TILE])
            tn = "%s_%s_t%d" % (prefix, stem, n)
            write_tile(crop, kept,
                       os.path.join(out, "images", tn + ".jpg"),
                       os.path.join(out, "labels", tn + ".txt"))
            n += 1
    return stem, n


def main():
    for d in (OUT_W, OUT_R, OUT_N):
        os.makedirs(os.path.join(d, "images"), exist_ok=True)
        os.makedirs(os.path.join(d, "labels"), exist_ok=True)
    os.makedirs(IDX, exist_ok=True)
    rnd = random.Random(SEED)

    # ---- A 车间 ----
    tr_img, tr_lab = {}, {}
    for p in glob.glob(os.path.join(ROOT, "data", "train_data_2", "train",
                                    "images", "**", "*.jpg"), recursive=True):
        tr_img[os.path.splitext(os.path.basename(p))[0]] = p
    for p in glob.glob(os.path.join(ROOT, "data", "train_data_2", "train",
                                    "labels", "**", "*.txt"), recursive=True):
        tr_lab[os.path.splitext(os.path.basename(p))[0]] = p
    stems = sorted(tr_img)
    whole_set = set(rnd.sample(stems, max(1, int(len(stems) * 0.2))))
    jobs = [(tr_img[s], tr_lab.get(s, ""), s, s in whole_set) for s in stems]
    print("车间 train 图: %d, 整图分支: %d" % (len(stems), len(whole_set)))
    with Pool(8) as pool:
        res = pool.map(tile_workshop, jobs)
    ntiles = sum(r[1] for r in res)
    print("车间 tile: %d" % ntiles)

    # val 整图（监控）
    val_img = sorted(glob.glob(os.path.join(ROOT, "data", "train_data_2", "val",
                                            "images", "**", "*.jpg"),
                               recursive=True))
    with open(os.path.join(IDX, "workshop_val_whole.txt"), "w") as fp:
        fp.write("\n".join(os.path.relpath(p, ROOT) for p in val_img))

    # ---- B 真实线 ----
    bench = sorted(os.path.basename(p) for p in
                   glob.glob(os.path.join(ROOT, "data",
                                          "Defect_dataset_16_rebuilt",
                                          "images", "*.jpg")))
    fold_of = {}
    import csv
    for r in csv.DictReader(open(os.path.join(ROOT, "data",
                                               "Defect_dataset_16_rebuilt",
                                               "fold_assignments.csv"),
                                 encoding="utf-8-sig")):
        fold_of[r["image"]] = int(r["fold"])
    jobs = []
    for name in bench:
        stem = os.path.splitext(name)[0]
        jobs.append((os.path.join(ROOT, "data", "Defect_dataset_16_rebuilt",
                                  "images", name),
                     os.path.join(ROOT, "data", "Defect_dataset_16_rebuilt",
                                  "labels", stem + ".txt"),
                     stem, "R", OUT_R))
    print("真实线基准图: %d" % len(jobs))
    with Pool(8) as pool:
        res_r = pool.map(tile_full, jobs)
    print("真实线 tile: %d" % sum(r[1] for r in res_r))

    # ---- D Normal ----
    import csv as _csv
    conflict = {r["image"] for r in _csv.DictReader(open(
        os.path.join(ROOT, "data", "Defect_dataset_16_rebuilt",
                     "rebuild_audit.csv"), encoding="utf-8-sig"))
        if r["source"] == "Normal_dataset" and r["status"] == "MISMATCH"}
    used = set(bench) | conflict
    norm_all = []
    for p in glob.glob(os.path.join(ROOT, "data", "Normal_dataset", "images",
                                    "**", "*.jpg"), recursive=True):
        b = os.path.basename(p)
        if b not in used:
            norm_all.append((b, p))
    rnd.shuffle(norm_all)
    sample = norm_all[:220]
    print("Normal 可用: %d, 抽样: %d" % (len(norm_all), len(sample)))
    jobs = [(p, "", os.path.splitext(b)[0], "N", OUT_N) for b, p in sample]
    with Pool(8) as pool:
        res_n = pool.map(tile_full, jobs)
    print("Normal tile: %d" % sum(r[1] for r in res_n))

    # ---- 索引 ----
    with open(os.path.join(IDX, "workshop_whole_20pct.txt"), "w") as fp:
        fp.write("\n".join(sorted(whole_set)))
    with open(os.path.join(IDX, "real_fold.csv"), "w") as fp:
        fp.write("stem,fold\n")
        for name in bench:
            fp.write("%s,%d\n" % (os.path.splitext(name)[0], fold_of[name]))
    print("TILING_DONE")


if __name__ == "__main__":
    main()
