# -*- coding: utf-8 -*-
"""
从 data/Normal_dataset 生成「现场无缺陷负样本」样本数据集，供人工检查效果。

本地 CPU 版：按线路段分层抽样 -> 1280 滑窗切图 -> 内容过滤 -> 空标签 -> 预览拼图。
GPU 实例就绪后，用 --n-images 扩到全量即可（同一套逻辑）。

输出：data/normal_field_v1_sample/
"""
from __future__ import annotations

import argparse, csv, json, os, random, re, shutil, time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r"E:\Work\Subway_defect_detection_main")
NORMAL = ROOT / "data" / "Normal_dataset"
DEFECT = ROOT / "data" / "Defect_dataset"          # 检测车实拍（含缺陷），用于域对比
EXT_CSV = ROOT / "docs/plans/8.28泛化性训练/8.28泛化性训练/DG-v1_20260828/external_test_770/image_summary.csv"

TILE, STRIDE, Q = 1280, 960, 92
random.seed(42); np.random.seed(42)


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def seg_of(n):
    m = re.match(r"^\d+_K(\d+)_(F\w+)-", n)
    return m.group(2) if m else "UNKNOWN"


def entropy(g):
    h = np.histogram(g, bins=256, range=(0, 256))[0].astype(np.float64)
    p = h / (h.sum() + 1e-9); nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def montage(tiles, cols, cell=320, bg=32):
    rows = (len(tiles) + cols - 1) // cols
    H = rows * cell
    W = cols * cell
    out = np.full((H, W, 3), bg, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        im = cv2.resize(t, (cell - 6, cell - 6), interpolation=cv2.INTER_AREA)
        y, x = r * cell + 3, c * cell + 3
        out[y:y + im.shape[0], x:x + im.shape[1]] = im
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=24, help="抽多少张源图")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--min-std", type=float, default=8.0)
    ap.add_argument("--min-entropy", type=float, default=3.0)
    ap.add_argument("--out", default="data/normal_field_v1_sample")
    args = ap.parse_args()

    out = ROOT / args.out
    if out.exists():
        shutil.rmtree(out)
    for s in ["images/train", "images/val", "labels/train", "labels/val", "preview"]:
        (out / s).mkdir(parents=True, exist_ok=True)

    # ---------- 1. 收集 + 剔除测试集重叠 ----------
    log("步骤 1/5 收集与去重")
    src = sorted(list((NORMAL / "images" / "train").glob("*.jpg")) +
                 list((NORMAL / "images" / "val").glob("*.jpg")))
    ext = set()
    if EXT_CSV.exists():
        with open(EXT_CSV, encoding="utf-8") as fp:
            ext = {r["image"] for r in csv.DictReader(fp)}
    kept = [p for p in src if p.name not in ext]
    log("  %d 张 -> 剔除与 770 张测试集重叠 %d 张 -> 剩 %d 张"
        % (len(src), len(src) - len(kept), len(kept)))

    # ---------- 2. 按区段分层抽样 ----------
    by_seg = defaultdict(list)
    for p in kept:
        by_seg[seg_of(p.name)].append(p)
    segs = sorted(by_seg, key=lambda s: -len(by_seg[s]))
    log("  区段分布: %s" % {s: len(by_seg[s]) for s in segs})

    picked, per = [], max(1, args.n_images // max(1, len(segs)))
    for s in segs:
        ps = by_seg[s][:]
        random.shuffle(ps)
        picked += ps[:per]
        if len(picked) >= args.n_images:
            break
    picked = picked[:args.n_images]
    log("  分层抽样选出 %d 张，区段: %s"
        % (len(picked), dict(Counter(seg_of(p.name) for p in picked))))

    # ---------- 3. 切分 ----------
    random.shuffle(picked)
    n_val = max(1, int(round(len(picked) * args.val_frac)))
    splits = {"val": picked[:n_val], "train": picked[n_val:]}
    log("步骤 2/5 切分: train %d / val %d" % (len(splits["train"]), len(splits["val"])))

    # ---------- 4. 滑窗切图 + 内容过滤 + 空标签 ----------
    log("步骤 3/5 滑窗切图 %dx%d / stride %d" % (TILE, TILE, STRIDE))
    rows, n_kept, n_drop = [], 0, 0
    preview_src = defaultdict(list)
    for split, paths in splits.items():
        idir, ldir = out / "images" / split, out / "labels" / split
        for p in paths:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is None:
                continue
            h, w = im.shape[:2]
            ys = list(range(0, max(1, h - TILE + 1), STRIDE))
            xs = list(range(0, max(1, w - TILE + 1), STRIDE))
            if ys and ys[-1] != h - TILE: ys.append(h - TILE)
            if xs and xs[-1] != w - TILE: xs.append(w - TILE)
            for yi, y in enumerate(ys):
                for xi, x in enumerate(xs):
                    t = im[y:y + TILE, x:x + TILE]
                    if t.shape[0] != TILE or t.shape[1] != TILE:
                        continue
                    g = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
                    sd, en = float(g.std()), entropy(g)
                    if sd < args.min_std or en < args.min_entropy:
                        n_drop += 1
                        rows.append(dict(image=p.name, split=split, tile_y=yi, tile_x=xi,
                                         luma_std=round(sd, 2), entropy=round(en, 3),
                                         kept=0))
                        continue
                    stem = "%s__t%02d_%02d" % (Path(p).stem, yi, xi)
                    cv2.imwrite(str(idir / (stem + ".jpg")), t, [cv2.IMWRITE_JPEG_QUALITY, Q])
                    (ldir / (stem + ".txt")).write_text("", encoding="utf-8")  # 空标签=负样本
                    n_kept += 1
                    rows.append(dict(image=p.name, split=split, tile_y=yi, tile_x=xi,
                                     luma_std=round(sd, 2), entropy=round(en, 3), kept=1))
                    if len(preview_src[p.name]) < 4:
                        preview_src[p.name].append(t)
            log("    %s -> %d tiles (源图 %d/%d)" % (p.name, len(ys) * len(xs), k + 1, len(paths)))
    log("  保留 %d 个 tile，内容过滤丢弃 %d 个 (%.1f%%)"
        % (n_kept, n_drop, 100.0 * n_drop / max(1, n_kept + n_drop)))

    with open(out / "preview" / "tile_stats.csv", "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # ---------- 5. data.yaml / manifest ----------
    classes = ["VHBNM","VHBNL","SVHBNM","SVHBNL","SVHTNL","CBHPM","CBVPM",
               "RHTBNM","RHTBNL","GWCSBNM","GWCSBNL","GWCNM","GWCNL",
               "BSBM","INSD","DRPS"]
    (out / "data.yaml").write_text(
        "# 现场无缺陷负样本 v1（样本集，供检查）\n"
        "# 用途：修正 P(缺陷|部件出现) 先验；并作为缺陷合成的目标域底图\n"
        "# 标签全部为空文件 = 纯负样本\n"
        "path: %s\ntrain: images/train\nval: images/val\n\nnc: 16\nnames: %s\n"
        % (str(out).replace("\\", "/"), json.dumps(classes, ensure_ascii=False)),
        encoding="utf-8")

    man = dict(
        generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        source="data/Normal_dataset (检测车实拍，标签全空 = 无缺陷)",
        source_total=len(src), dropped_overlap_with_test770=len(src) - len(kept),
        sampled_images=len(picked),
        segments=dict(Counter(seg_of(p.name) for p in picked)),
        tile=TILE, stride=STRIDE, min_std=args.min_std, min_entropy=args.min_entropy,
        tiles_kept=n_kept, tiles_dropped=n_drop,
        train_images=len(splits["train"]), val_images=len(splits["val"]),
        note="全部标签为空文件；本目录为 CPU 生成的样本集，用于人工检查效果。"
    )
    (out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- 6. 预览 ----------
    log("步骤 4/5 生成预览")
    # P1: 原图缩略 + 它的 4 个 tile
    src_cards = []
    for name, tiles in list(preview_src.items())[:8]:
        p = NORMAL / "images" / "train" / name
        if not p.exists():
            p = NORMAL / "images" / "val" / name
        im = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if im is None:
            continue
        thumb = cv2.resize(im, (314, 314), interpolation=cv2.INTER_AREA)
        card = [thumb] + [cv2.resize(t, (314, 314), interpolation=cv2.INTER_AREA) for t in tiles[:4]]
        while len(card) < 5:
            card.append(np.full((314, 314, 3), 24, np.uint8))
        src_cards.append(np.hstack(card))
    if src_cards:
        cv2.imwrite(str(out / "preview" / "01_source_and_tiles.jpg"),
                    np.vstack(src_cards), [cv2.IMWRITE_JPEG_QUALITY, 88])

    # P2: 大量 tile 网格
    allt = []
    for split in ["train", "val"]:
        for f in sorted((out / "images" / split).glob("*.jpg"))[:300]:
            t = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if t is not None:
                allt.append(t)
        if len(allt) >= 96:
            break
    if allt:
        cv2.imwrite(str(out / "preview" / "02_tile_grid.jpg"),
                    montage(allt[:96], 10, cell=200), [cv2.IMWRITE_JPEG_QUALITY, 85])

    # P3: 域对比 —— 现场无缺陷 tile vs 检测车含缺陷 tile
    cmp_rows = []
    norm_t = allt[:6]
    def_t = []
    if DEFECT.is_dir():
        for f in sorted((DEFECT / "images").glob("*.jpg"))[:6]:
            im = cv2.imread(str(f), cv2.IMREAD_COLOR)
            if im is None:
                continue
            h, w = im.shape[:2]
            t = im[max(0, h // 2 - 640):h // 2 + 640, max(0, w // 2 - 640):w // 2 + 640]
            t = cv2.resize(t, (TILE // 2, TILE // 2), interpolation=cv2.INTER_AREA)
            def_t.append(t)
    if norm_t and def_t:
        n = min(len(norm_t), len(def_t))
        top = np.hstack([cv2.resize(t, (314, 314)) for t in norm_t[:n]])
        bot = np.hstack([cv2.resize(t, (314, 314)) for t in def_t[:n]])
        gap = np.full((12, top.shape[1], 3), 60, np.uint8)
        cv2.imwrite(str(out / "preview" / "03_domain_comparison.jpg"),
                    np.vstack([top, gap, bot]), [cv2.IMWRITE_JPEG_QUALITY, 85])

    log("步骤 5/5 完成 -> %s" % out)
    print()
    print("=" * 70)
    print("样本数据集已生成")
    print("=" * 70)
    print("  源图        : %d 张（分层抽样自 %d 张可用图）" % (len(picked), len(kept)))
    print("  切图        : %d 个保留 / %d 个内容过滤丢弃" % (n_kept, n_drop))
    print("  train/val   : %d / %d 张源图" % (len(splits["train"]), len(splits["val"])))
    print("  标签        : 全部为空文件（纯负样本）")
    print("  预览        : preview/01_source_and_tiles.jpg")
    print("                preview/02_tile_grid.jpg")
    print("                preview/03_domain_comparison.jpg（上=现场无缺陷 / 下=检测车含缺陷）")
    print("  统计        : preview/tile_stats.csv")
    print("  清单        : manifest.json")


if __name__ == "__main__":
    main()
