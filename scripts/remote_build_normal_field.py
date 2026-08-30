#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
远程全量管线：把 Normal_dataset（1000 张检测车实拍无缺陷图）建成训练数据集。

AI 手段：
  1. DINOv2 (timm, GPU) 自监督特征
  2. pHash 近重复检测
  3. k-center greedy 多样性选择（按区段分层保底）
  4. kNN 离群点检测 -> 输出需人工复核清单
  5. 内容过滤（信息量低的空 tile 丢弃）

工程：滑窗切图用多进程（实例有 208 核）。

输出：<out>/{images,labels}/{train,val} + data.yaml + manifest.json + stats + preview
"""
from __future__ import annotations

import argparse, csv, json, os, random, re, shutil, time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

random.seed(42); np.random.seed(42)

TILE, STRIDE, Q = 1280, 960, 92
CLASSES = ["VHBNM","VHBNL","SVHBNM","SVHBNL","SVHTNL","CBHPM","CBVPM",
           "RHTBNM","RHTBNL","GWCSBNM","GWCSBNL","GWCNM","GWCNL",
           "BSBM","INSD","DRPS"]

_W = {}   # worker 全局（切图用）


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def seg_of(n):
    m = re.match(r"^\d+_K(\d+)_(F\w+)-", n)
    return m.group(2) if m else "UNKNOWN"


def entropy(g):
    h = np.histogram(g, bins=256, range=(0, 256))[0].astype(np.float64)
    p = h / (h.sum() + 1e-9); nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())


def phash(im, size=32):
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (size, size), interpolation=cv2.INTER_AREA)
    d = cv2.dct(g.astype(np.float32))[:8, :8].flatten()
    return (d[1:] > np.median(d[1:])).astype(np.uint8)


def tile_positions(size, tile=TILE, stride=STRIDE):
    ps = list(range(0, max(1, size - tile + 1), stride))
    if ps and ps[-1] != size - tile:
        ps.append(size - tile)
    return ps


# ---------------- 切图 worker ----------------
def _init_worker(out, min_std, min_ent):
    _W["out"] = out; _W["min_std"] = min_std; _W["min_ent"] = min_ent


def _tile_one(job):
    src, split, stem_prefix = job
    out = _W["out"]; min_std = _W["min_std"]; min_ent = _W["min_ent"]
    idir = os.path.join(out, "images", split)
    ldir = os.path.join(out, "labels", split)
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    if im is None:
        return []
    h, w = im.shape[:2]
    ys = tile_positions(h); xs = tile_positions(w)
    rows = []
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            t = im[y:y + TILE, x:x + TILE]
            if t.shape[0] != TILE or t.shape[1] != TILE:
                continue
            g = cv2.cvtColor(t, cv2.COLOR_BGR2GRAY)
            sd, en = float(g.std()), entropy(g)
            keep = sd >= min_std and en >= min_ent
            if keep:
                stem = "%s__t%02d_%02d" % (stem_prefix, yi, xi)
                cv2.imwrite(os.path.join(idir, stem + ".jpg"), t,
                            [cv2.IMWRITE_JPEG_QUALITY, Q])
                open(os.path.join(ldir, stem + ".txt"), "w").close()
            rows.append(dict(image=os.path.basename(src), split=split,
                             ty=yi, tx=xi, luma_std=round(sd, 2),
                             entropy=round(en, 3), kept=int(keep),
                             luma_mean=round(float(g.mean()), 2)))
    return rows


# ---------------- 多样性选择 ----------------
def kcenter(feat, k, seed=None):
    n = len(feat)
    if k >= n:
        return list(range(n))
    sim = feat @ feat.T
    dist = 1.0 - sim
    s = int(np.argmax(dist.sum(1))) if seed is None else seed
    chosen = [s]; mind = dist[s].copy()
    for _ in range(k - 1):
        nxt = int(np.argmax(mind)); chosen.append(nxt)
        mind = np.minimum(mind, dist[nxt])
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal", default="/root/autodl-tmp/subway/data/Normal_dataset")
    ap.add_argument("--test-list", default="",
                    help="外部测试集文件名清单 csv（用于剔除重叠）")
    ap.add_argument("--out", default="/root/autodl-tmp/subway/data/normal_field_v1")
    ap.add_argument("--n-select", type=int, default=0, help="0=全选")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--min-std", type=float, default=8.0)
    ap.add_argument("--min-entropy", type=float, default=3.0)
    ap.add_argument("--feat-size", type=int, default=448)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--dup-hamming", type=int, default=-1,
                    help="pHash 汉明阈值，<=该值视为重复；-1=禁用。"
                         "实测：隧道场景 pHash 误判率>94%（判为重复的对 95% 公里标相差数千米），默认禁用")
    ap.add_argument("--no-dino", action="store_true")
    args = ap.parse_args()

    out = args.out
    if os.path.exists(out):
        log("清理已存在输出 %s" % out); shutil.rmtree(out)
    for s in ["images/train", "images/val", "labels/train", "labels/val",
              "stats", "preview"]:
        os.makedirs(os.path.join(out, s), exist_ok=True)

    # 1. 收集
    log("1/7 收集源图")
    src = []
    for sub in ["train", "val"]:
        d = Path(args.normal) / "images" / sub
        if d.is_dir():
            src += sorted(str(p) for p in d.glob("*.jpg"))
    log("   共 %d 张" % len(src))

    ext = set()
    if args.test_list and os.path.exists(args.test_list):
        with open(args.test_list, encoding="utf-8") as fp:
            ext = {r["image"] for r in csv.DictReader(fp)}
    kept = [p for p in src if os.path.basename(p) not in ext]
    log("   剔除与测试集重叠 %d 张 -> %d 张" % (len(src) - len(kept), len(kept)))

    # 2. 去重
    #    主键 = 区段 + 公里标（物理位置，可靠）；pHash 仅作二次过滤（几乎完全相同的图）
    log("2/7 去重（主键=区段+公里标，pHash 阈值<=%d）" % args.dup_hamming)
    seen_key, uniq, dup_key, dup_hash = set(), [], [], []
    hashes = {}
    for p in kept:
        n = os.path.basename(p)
        m = re.match(r"^\d+_K(\d+)_(F\w+)-", n)
        key = ("%s_K%s" % (m.group(2), m.group(1))) if m else ("NAME_" + n)
        if key in seen_key:
            dup_key.append((n, key))
            continue
        if args.dup_hamming >= 0:
            im = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_8)
            if im is None:
                continue
            h = phash(im)
            dup = next((q for q, hq in hashes.items()
                        if int(np.count_nonzero(h != hq)) <= args.dup_hamming), None)
            if dup:
                dup_hash.append((n, dup))
                continue
            hashes[n] = h
        seen_key.add(key)
        uniq.append(p)
    log("   同位置去重(公里标) 剔除 %d 张" % len(dup_key))
    log("   pHash 近似重复   剔除 %d 张" % len(dup_hash))
    log("   -> %d 张唯一" % len(uniq))
    segc = Counter(seg_of(os.path.basename(p)) for p in uniq)
    log("   区段分布 %s" % dict(segc.most_common()))

    # 3. DINOv2
    feat = None
    if not args.no_dino:
        log("3/7 DINOv2 特征 (GPU)")
        import torch, timm
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                                  pretrained=True, num_classes=0).eval().to(dev)
        cfg = timm.data.resolve_data_config(model.pretrained_cfg)
        tf = timm.data.create_transform(**cfg)
        fs, bs = [], 32
        for i in range(0, len(uniq), bs):
            arrs = []
            for p in uniq[i:i + bs]:
                # 1/4 降采样读取（1280px，足够做语义特征）
                im = cv2.imread(p, cv2.IMREAD_REDUCED_COLOR_4)
                im = cv2.cvtColor(cv2.resize(im, (args.feat_size, args.feat_size),
                                             interpolation=cv2.INTER_AREA),
                                  cv2.COLOR_BGR2RGB)
                arrs.append(tf(image=im).unsqueeze(0))
            with torch.no_grad():
                f = model(torch.cat(arrs).to(dev)).float().cpu().numpy()
            fs.append(f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9))
            if (i // bs) % 5 == 0:
                log("   %d/%d" % (min(i + bs, len(uniq)), len(uniq)))
        feat = np.vstack(fs)
        log("   特征 %s" % (feat.shape,))
    else:
        log("3/7 跳过 DINOv2")

    # 4. 选择
    log("4/7 多样性选择")
    if args.n_select <= 0 or args.n_select >= len(uniq):
        sel_idx = list(range(len(uniq)))
    elif feat is None:
        idx = list(range(len(uniq))); random.shuffle(idx)
        sel_idx = sorted(idx[:args.n_select])
    else:
        by_seg = defaultdict(list)
        for i, p in enumerate(uniq):
            by_seg[seg_of(os.path.basename(p))].append(i)
        sel_idx, left = [], args.n_select
        for seg, idxs in sorted(by_seg.items(), key=lambda x: -len(x[1])):
            t = min(2, len(idxs)); sel_idx += idxs[:t]; left -= t
        rest = [i for i in range(len(uniq)) if i not in set(sel_idx)]
        if left > 0 and rest:
            sel_idx += [rest[c] for c in kcenter(feat[rest], min(left, len(rest)))]
        sel_idx = sorted(set(sel_idx))
    log("   选出 %d 张" % len(sel_idx))

    # 5. 离群点
    flagged = []
    if feat is not None:
        sim = feat @ feat.T; np.fill_diagonal(sim, -1)
        knn = np.sort(sim, 1)[:, -5:].mean(1)
        thr = np.percentile(knn, 2)
        flagged = sorted([(os.path.basename(uniq[i]), float(knn[i]))
                          for i in range(len(uniq)) if knn[i] <= thr],
                         key=lambda x: x[1])
    log("   离群 %d 张（需人工复核）" % len(flagged))

    # 6. 分层切分
    by_seg = defaultdict(list)
    for i in sel_idx:
        by_seg[seg_of(os.path.basename(uniq[i]))].append(uniq[i])
    tr, va = [], []
    for seg, ps in by_seg.items():
        ps = ps[:]; random.shuffle(ps)
        nv = max(1, int(round(len(ps) * args.val_frac))) if len(ps) > 3 else 0
        va += ps[:nv]; tr += ps[nv:]
    log("5/7 切分 train %d / val %d" % (len(tr), len(va)))

    # 7. 并行切图
    log("6/7 并行切图 (workers=%d)" % args.workers)
    jobs = [(p, "train", Path(p).stem) for p in tr] + \
           [(p, "val", Path(p).stem) for p in va]
    t0 = time.time()
    with Pool(args.workers, initializer=_init_worker,
              initargs=(out, args.min_std, args.min_entropy)) as pool:
        res = pool.map(_tile_one, jobs, chunksize=1)
    rows = [r for rr in res for r in rr]
    nk = sum(r["kept"] for r in rows)
    nd = len(rows) - nk
    log("   保留 %d / 丢弃 %d (%.1f%%)  用时 %.1fs"
        % (nk, nd, 100.0 * nd / max(1, len(rows)), time.time() - t0))

    with open(os.path.join(out, "stats", "tiles.csv"), "w",
              encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(out, "stats", "flagged_for_review.txt"), "w",
              encoding="utf-8") as fp:
        fp.write("\n".join("%s\t%.4f" % (a, b) for a, b in flagged))

    # data.yaml / manifest
    with open(os.path.join(out, "data.yaml"), "w", encoding="utf-8") as fp:
        fp.write("# 现场无缺陷负样本 v1 —— 目标域负样本 / 缺陷合成底图\n"
                 "# 标签全部为空文件 = 纯负样本\n"
                 "path: %s\ntrain: images/train\nval: images/val\n\nnc: 16\n"
                 "names: %s\n" % (out, json.dumps(CLASSES, ensure_ascii=False)))
    man = dict(generated_at=time.strftime("%Y-%m-%d %H:%M:%S"),
               source=args.normal, source_total=len(src),
               dropped_overlap=len(src) - len(kept),
               dropped_same_position=len(dup_key), dropped_phash=len(dup_hash),
               unique_images=len(uniq), selected_images=len(sel_idx),
               segments=dict(Counter(seg_of(os.path.basename(uniq[i]))
                                     for i in sel_idx)),
               tile=TILE, stride=STRIDE, min_std=args.min_std,
               min_entropy=args.min_entropy, tiles_kept=nk, tiles_dropped=nd,
               train_images=len(tr), val_images=len(va),
               flagged=len(flagged),
               note="全部标签为空文件（纯负样本）")
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as fp:
        json.dump(man, fp, ensure_ascii=False, indent=1)

    # 预览
    log("7/7 生成预览")
    def mont(items, cols, cell=256):
        rowsn = (len(items) + cols - 1) // cols
        canvas = np.full((rowsn * cell, cols * cell, 3), 28, np.uint8)
        for i, t in enumerate(items):
            r, c = divmod(i, cols)
            x = cv2.resize(t, (cell - 6, cell - 6), interpolation=cv2.INTER_AREA)
            canvas[r * cell + 3:r * cell + 3 + x.shape[0],
                   c * cell + 3:c * cell + 3 + x.shape[1]] = x
        return canvas
    pool_dir = os.path.join(out, "images", "train")
    fs = sorted(os.listdir(pool_dir))[:120]
    if fs:
        ims = [cv2.imread(os.path.join(pool_dir, f)) for f in fs]
        ims = [i for i in ims if i is not None]
        if ims:
            for nm, n in [("grid_120tiles.jpg", 120), ("grid_48tiles.jpg", 48)]:
                cv2.imwrite(os.path.join(out, "preview", nm),
                            mont(ims[:n], 10 if n > 60 else 8),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
    # 低信息量被丢弃的样例（供核对过滤阈值是否合理）
    bad = sorted(rows, key=lambda r: r["luma_std"])[:24]
    src_bad = {}
    for r in bad:
        src_bad.setdefault(r["image"], []).append(r)
    cells = []
    for name, rs in list(src_bad.items())[:4]:
        p = next((q for q in uniq if os.path.basename(q) == name), None)
        if not p:
            continue
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:
            continue
        for r in rs[:2]:
            ys = tile_positions(im.shape[0]); xs = tile_positions(im.shape[1])
            t = im[ys[r["ty"]]:ys[r["ty"]] + TILE, xs[r["tx"]]:xs[r["tx"]] + TILE]
            if t.shape[:2] == (TILE, TILE):
                cells.append(t)
    if cells:
        cv2.imwrite(os.path.join(out, "preview", "dropped_lowinfo.jpg"),
                    mont(cells, 4, cell=300), [cv2.IMWRITE_JPEG_QUALITY, 85])

    log("完成 -> %s" % out)
    print(json.dumps(man, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
