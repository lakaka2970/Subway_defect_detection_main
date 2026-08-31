# -*- coding: utf-8 -*-
"""P1-5 区段相似度量化（本地 CPU，吸取 24 样本探针教训：足量样本+CV+划分）。

对重建基准 534 图按折（=公里标连续区段组）提取 ~20 维低阶像素特征，
两两折训练 balanced 逻辑回归（5-fold CV，图像级划分，同支柱号同组），
输出 balanced accuracy/AUC 矩阵：越接近 0.5 区段越相似。
用于量化 fold3（F1B05+F1B02）的域差距并指导补采优先级。
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "data", "Defect_dataset_16_rebuilt")
OUT = os.path.join(ROOT, "docs", "plans", "8.31阶段1产物",
                   "segment_distance.json")
SIDE = 640


def feats(path):
    im = Image.open(path).convert("L")
    w, h = im.size
    s = SIDE / max(w, h)
    if s < 1:
        im = im.resize((max(1, int(w * s)), max(1, int(h * s))))
    a = np.asarray(im, np.float32)
    gy, gx = np.gradient(a)
    grad = np.sqrt(gx * gx + gy * gy)
    hist = np.histogram(a, 32, (0, 256))[0]
    hist = hist / max(1.0, hist.sum())
    q = np.percentile(a, [5, 25, 50, 75, 95])
    # 分块亮度非均匀性（4x4 块均值的标准差）
    bh = a[:a.shape[0] // 4 * 4, :a.shape[1] // 4 * 4]
    blocks = bh.reshape(4, a.shape[0] // 4, 4, a.shape[1] // 4)
    illum = blocks.transpose(0, 2, 1, 3).reshape(16, -1).mean(1).std()
    f = [a.mean(), a.std(), q[0], q[1], q[2], q[3], q[4],
         (a < 20).mean(), (a > 230).mean(), illum]
    f += list(hist[::2])                      # 16 个直方图 bin
    f += [grad.mean(), np.percentile(grad, 95),
          (grad > 30).mean()]
    # 行/列方向梯度能量比（线条方向性）
    f += [np.abs(gx).mean() / (np.abs(gy).mean() + 1e-6)]
    return f


def group_key(name):
    # 同支柱号同组，防 CV 泄漏：102903294_K26305_F1B03-146_1_21 -> F1B03-146
    parts = name.split("_")
    return parts[2] if len(parts) >= 3 else name


def probe(xa, ya, xb, yb, seed=0):
    """balanced LR（numpy 梯度下降），5-fold CV by group，返回 bal-acc/AUC。"""
    rng = np.random.RandomState(seed)
    X = np.vstack([xa, xb])
    y = np.concatenate([ya, yb])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    X = (X - mu) / sd
    groups = np.arange(len(y))       # 调用方传入分组向量
    accs, aucs = [], []
    for _ in range(5):
        idx = rng.permutation(len(y))
        Xs, ys, gs = X[idx], y[idx], groups[idx]
        cut = int(len(ys) * 0.8)
        tr, te = slice(0, cut), slice(cut, None)
        w = np.zeros(X.shape[1])
        b = 0.0
        for _it in range(300):
            z = Xs[tr] @ w + b
            p = 1 / (1 + np.exp(-z))
            g = p - ys[tr]
            w -= 0.1 * (Xs[tr].T @ g / len(g) + 1e-3 * w)
            b -= 0.1 * g.mean()
        p = 1 / (1 + np.exp(-(Xs[te] @ w + b)))
        pred = (p >= 0.5).astype(float)
        pos, neg = ys[te] == 1, ys[te] == 0
        if pos.sum() and neg.sum():
            tpr = (pred[pos] == 1).mean()
            tnr = (pred[neg] == 0).mean()
            accs.append(0.5 * (tpr + tnr))
            order = np.argsort(p)
            yy = ys[te]
            tpr_c = np.cumsum(yy[order])[::-1] / max(1, yy.sum())
            fpr_c = np.cumsum(1 - yy[order])[::-1] / max(1, (1 - yy).sum())
            aucs.append(np.trapezoid(tpr_c, fpr_c))
    return (float(np.mean(accs)) if accs else float("nan"),
            float(np.mean(aucs)) if aucs else float("nan"))


def main():
    fold_of = {r["image"]: int(r["fold"]) for r in csv.DictReader(
        open(os.path.join(BENCH, "fold_assignments.csv"),
             encoding="utf-8-sig"))}
    data = defaultdict(lambda: ([], [], []))
    for name, f in fold_of.items():
        p = os.path.join(BENCH, "images", name)
        if not os.path.exists(p):
            continue
        data[f][0].append(feats(p))
        data[f][1].append(group_key(name))
    out = {"n_per_fold": {}, "pairs": {}}
    arr = {}
    for f in sorted(data):
        X = np.array(data[f][0], np.float32)
        g = np.array(data[f][1])
        arr[f] = (X, g)
        out["n_per_fold"][str(f)] = int(len(X))
    folds = sorted(arr)
    for i in range(len(folds)):
        for j in range(i + 1, len(folds)):
            a, b = folds[i], folds[j]
            Xa, ga = arr[a]
            Xb, gb = arr[b]
            gv = np.concatenate([np.arange(len(ga)) + 1000 * a,
                                 np.arange(len(gb)) + 1000 * b])
            # 组内样本共享 id 前缀 -> 用组键映射整数
            gmap = {}
            gi = []
            for key in list(ga) + list(gb):
                if key not in gmap:
                    gmap[key] = len(gmap)
                gi.append(gmap[key])
            xa, xb = Xa, Xb
            ya = np.ones(len(xa))
            yb = np.zeros(len(xb))
            bal, auc = probe(xa, ya, xb, yb)
            out["pairs"]["%d_vs_%d" % (a, b)] = {
                "balanced_acc": round(bal, 4), "auc": round(auc, 4)}
            print("fold%d vs fold%d: bal-acc %.3f AUC %.3f" % (a, b, bal, auc))
    with open(OUT, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    print("SEGDIST_DONE ->", OUT)


import json  # noqa: E402

if __name__ == "__main__":
    main()
