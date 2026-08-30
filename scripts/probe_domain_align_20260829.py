# -*- coding: utf-8 -*-
"""
离线像素域对齐可行性实测（2026-08-29）

问题：用离线手段把 Defect_dataset_2（车间）的图片参数改到贴近 Defect_dataset（检测车实拍），
      能把两域的可分性压到多低？

做法：
  1. 车间图 / 检测车图 各自 square-crop + resize 到 640x640（消除几何差异，只留像素风格差异）
  2. 用同一套特征提取器为两域抽 ~30 维低阶特征
  3. 训域探针（balanced logistic regression），得到 baseline 可分性
  4. 对车间图施加"标定驱动的确定性退化管线"：灰度化 -> 亮度直方图匹配 -> 模糊 -> 噪声
     （模糊 sigma 与噪声 sigma 由网格搜索在标定子集上联合标定）
  5. 重算特征、重训探针，比较可分性下降幅度，并逐特征报告"域差距弥合率"
  6. 分组探针：纯风格特征 / 结构退化特征 / 全部特征，分别报告

只做读操作；退化后的样例图写到 output/domain_align_probe_20260829/samples/
"""
import os, sys, json, math, random, csv
import numpy as np
import cv2

random.seed(42)
np.random.seed(42)

ROOT = r"E:\Work\Subway_defect_detection_main"
WS_DIR = os.path.join(ROOT, "data", "Defect_dataset_2", "Defect_dataset", "images")
ACT_DIR = os.path.join(ROOT, "data", "Defect_dataset", "images")
OUT_DIR = os.path.join(ROOT, "output", "domain_align_probe_20260829")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "samples"), exist_ok=True)

N_WS = 300
N_ACT = 400
ANALYZE = 640          # 与 8.28 分析一致：最长边 640
CALIB_N = 60           # 标定子集


# ---------------- 特征提取 ----------------
def luma(bgr):
    return (0.114 * bgr[:, :, 0] + 0.587 * bgr[:, :, 1] + 0.299 * bgr[:, :, 2])


def colorfulness(bgr):
    b, g, r = bgr[:, :, 0].astype(np.float32), bgr[:, :, 1].astype(np.float32), bgr[:, :, 2].astype(np.float32)
    rg = r - g
    yb = 0.5 * (r + g) - b
    return float(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))


def fft_high_ratio(gray):
    f = np.fft.fftshift(np.fft.fft2(gray.astype(np.float32)))
    mag = np.abs(f)
    h, w = mag.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / (min(h, w) / 2.0)
    hi = mag[r > 0.5].sum()
    return float(hi / (mag.sum() + 1e-9))


def extract(bgr):
    y = luma(bgr)
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    f = {}
    f["luma_mean"] = float(y.mean()); f["luma_std"] = float(y.std())
    ps = np.percentile(y, [1, 5, 50, 95, 99])
    for i, k in enumerate(["luma_p01", "luma_p05", "luma_p50", "luma_p95", "luma_p99"]):
        f[k] = float(ps[i])
    f["dark_frac"] = float((y < 25).mean())
    f["bright_frac"] = float((y > 230).mean())
    f["clip_frac"] = float(((y >= 254) | (y <= 1)).mean())
    f["sat_mean"] = float(hsv[:, :, 1].mean()); f["sat_std"] = float(hsv[:, :, 1].std())
    f["colorfulness"] = colorfulness(bgr)
    e = cv2.Canny(g, 50, 150)
    f["edge_density"] = float((e > 0).mean())
    sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3); sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    gm = np.sqrt(sx ** 2 + sy ** 2)
    f["grad_mean"] = float(gm.mean()); f["grad_p95"] = float(np.percentile(gm, 95))
    f["laplacian_var"] = float(cv2.Laplacian(g, cv2.CV_64F).var())
    hist = np.histogram(g, bins=256, range=(0, 256))[0].astype(np.float64)
    p = hist / (hist.sum() + 1e-9); nz = p[p > 0]
    f["entropy"] = float(-(nz * np.log2(nz)).sum())
    f["fft_high_ratio"] = fft_high_ratio(g)
    bh, bw = g.shape[0] // 8, g.shape[1] // 8
    blocks = g[:bh * 8, :bw * 8].reshape(8, bh, 8, bw).mean(axis=(1, 3))
    f["illum_nonuniformity"] = float(blocks.std())
    h16 = np.histogram(g, bins=16, range=(0, 256))[0].astype(np.float64)
    h16 = h16 / (h16.sum() + 1e-9)
    for i in range(16):
        f["hist_%02d" % i] = float(h16[i])
    return f


FEAT_KEYS = None
STYLE_KEYS = ["luma_mean", "luma_std", "luma_p01", "luma_p05", "luma_p50", "luma_p95", "luma_p99",
              "dark_frac", "bright_frac", "clip_frac", "sat_mean", "sat_std", "colorfulness",
              "entropy", "illum_nonuniformity"] + ["hist_%02d" % i for i in range(16)]
STRUCT_KEYS = ["edge_density", "grad_mean", "grad_p95", "laplacian_var", "fft_high_ratio"]


# ---------------- 载入与预处理 ----------------
def prep(path, size=ANALYZE):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    h, w = im.shape[:2]
    s = min(h, w)
    im = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)


def sample_dir(d, n):
    fs = [x for x in os.listdir(d) if x.lower().endswith((".jpg", ".jpeg", ".png"))]
    random.shuffle(fs)
    return [os.path.join(d, x) for x in fs[:n]]


# ---------------- 退化管线 ----------------
def hist_match(src_bgr, ref_cdf):
    """按通道做直方图匹配到参考累积分布 ref_cdf (256,)"""
    out = np.empty_like(src_bgr)
    for c in range(3):
        s = src_bgr[:, :, c]
        sh = np.histogram(s, bins=256, range=(0, 256))[0].astype(np.float64)
        sc = (sh.cumsum()) / (sh.sum() + 1e-9)
        lut = np.interp(sc, ref_cdf, np.arange(256)).astype(np.uint8)
        out[:, :, c] = lut[s]
    return out


def degrade(bgr, ref_cdf, blur_sigma, noise_sigma):
    im = bgr
    im = hist_match(im, ref_cdf)                       # 亮度/对比度/暗部占比/熵
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)           # 灰度：饱和度/色彩丰富度 -> 0
    if blur_sigma > 0:
        k = int(blur_sigma * 6) // 2 * 2 + 1
        g = cv2.GaussianBlur(g, (k, k), blur_sigma)    # 模糊：laplacian / 边缘密度 / 梯度
    if noise_sigma > 0:
        g = g.astype(np.float32)
        g = g + np.random.normal(0, noise_sigma, g.shape)   # 噪声：fft_high_ratio / 熵
        g = np.clip(g, 0, 255).astype(np.uint8)
    return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)


def vec(fs, keys):
    return np.array([[f[k] for k in keys] for f in fs], dtype=np.float64)


def probe_acc(a, b, keys):
    """balanced logistic regression, 5-fold CV"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    X = np.vstack([vec(a, keys), vec(b, keys)])
    y = np.array([0] * len(a) + [1] * len(b))
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    return float(cross_val_score(clf, X, y, cv=cv, scoring="balanced_accuracy").mean())


def main():
    print("[1/6] 载入图像 ...")
    ws_paths = sample_dir(WS_DIR, N_WS)
    act_paths = sample_dir(ACT_DIR, N_ACT)
    ws = [prep(p) for p in ws_paths]
    act = [prep(p) for p in act_paths]
    ws = [x for x in ws if x is not None]
    act = [x for x in act if x is not None]
    print("     车间 %d 张 / 检测车 %d 张（统一 %dx%d 方形）" % (len(ws), len(act), ANALYZE, ANALYZE))

    print("[2/6] 抽取特征 ...")
    f_ws = [extract(x) for x in ws]
    f_act = [extract(x) for x in act]
    global FEAT_KEYS
    FEAT_KEYS = list(f_ws[0].keys())
    print("     特征维度 %d" % len(FEAT_KEYS))

    print("[3/6] 基线域探针 ...")
    base = {
        "all": probe_acc(f_ws, f_act, FEAT_KEYS),
        "style": probe_acc(f_ws, f_act, STYLE_KEYS),
        "struct": probe_acc(f_ws, f_act, STRUCT_KEYS),
    }
    print("     baseline  balanced acc: all=%.4f  style=%.4f  struct=%.4f"
          % (base["all"], base["style"], base["struct"]))

    print("[4/6] 标定参考累积分布（来自检测车域）...")
    ref_luts = []
    for im in act[:200]:
        for c in range(3):
            h = np.histogram(im[:, :, c], bins=256, range=(0, 256))[0].astype(np.float64)
            ref_luts.append(h.cumsum() / (h.sum() + 1e-9))
    ref_cdf = np.median(np.array(ref_luts), axis=0)
    ref_cdf = ref_cdf / ref_cdf[-1]

    print("[5/6] 联合标定 blur_sigma / noise_sigma ...")
    tgt = {k: float(np.median([f[k] for f in f_act])) for k in STRUCT_KEYS}
    cur0 = {k: float(np.median([f[k] for f in f_ws])) for k in STRUCT_KEYS}
    print("     目标(检测车): laplacian_var=%.1f edge_density=%.4f fft_high=%.4f grad_mean=%.2f"
          % (tgt["laplacian_var"], tgt["edge_density"], tgt["fft_high_ratio"], tgt["grad_mean"]))
    print("     当前(车间)  : laplacian_var=%.1f edge_density=%.4f fft_high=%.4f grad_mean=%.2f"
          % (cur0["laplacian_var"], cur0["edge_density"], cur0["fft_high_ratio"], cur0["grad_mean"]))

    calib_idx = list(range(min(CALIB_N, len(ws))))
    best = None
    for bs in [0.0, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.2]:
        for ns in [0.0, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0, 11.0]:
            fs = [extract(degrade(ws[i], ref_cdf, bs, ns)) for i in calib_idx]
            cur = {k: float(np.median([f[k] for f in fs])) for k in STRUCT_KEYS}
            rel = lambda k: abs(cur[k] - tgt[k]) / (abs(tgt[k]) + 1e-9)
            cost = (rel("laplacian_var") + rel("edge_density") + rel("fft_high_ratio") + rel("grad_mean")) / 4.0
            if best is None or cost < best[0]:
                best = (cost, bs, ns)
    _, BLUR, NOISE = best
    print("     最优参数: blur_sigma=%.2f  noise_sigma=%.2f  (标定代价 %.4f)" % (BLUR, NOISE, best[0]))

    print("[6/6] 施加退化并重测 ...")
    f_ws_d = [extract(degrade(ws[i], ref_cdf, BLUR, NOISE)) for i in range(len(ws))]
    after = {
        "all": probe_acc(f_ws_d, f_act, FEAT_KEYS),
        "style": probe_acc(f_ws_d, f_act, STYLE_KEYS),
        "struct": probe_acc(f_ws_d, f_act, STRUCT_KEYS),
    }
    print("     degraded    balanced acc: all=%.4f  style=%.4f  struct=%.4f"
          % (after["all"], after["style"], after["struct"]))

    # 逐特征弥合率
    rows = []
    for k in FEAT_KEYS:
        a0 = float(np.median([f[k] for f in f_ws]))
        b = float(np.median([f[k] for f in act and f_act]))
        a1 = float(np.median([f[k] for f in f_ws_d]))
        gap = abs(a0 - b)
        closed = (1.0 - abs(a1 - b) / gap) * 100.0 if gap > 1e-12 else 100.0
        rows.append({"feature": k, "workshop": a0, "degraded": a1, "actual": b,
                     "gap_closed_pct": max(-100.0, min(100.0, closed))})

    # 样例图
    for i in range(6):
        cv2.imwrite(os.path.join(OUT_DIR, "samples", "ws_%02d_orig.jpg" % i), ws[i])
        cv2.imwrite(os.path.join(OUT_DIR, "samples", "ws_%02d_degraded.jpg" % i),
                    degrade(ws[i], ref_cdf, BLUR, NOISE))
    for i in range(6):
        cv2.imwrite(os.path.join(OUT_DIR, "samples", "act_%02d.jpg" % i), act[i])

    out = {
        "n_workshop": len(ws), "n_actual": len(act), "analyze_size": ANALYZE,
        "params": {"blur_sigma": BLUR, "noise_sigma": NOISE, "calib_cost": best[0]},
        "baseline_balanced_acc": base,
        "degraded_balanced_acc": after,
        "targets": tgt,
        "per_feature": rows,
        "style_keys": STYLE_KEYS, "struct_keys": STRUCT_KEYS,
    }
    with open(os.path.join(OUT_DIR, "result.json"), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT_DIR, "per_feature.csv"), "w", encoding="utf-8-sig", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print()
    print("=" * 78)
    print("结果汇总")
    print("=" * 78)
    print("域可分性（balanced accuracy，5-fold CV）")
    print("  特征组          基线      离线退化后    下降")
    for k in ["all", "style", "struct"]:
        print("  %-12s   %.4f    %.4f      %+.1f pp" % (k, base[k], after[k], (after[k] - base[k]) * 100))
    print()
    print("结构类特征的目标达成情况")
    for k in STRUCT_KEYS:
        a0 = float(np.median([f[k] for f in f_ws])); a1 = float(np.median([f[k] for f in f_ws_d]))
        print("  %-18s 车间 %10.4f -> 退化后 %10.4f   目标 %10.4f" % (k, a0, a1, tgt[k]))
    print()
    print("产物:", OUT_DIR)


if __name__ == "__main__":
    main()
