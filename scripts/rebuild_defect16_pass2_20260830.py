# -*- coding: utf-8 -*-
"""第2遍：修复 90 张 MISMATCH 图，使重建标签与归档 GT 计数逐图一致。

假设与依据：
  - 归档 gt_miss_taxonomy.csv 是 16 类基准的权威逐图类别计数；
  - box_features.csv（actual 域）保存每个 GT 的原图像素宽高（无中心坐标）；
  - 重标注相对旧 7 类的差异主要是"垂直悬吊误判纠正"：部分旧 0-4 类框被改判为
    CBHPM/CBVPM（几何近似保留），另有少量旧框被删除。
恢复策略（纯几何尺寸匹配，不使用任何模型预测）：
  1. 缺失框：从该图被删的 0-4 类旧框中，按相对宽高最接近者恢复，改标为期望类；
  2. 多余框：按期望框尺寸在旧保留框中做最优子集匹配，删除未匹配者。
产出：覆盖 data/Defect_dataset_16_rebuilt/labels/，并写第二遍审计。
"""
import csv
import json
import os
import re
from collections import Counter, defaultdict

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "docs", "plans", "8.28泛化性训练")
OUT = os.path.join(ROOT, "data", "Defect_dataset_16_rebuilt")
IMG_DIR = os.path.join(ROOT, "data", "Defect_dataset", "images")
LAB_DIR = os.path.join(ROOT, "data", "Defect_dataset", "labels")

NAMES16 = ["VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
           "RHTBNM", "RHTBNL", "GWCSBNM", "GWCSBNL", "GWCNM", "GWCNL",
           "BSBM", "INSD", "DRPS"]
CID = {n: i for i, n in enumerate(NAMES16)}
KEEP, VHB = {5, 6}, {0, 1, 2, 3, 4}
TOL = 0.20  # 相对尺寸匹配容差


def index_pool(pattern):
    import glob
    d = {}
    for p in glob.glob(os.path.join(ROOT, pattern), recursive=True):
        d[os.path.basename(p)] = p
    return d


def parse_label(path):
    boxes = []
    if not os.path.exists(path):
        return boxes
    for ln in open(path, encoding="utf-8"):
        f = ln.split()
        if len(f) == 5:
            boxes.append([int(f[0])] + [float(x) for x in f[1:]])
    return boxes


def main():
    audit = json.load(open(os.path.join(OUT, "rebuild_audit.json"),
                           encoding="utf-8"))
    rows = list(csv.DictReader(open(os.path.join(OUT, "rebuild_audit.csv"),
                                  encoding="utf-8-sig")))
    mism = [r for r in rows if r["status"] == "MISMATCH"]
    print("MISMATCH 图: %d" % len(mism))

    # 归档期望：逐图逐类计数 + 逐框尺寸
    tax = list(csv.DictReader(open(
        os.path.join(BASE, "domain_shift_analysis_20260828",
                     "gt_miss_taxonomy.csv"), encoding="utf-8-sig")))
    expect = defaultdict(Counter)
    for r in tax:
        expect[r["image"]][r["gt_class"]] += 1
    bf = list(csv.DictReader(open(
        os.path.join(BASE, "domain_shift_analysis_20260828",
                     "box_features.csv"), encoding="utf-8-sig")))
    gt_size = defaultdict(lambda: defaultdict(list))  # image -> {class -> [(w,h),...]}
    for r in bf:
        if r["domain"] != "actual":
            continue
        gt_size[r["image"]].setdefault(r["class_name"], []).append(
            (float(r["box_width_px"]), float(r["box_height_px"])))

    img_pool = index_pool(r"data\Defect_dataset\images\**\*.jpg")
    lab_pool = {}
    import glob
    for p in glob.glob(os.path.join(LAB_DIR, "**", "*.txt"), recursive=True):
        lab_pool[os.path.splitext(os.path.basename(p))[0]] = p

    stats = Counter()
    detail = []
    for r in mism:
        name = r["image"]
        stem = os.path.splitext(name)[0]
        exp = {k: v for k, v in expect.get(name, Counter()).items() if v}
        old = parse_label(lab_pool.get(stem, ""))
        ip = img_pool.get(name)
        if ip is None or not old and not exp:
            stats["skip_no_source"] += 1
            continue
        W, H = Image.open(ip).size

        keep_old = [b for b in old if b[0] in KEEP]       # 旧 5/6 类框
        vhb_old = [b for b in old if b[0] in VHB]          # 被删的垂直悬吊框
        final = []
        rec, drop, unfilled = 0, 0, 0
        for cname in ("CBHPM", "CBVPM"):
            n_exp = exp.get(cname, 0)
            same = [b for b in keep_old if NAMES16[b[0]] == cname]
            sizes = gt_size.get(name, {}).get(cname, [])
            # 相对尺寸
            exp_rel = [(w / W, h / H) for w, h in sizes]
            if len(same) > n_exp:
                # 多余：保留与期望尺寸最匹配的，其余删除
                cands = list(same)
                chosen = []
                for rel in exp_rel[:n_exp]:
                    best, bd = None, 9.9
                    for b in cands:
                        d = abs(b[3] - rel[0]) + abs(b[4] - rel[1])
                        if d < bd:
                            bd, best = d, b
                    if best is not None:
                        chosen.append(best)
                        cands.remove(best)
                drop += len(same) - len(chosen)
                final.extend(chosen)
            elif len(same) < n_exp:
                final.extend(same)
                need = n_exp - len(same)
                pool = list(vhb_old)
                for rel in exp_rel:
                    if need <= 0:
                        break
                    best, bd = None, 9.9
                    for b in pool:
                        d = abs(b[3] - rel[0]) + abs(b[4] - rel[1])
                        if d < bd:
                            bd, best = d, b
                    if best is not None and bd <= TOL * 2:
                        nb = [CID[cname]] + best[1:]
                        final.append(nb)
                        pool.remove(best)
                        vhb_old.remove(best)
                        rec += 1
                        need -= 1
                unfilled += need
            else:
                final.extend(same)
        # 写标签
        lines = ["%d %.6f %.6f %.6f %.6f" % tuple(b) for b in final]
        with open(os.path.join(OUT, "labels", stem + ".txt"), "w",
                  encoding="utf-8") as fp:
            fp.write("\n".join(lines) + ("\n" if lines else ""))
        act = Counter(NAMES16[b[0]] for b in final)
        ok = all(act.get(c, 0) == n for c, n in exp.items()) and \
            sum(act.values()) == sum(exp.values())
        stats["fixed" if ok else "still_mismatch"] += 1
        stats["boxes_recovered_from_vhb"] += rec
        stats["boxes_dropped_extra"] += drop
        stats["boxes_unfilled"] += unfilled
        detail.append({"image": name, "exp": dict(exp), "act": dict(act),
                       "recovered": rec, "dropped": drop,
                       "unfilled": unfilled, "ok": ok})

    with open(os.path.join(OUT, "pass2_audit.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": dict(stats), "detail": detail}, f,
                  ensure_ascii=False, indent=1)
    print("第二遍统计:", dict(stats))
    bad = [d for d in detail if not d["ok"]]
    print("仍不一致: %d 张" % len(bad))
    for d in bad[:10]:
        print("  %s exp=%s act=%s unfilled=%d" %
              (d["image"], d["exp"], d["act"], d["unfilled"]))


if __name__ == "__main__":
    main()
