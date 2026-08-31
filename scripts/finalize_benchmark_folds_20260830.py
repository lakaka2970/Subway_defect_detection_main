# -*- coding: utf-8 -*-
"""定稿基准：剔除 35 张冲突图，按区段/公里标设计 4 折 LOSO，写最终清单。

折设计原则：
  - 原子组 = 区段；F1B04 图像最多，按公里标中位数拆成两半；
  - 折 = [F1B04a, F1B04b, F1B03, F1B05+F1B02]；
  - GT=0 图（含 34 张干净 Normal）按公里标就近并入折，作为误报测试床；
  - 每折测试集 = 该折全部图（含 GT=0），训练时该折图像不进梯度。
"""
import csv
import glob
import json
import os
import re
import statistics
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "Defect_dataset_16_rebuilt")
BASE = os.path.join(ROOT, "docs", "plans", "8.28泛化性训练")
PAT = re.compile(r"^(\d+)_K(\d+)_(F1[AB]\d+)-(\d+[a-z]?)_(\d+)_(\d+)\.jpg$")


def main():
    p2 = json.load(open(os.path.join(OUT, "pass2_audit.json"), encoding="utf-8"))
    # 冲突 35 张：Normal 来源且 MISMATCH
    rows = list(csv.DictReader(open(os.path.join(OUT, "rebuild_audit.csv"),
                                  encoding="utf-8-sig")))
    conflict = [r["image"] for r in rows
                if r["source"] == "Normal_dataset" and r["status"] == "MISMATCH"]
    conf_set = set(conflict)

    # 从基准目录删除冲突图
    for name in conflict:
        for sub in ("images", "labels"):
            p = os.path.join(OUT, sub, os.path.splitext(name)[0] +
                             (".jpg" if sub == "images" else ".txt"))
            if os.path.exists(p):
                os.remove(p)

    # 期望 GT
    tax = list(csv.DictReader(open(
        os.path.join(BASE, "domain_shift_analysis_20260828",
                     "gt_miss_taxonomy.csv"), encoding="utf-8-sig")))
    expect = defaultdict(Counter)
    for r in tax:
        expect[r["image"]][r["gt_class"]] += 1

    # 基准图清单 = OUT/images 现存
    bench = sorted(os.path.basename(p) for p in
                   glob.glob(os.path.join(OUT, "images", "*.jpg")))
    print("最终基准: %d 张（剔除冲突 %d 张）" % (len(bench), len(conflict)))

    # 分组
    group = {}
    for name in bench:
        m = PAT.match(name)
        if m:
            group[name] = (m.group(3), int(m.group(2)))

    # F1B04 中位数拆分
    f1b04 = sorted(k for s, k in group.values() if s == "F1B04")
    med = int(statistics.median(f1b04)) if f1b04 else 0

    def fold_of(name):
        g = group.get(name)
        if g is None:
            return None
        seg, k = g
        if seg == "F1B04":
            return 0 if k <= med else 1
        if seg == "F1B03":
            return 2
        return 3  # F1B05 / F1B02

    folds = {0: [], 1: [], 2: [], 3: []}
    ungrouped = []
    for name in bench:
        f = fold_of(name)
        if f is None:
            ungrouped.append(name)
        else:
            folds[f].append(name)
    # 无分组图轮询并入（应只有极少数）
    for i, name in enumerate(ungrouped):
        folds[i % 4].append(name)

    # 平衡表
    tot = Counter()
    print("\n折 | 图数 | GT=0图 | GT | CBHPM | CBVPM")
    for f in range(4):
        gt = Counter()
        n0 = 0
        for name in folds[f]:
            e = expect.get(name, Counter())
            if not e:
                n0 += 1
            gt.update(e)
        tot.update(gt)
        print(" %d | %3d | %3d | %3d | %3d | %3d" %
              (f, len(folds[f]), n0, sum(gt.values()),
               gt.get("CBHPM", 0), gt.get("CBVPM", 0)))
    print("合计 GT:", dict(tot), " 无分组图: %d" % len(ungrouped))

    with open(os.path.join(OUT, "fold_assignments.csv"), "w", newline="",
              encoding="utf-8") as fp:
        w = csv.writer(fp)
        w.writerow(["image", "fold", "segment", "kpost"])
        for f in range(4):
            for name in folds[f]:
                g = group.get(name)
                w.writerow([name, f, g[0] if g else "", g[1] if g else ""])
    manifest = {
        "n_benchmark": len(bench),
        "n_excluded_conflict": len(conflict),
        "gt_totals": dict(tot),
        "folds": {str(f): len(folds[f]) for f in range(4)},
        "f1b04_kpost_median": med,
    }
    with open(os.path.join(OUT, "final_manifest.json"), "w",
              encoding="utf-8") as fp:
        json.dump(manifest, fp, ensure_ascii=False, indent=1)
    print("已写 fold_assignments.csv / final_manifest.json")


if __name__ == "__main__":
    main()
