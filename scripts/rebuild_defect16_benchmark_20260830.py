# -*- coding: utf-8 -*-
"""重建 Defect_dataset_16 的可恢复部分（只读归档 + 本地旧数据，产出独立基准目录）。

来源说明（用户 2026-08-30 确认）：
  Defect_dataset_16 = 770 张检测车图的旧 7 类标注（删去全部垂直悬吊类 0-4）
                      + Dataset_2（车间）共同构成的 16 类体系。
本地可恢复：
  - 500 张在 data/Defect_dataset（旧 7 类标签完整）→ 保留 5/6 类框即 16 类 CBHPM/CBVPM
  - 69 张在 data/Normal_dataset（GT=0，空标签）
不可恢复：201 张（INSD/DRPS/RHT 全部 403 GT），已确认丢失。

产出：
  data/Defect_dataset_16_rebuilt/{images,labels}/  569 张 + 标签
  data/Defect_dataset_16_rebuilt/rebuild_audit.{json,csv}
  data/Defect_dataset_16_rebuilt/grouping_stats.json（折设计依据）
"""
import csv
import glob
import json
import os
import re
import shutil
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "docs", "plans", "8.28泛化性训练")
OUT = os.path.join(ROOT, "data", "Defect_dataset_16_rebuilt")

NAMES16 = ["VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
           "RHTBNM", "RHTBNL", "GWCSBNM", "GWCSBNL", "GWCNM", "GWCNL",
           "BSBM", "INSD", "DRPS"]
KEEP = {5, 6}          # CBHPM, CBVPM（旧编号与 16 类编号一致）
VHB = {0, 1, 2, 3, 4}  # 垂直悬吊系，按标注口径删除

PAT = re.compile(r"^(\d+)_K(\d+)_(F1[AB]\d+)-(\d+)_(\d+)_(\d+)\.jpg$")


def index_pool(pattern):
    d = {}
    for p in glob.glob(os.path.join(ROOT, pattern), recursive=True):
        d[os.path.basename(p)] = p
    return d


def main():
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels"), exist_ok=True)

    summary = list(csv.DictReader(open(
        os.path.join(BASE, "DG-v1_20260828", "external_test_770", "image_summary.csv"),
        encoding="utf-8-sig")))
    tax = list(csv.DictReader(open(
        os.path.join(BASE, "domain_shift_analysis_20260828", "gt_miss_taxonomy.csv"),
        encoding="utf-8-sig")))
    expect = defaultdict(Counter)
    for r in tax:
        expect[r["image"]][r["gt_class"]] += 1

    pool_a = index_pool(r"data\Defect_dataset\images\**\*.jpg")
    lab_a = {}
    for p in glob.glob(os.path.join(ROOT, "data", "Defect_dataset", "labels",
                                    "**", "*.txt"), recursive=True):
        lab_a[os.path.splitext(os.path.basename(p))[0]] = p
    pool_b = index_pool(r"data\Normal_dataset\images\**\*.jpg")

    audit_rows, stats = [], Counter()
    group = {}
    for r in summary:
        name = r["image"]
        exp = {k: v for k, v in expect.get(name, Counter()).items() if v}
        row = {"image": name, "exp": exp, "source": None, "status": None,
               "act": {}, "dropped_vhb": 0, "note": ""}
        if name in pool_a:
            row["source"] = "Defect_dataset"
            lp = lab_a.get(os.path.splitext(name)[0])
            if lp is None:
                row["status"] = "NO_OLD_LABEL"
            else:
                kept, dropped = [], 0
                for ln in open(lp, encoding="utf-8"):
                    f = ln.split()
                    if len(f) != 5:
                        continue
                    c = int(f[0])
                    if c in KEEP:
                        kept.append(ln.strip())
                    elif c in VHB:
                        dropped += 1
                    else:
                        row["note"] += " unexpected_cls_%d" % c
                row["dropped_vhb"] = dropped
                act = Counter()
                for ln in kept:
                    act[NAMES16[int(ln.split()[0])]] += 1
                row["act"] = dict(act)
                with open(os.path.join(OUT, "labels",
                                       os.path.splitext(name)[0] + ".txt"),
                          "w", encoding="utf-8") as fp:
                    fp.write("\n".join(kept) + ("\n" if kept else ""))
                shutil.copy2(pool_a[name], os.path.join(OUT, "images", name))
                row["status"] = "OK" if act == Counter(exp) else "MISMATCH"
        elif name in pool_b:
            row["source"] = "Normal_dataset"
            open(os.path.join(OUT, "labels",
                              os.path.splitext(name)[0] + ".txt"), "w").close()
            shutil.copy2(pool_b[name], os.path.join(OUT, "images", name))
            row["act"] = {}
            row["status"] = "OK" if not exp else "MISMATCH"
        else:
            row["status"] = "MISSING"
        stats[row["status"]] += 1
        audit_rows.append(row)
        m = PAT.match(name)
        if m and row["status"] in ("OK", "MISMATCH"):
            group[name] = {"seq": int(m.group(1)), "kpost": int(m.group(2)),
                           "segment": m.group(3), "span": int(m.group(4)),
                           "cam": "%s_%s" % (m.group(5), m.group(6))}

    # 汇总
    tot_exp, tot_act = Counter(), Counter()
    for row in audit_rows:
        if row["status"] in ("OK", "MISMATCH"):
            tot_exp.update(row["exp"])
            tot_act.update(row["act"])
    mism = [r for r in audit_rows if r["status"] == "MISMATCH"]

    # 分组统计（折设计依据）
    gstats = {"n_with_group": len(group)}
    if group:
        kp = sorted(v["kpost"] for v in group.values())
        gstats["kpost_min"], gstats["kpost_max"] = kp[0], kp[-1]
        gstats["segment_counts"] = dict(Counter(
            v["segment"] for v in group.values()))
        gstats["n_spans"] = len({(v["segment"], v["span"]) for v in group.values()})
        gstats["cam_counts"] = dict(Counter(v["cam"] for v in group.values()))
        # 缺陷图的公里标四分位
        import statistics
        gstats["kpost_quartiles"] = [
            int(statistics.quantiles(kp, n=4)[i]) for i in range(3)]
        # 每张图的 GT 数（用于折平衡检查）
        per_kp = [(v["kpost"], sum(expect.get(k, Counter()).values()))
                  for k, v in group.items()]
        gstats["gt_by_kpost_quartile"] = []
        qs = gstats["kpost_quartiles"]
        for lo, hi in [(-1, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]),
                       (qs[2], 10 ** 9)]:
            gstats["gt_by_kpost_quartile"].append(
                sum(g for k, g in per_kp if lo < k <= hi) if lo >= 0
                else sum(g for k, g in per_kp if k <= hi))

    audit = {
        "n_770": len(summary),
        "status_counts": dict(stats),
        "gt_expected_recoverable": dict(tot_exp),
        "gt_rebuilt_actual": dict(tot_act),
        "vhb_boxes_dropped": sum(r["dropped_vhb"] for r in audit_rows),
        "n_mismatch": len(mism),
        "mismatch_examples": [
            {"image": r["image"], "exp": r["exp"], "act": r["act"]}
            for r in mism[:20]],
    }
    with open(os.path.join(OUT, "rebuild_audit.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": audit, "grouping": gstats}, f,
                  ensure_ascii=False, indent=1)
    with open(os.path.join(OUT, "rebuild_audit.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "source", "status", "exp", "act", "dropped_vhb",
                    "note"])
        for r in audit_rows:
            w.writerow([r["image"], r["source"], r["status"],
                        json.dumps(r["exp"], ensure_ascii=False),
                        json.dumps(r["act"], ensure_ascii=False),
                        r["dropped_vhb"], r["note"]])
    with open(os.path.join(OUT, "grouping_stats.json"), "w",
              encoding="utf-8") as f:
        json.dump(group, f, ensure_ascii=False, indent=0)

    print("=== 重建审计 ===")
    print("状态分布:", dict(stats))
    print("期望 GT（可恢复图）:", dict(tot_exp))
    print("实建 GT:", dict(tot_act))
    print("删除的垂直悬吊框:", audit["vhb_boxes_dropped"])
    print("MISMATCH %d 张，示例:" % len(mism))
    for r in mism[:10]:
        print("  %s exp=%s act=%s" % (r["image"], r["exp"], r["act"]))
    print("\n=== 分组统计 ===")
    print(json.dumps(gstats, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
