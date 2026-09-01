#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全量数据审计 第三阶段（2026-09-01）——三处账目对不上的地方
  Q1. 旧7类标签 CBHPM 534 / CBVPM 405，重建基准却只有 204 / 172。差的 563 个框去哪了？
  Q2. Normal_dataset ∩ 基准 实测 44 张，报告称 34 张并入 + 35 张剔除 = 69。对不上。
  Q3. fold3 公里标跨度 K16446–K58825（42 公里），是"一个区段"还是两个远距区段拼成？
"""
import json, csv
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(r"E:\Work\Subway_defect_detection_main")
DATA = ROOT / "data"
OUT = ROOT / "docs" / "plans" / "9.01全量数据盘点"
OUT.mkdir(parents=True, exist_ok=True)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
R = {}

def stems(p):
    p = Path(p)
    return {f.stem for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXT} if p.exists() else set()

def boxes(lab_root):
    """返回 {stem: [(cls, cx, cy, w, h), ...]}"""
    lab_root = Path(lab_root)
    out = {}
    if not lab_root.exists():
        return out
    for f in lab_root.rglob("*.txt"):
        if f.name == "classes.txt":
            continue
        b = []
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            p = line.split()
            if len(p) < 5:
                continue
            try:
                b.append((int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
            except ValueError:
                pass
        out[f.stem] = b
    return out

def seg_of(s):
    for p in s.split("_"):
        if p.startswith("F1") and "-" in p:
            return p.split("-")[0]
    return "UNKNOWN"

def km_of(s):
    for p in s.split("_"):
        if p.startswith("K") and len(p) > 1 and p[1:].isdigit():
            return int(p[1:])
    return None

rb = DATA / "Defect_dataset_16_rebuilt"
dd = DATA / "Defect_dataset"
nd = DATA / "Normal_dataset"

bench_imgs = stems(rb / "images")
bench_boxes = boxes(rb / "labels")
old_boxes = boxes(dd / "labels")
old_imgs = stems(dd / "images") or stems(dd)
normal_imgs = stems(nd / "images")

# ============================================================ Q1
q1 = {}
common = sorted(set(old_boxes) & set(bench_boxes))
q1["n_common_images"] = len(common)
old_c, new_c = Counter(), Counter()
for s in common:
    for b in old_boxes[s]:
        if b[0] in (5, 6):
            old_c[b[0]] += 1
    for b in bench_boxes[s]:
        new_c[b[0]] += 1
q1["old_7class_CBHPM_CBVPM_on_common_imgs"] = {str(k): v for k, v in sorted(old_c.items())}
q1["rebuilt_16class_on_common_imgs"] = {str(k): v for k, v in sorted(new_c.items())}
q1["delta"] = {k: new_c[int(k)] - old_c[int(k)] for k in q1["old_7class_CBHPM_CBVPM_on_common_imgs"]}

# 逐图差异分布
diff = Counter()
examples = []
for s in common:
    o = sum(1 for b in old_boxes[s] if b[0] in (5, 6))
    n = len(bench_boxes[s])
    diff[n - o] += 1
    if n - o != 0 and len(examples) < 12:
        examples.append({"image": s, "old_cb": o, "new_cb": n})
q1["per_image_delta_hist"] = {str(k): v for k, v in sorted(diff.items())}
q1["examples_of_mismatch"] = examples

# 旧7类里 5/6 类框总数（全部 899 张）
allold = Counter()
for s, bs in old_boxes.items():
    for b in bs:
        allold[b[0]] += 1
q1["old7class_total_hist"] = {str(k): v for k, v in sorted(allold.items())}
q1["old7class_n_images"] = len(old_boxes)
q1["rebuilt_n_images"] = len(bench_boxes)

# 审计文件
for f in ["rebuild_audit.json", "pass2_audit.json"]:
    p = rb / f
    if p.exists():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                q1[f] = {k: (v if not isinstance(v, (list, dict)) else f"<{type(v).__name__} len={len(v)}>")
                         for k, v in list(d.items())[:30]}
            else:
                q1[f] = f"<list len={len(d)}>"
        except Exception as e:
            q1[f] = f"error: {e}"
R["Q1_box_count_gap"] = q1

# ============================================================ Q2
q2 = {}
inter = sorted(normal_imgs & bench_imgs)
q2["n_normal_inter_bench"] = len(inter)
q2["n_normal_total"] = len(normal_imgs)
q2["segments_of_overlap"] = dict(Counter(seg_of(s) for s in inter).most_common())
q2["overlap_with_GT"] = sum(1 for s in inter if len(bench_boxes.get(s, [])) > 0)
q2["overlap_empty_GT"] = sum(1 for s in inter if len(bench_boxes.get(s, [])) == 0)
q2["normal_inter_olddefect"] = len(normal_imgs & old_imgs)
q2["overlap_examples"] = [{"img": s, "seg": seg_of(s), "km": km_of(s), "n_gt": len(bench_boxes.get(s, []))}
                          for s in inter[:15]]
# 基准里来自 Normal 的图（即在 Normal 中但不在旧 Defect_dataset 中）
only_normal = [s for s in inter if s not in old_imgs]
q2["n_in_normal_not_in_olddefect"] = len(only_normal)
q2["n_in_both_normal_and_olddefect"] = len(inter) - len(only_normal)
# 基准 534 = 旧 Defect_dataset 的 500 + 来自 Normal 的 34？
q2["bench_composition"] = {
    "n_bench": len(bench_imgs),
    "from_old_defect_dataset": len(bench_imgs & old_imgs),
    "from_normal_only": len(bench_imgs - old_imgs),
    "unexplained": len(bench_imgs - old_imgs - normal_imgs),
}
R["Q2_normal_overlap"] = q2

# ============================================================ Q3
q3 = {}
fold_csv = rb / "fold_assignments.csv"
if fold_csv.exists():
    rows = list(csv.DictReader(fold_csv.open(encoding="utf-8")))
    fk = [k for k in rows[0] if "fold" in k.lower()][0]
    ik = [k for k in rows[0] if k.lower() in ("image", "stem", "name", "file", "img")][0]
    per = defaultdict(list)
    for r in rows:
        per[r[fk]].append(Path(r[ik]).stem)
    for k in sorted(per):
        ss = per[k]
        byseg = defaultdict(list)
        for s in ss:
            byseg[seg_of(s)].append(km_of(s))
        q3[f"fold{k}"] = {
            "n": len(ss),
            "by_segment": {sg: {"n": len(v), "km_min": min(x for x in v if x), "km_max": max(x for x in v if x)}
                           for sg, v in sorted(byseg.items())},
        }
# Normal_dataset 各区段公里标范围（看能否提供新区段覆盖）
byn = defaultdict(list)
for s in normal_imgs:
    byn[seg_of(s)].append(km_of(s))
q3["Normal_dataset_km_ranges"] = {sg: {"n": len(v), "km_min": min(x for x in v if x), "km_max": max(x for x in v if x)}
                                  for sg, v in sorted(byn.items())}
# 基准各区段公里标范围
byb = defaultdict(list)
for s in bench_imgs:
    byb[seg_of(s)].append(km_of(s))
q3["benchmark_km_ranges"] = {sg: {"n": len(v), "km_min": min(x for x in v if x), "km_max": max(x for x in v if x)}
                             for sg, v in sorted(byb.items())}
# 旧 Defect_dataset 区段范围
byo = defaultdict(list)
for s in old_imgs:
    byo[seg_of(s)].append(km_of(s))
q3["old_defect_dataset_km_ranges"] = {sg: {"n": len(v), "km_min": min(x for x in v if x), "km_max": max(x for x in v if x)}
                                      for sg, v in sorted(byo.items())}
R["Q3_fold3_geometry"] = q3

out = OUT / "audit_phase3.json"
out.write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")

print("=" * 80)
print("Q1. 框数差账：旧7类 CBHPM/CBVPM vs 重建基准")
print(f"  共同图像数: {q1['n_common_images']}")
print(f"  旧7类(仅5/6类)框数: {q1['old_7class_CBHPM_CBVPM_on_common_imgs']}")
print(f"  重建16类框数      : {q1['rebuilt_16class_on_common_imgs']}")
print(f"  逐图差值分布(new-old): {q1['per_image_delta_hist']}")
print(f"  旧7类全量直方图: {q1['old7class_total_hist']}  (共 {q1['old7class_n_images']} 图)")
if "rebuild_audit.json" in q1:
    print(f"  rebuild_audit: {q1['rebuild_audit.json']}")
if "pass2_audit.json" in q1:
    print(f"  pass2_audit  : {q1['pass2_audit.json']}")
print()
print("=" * 80)
print("Q2. Normal 与基准的重叠")
for k, v in q2.items():
    if k != "overlap_examples":
        print(f"  {k}: {v}")
print()
print("=" * 80)
print("Q3. 折几何 / 区段公里标")
for k in sorted(q3):
    print(f"  {k}: {json.dumps(q3[k], ensure_ascii=False)}")
print(f"\n完整结果: {out}")
