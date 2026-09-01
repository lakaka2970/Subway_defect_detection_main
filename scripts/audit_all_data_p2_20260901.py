#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全量数据审计 第二阶段（2026-09-01）
修正递归目录计数，并做四项关键核查：
  A. 重建基准（534 图）内部构成与折分配的独立复核
  B. 泄漏风险：基准图像是否出现在任何历史训练集里（train_data_3 系列）
  C. Normal_dataset / tiles_normal 与基准的重叠
  D. 各类资产的"真实可用监督"盘点：到底哪些类在目标域有多少 GT
"""
import json, csv, sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(r"E:\Work\Subway_defect_detection_main")
DATA = ROOT / "data"
OUT = ROOT / "docs" / "plans" / "9.01全量数据盘点"
OUT.mkdir(parents=True, exist_ok=True)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
R = {}

def rglob_stems(p, exts=IMG_EXT):
    p = Path(p)
    if not p.exists():
        return set()
    return {f.stem for f in p.rglob("*") if f.is_file() and f.suffix.lower() in exts}

def label_stats_any(lab_root):
    """递归统计标签：框数、类直方图、空标签、每图框数"""
    lab_root = Path(lab_root)
    if not lab_root.exists():
        return None
    files = [f for f in lab_root.rglob("*.txt") if f.name != "classes.txt"]
    total, empty, cls, per_img = 0, 0, Counter(), Counter()
    for f in files:
        n = 0
        for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or len(line.split()) < 5:
                continue
            try:
                c = int(float(line.split()[0]))
            except ValueError:
                continue
            cls[c] += 1
            n += 1
        total += n
        per_img[n] += 1
        if n == 0:
            empty += 1
    return {"n_files": len(files), "n_boxes": total, "n_empty": empty,
            "class_hist": {str(k): v for k, v in sorted(cls.items())},
            "per_img_hist": {str(k): v for k, v in sorted(per_img.items())}}

def seg_of(stem):
    for p in stem.split("_"):
        if p.startswith("F1") and "-" in p:
            return p.split("-")[0]
    return "UNKNOWN"

def km_of(stem):
    for p in stem.split("_"):
        if p.startswith("K") and len(p) > 1 and p[1:].isdigit():
            return int(p[1:])
    return None

CLS16 = ["VHBNM","VHBNL","SVHBNM","SVHBNL","SVHTNL","CBHPM","CBVPM","RHTBNM",
         "RHTBNL","GWCSBNM","GWCSBNL","GWCNM","GWCNL","BSBM","INSD","DRPS"]
def named(hist):
    out = {}
    for k, v in hist.items():
        i = int(k)
        out[f"{i} {CLS16[i]}" if i < 16 else str(i)] = v
    return out

# ---------------------------------------------------------------- A. 重建基准
rb = DATA / "Defect_dataset_16_rebuilt"
rb_imgs = rglob_stems(rb / "images")
R["A_rebuilt_benchmark"] = {
    "n_images": len(rb_imgs),
    "label_stats": label_stats_any(rb / "labels"),
}
R["A_rebuilt_benchmark"]["label_stats_named"] = named(R["A_rebuilt_benchmark"]["label_stats"]["class_hist"])
R["A_rebuilt_benchmark"]["segment_dist"] = dict(Counter(seg_of(s) for s in rb_imgs).most_common())

fold_csv = rb / "fold_assignments.csv"
if fold_csv.exists():
    rows = list(csv.DictReader(fold_csv.open(encoding="utf-8")))
    R["A_rebuilt_benchmark"]["fold_columns"] = list(rows[0].keys())
    fk = [k for k in rows[0].keys() if "fold" in k.lower()][0]
    ik = [k for k in rows[0].keys() if k.lower() in ("image", "stem", "name", "file", "img")][0]
    per = defaultdict(list)
    for r in rows:
        per[r[fk]].append(r[ik])
    R["A_rebuilt_benchmark"]["folds"] = {}
    for k in sorted(per):
        stems = [Path(x).stem for x in per[k]]
        R["A_rebuilt_benchmark"]["folds"][k] = {
            "n": len(stems),
            "segments": dict(Counter(seg_of(s) for s in stems).most_common()),
            "km_range": [min([km_of(s) for s in stems if km_of(s)] or [0]),
                         max([km_of(s) for s in stems if km_of(s)] or [0])],
        }

# ---------------------------------------------------------------- B. 泄漏风险
bench = rb_imgs
risk = {}
for name, path in [
    ("train_data_3_raw", DATA / "train_data_3_raw"),
    ("train_data_3_raw_2560", DATA / "train_data_3_raw_2560"),
    ("train_data_2", DATA / "train_data_2"),
    ("Defect_dataset(旧7类)", DATA / "Defect_dataset"),
]:
    stems = rglob_stems(path)
    inter = bench & stems
    risk[name] = {
        "n_images_total": len(stems),
        "n_overlap_with_benchmark": len(inter),
        "overlap_segments": dict(Counter(seg_of(s) for s in inter).most_common()),
    }
    if name.startswith("train_data_3"):
        # 该训练集中"检测车实拍"图（有 F1 段）的总数
        f1 = {s for s in stems if seg_of(s) != "UNKNOWN"}
        risk[name]["n_field_images_in_trainset"] = len(f1)
        risk[name]["field_segments"] = dict(Counter(seg_of(s) for s in f1).most_common())
R["B_leakage_risk"] = risk

# ---------------------------------------------------------------- C. Normal 资产
nd = DATA / "Normal_dataset"
nd_imgs = rglob_stems(nd / "images")
R["C_normal_assets"] = {
    "Normal_dataset": {
        "n_images": len(nd_imgs),
        "label_stats": label_stats_any(nd / "labels"),
        "segment_dist": dict(Counter(seg_of(s) for s in nd_imgs).most_common()),
        "n_overlap_with_benchmark": len(nd_imgs & bench),
    }
}
nfs = DATA / "normal_field_v1_sample"
nfs_imgs = rglob_stems(nfs / "images")
R["C_normal_assets"]["normal_field_v1_sample"] = {
    "n_tiles": len(nfs_imgs),
    "label_stats": label_stats_any(nfs / "labels"),
    "source_image_segments": dict(Counter(seg_of(s.split("_t")[0]) for s in nfs_imgs).most_common()),
}
# tiles_normal 的源图区段
tn = DATA / "tiles_normal" / "images"
if tn.exists():
    tn_stems = {f.stem for f in tn.iterdir() if f.suffix.lower() in IMG_EXT}
    src = {s.split("_t")[0] if "_t" in s else s for s in tn_stems}
    R["C_normal_assets"]["tiles_normal"] = {
        "n_tiles": len(tn_stems),
        "n_source_images": len(src),
        "segment_dist_of_sources": dict(Counter(seg_of(s) for s in src).most_common()),
        "n_source_overlap_with_benchmark": len({s for s in src if s in bench}),
    }

# ---------------------------------------------------------------- D. 监督盘点
sup = {}
# 目标域：重建基准 + 旧 7 类
sup["target_domain_rebuilt"] = named(label_stats_any(rb / "labels")["class_hist"])
dd_lab = label_stats_any(DATA / "Defect_dataset" / "labels")
sup["target_domain_old7class"] = named(dd_lab["class_hist"])
# 源域：train_data_2（当前训练集）
td2 = label_stats_any(DATA / "train_data_2")
sup["source_domain_train_data_2"] = named(td2["class_hist"])
# 车间全量 Defect_dataset_2
dd2 = label_stats_any(DATA / "Defect_dataset_2" / "Defect_dataset")
sup["source_domain_Defect_dataset_2"] = named(dd2["class_hist"])
R["D_supervision_inventory"] = sup

# 目标域可评类
tg = set(int(k) for k in label_stats_any(rb / "labels")["class_hist"])
src = set(int(k) for k in td2["class_hist"])
R["D_supervision_inventory"]["_analysis"] = {
    "classes_evaluable_on_target": sorted(tg),
    "classes_with_source_supervision": sorted(src),
    "classes_source_only": sorted(src - tg),
    "classes_no_supervision_at_all": [i for i in range(16) if i not in src],
    "n_classes": 16,
}

# ---------------------------------------------------------------- E. 源域图幅
def sample_size(p, n=3):
    p = Path(p)
    if not p.exists():
        return None
    fs = [f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in IMG_EXT][:n]
    out = []
    for f in fs:
        try:
            from PIL import Image
            with Image.open(f) as im:
                out.append(f"{f.name}: {im.size[0]}x{im.size[1]}")
        except Exception:
            out.append(f"{f.name}: (PIL 不可用)")
    return out
R["E_image_sizes"] = {
    "rebuilt_benchmark": sample_size(rb / "images"),
    "Defect_dataset_2": sample_size(DATA / "Defect_dataset_2" / "Defect_dataset" / "images"),
    "Normal_dataset": sample_size(nd / "images"),
}

out = OUT / "audit_phase2.json"
out.write_text(json.dumps(R, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------------- 打印
print("=" * 80)
print("A. 重建基准复核")
A = R["A_rebuilt_benchmark"]
print(f"  图像 {A['n_images']} | 框 {A['label_stats']['n_boxes']} | 空标签 {A['label_stats']['n_empty']}")
print(f"  类别直方图: {A['label_stats_named']}")
print(f"  区段分布: {A['segment_dist']}")
print(f"  折: {json.dumps(A.get('folds', {}), ensure_ascii=False)}")
print()
print("=" * 80)
print("B. 泄漏风险（基准 534 图是否出现在历史训练集）")
for k, v in R["B_leakage_risk"].items():
    print(f"  {k:<26} 总图 {v['n_images_total']:>5} | 与基准重叠 {v['n_overlap_with_benchmark']:>4} | {v.get('overlap_segments', {})}")
    if "n_field_images_in_trainset" in v:
        print(f"  {'':<26} 内含检测车实拍图 {v['n_field_images_in_trainset']} 张, 区段 {v['field_segments']}")
print()
print("=" * 80)
print("C. Normal 资产")
for k, v in R["C_normal_assets"].items():
    print(f"  {k}: {json.dumps(v, ensure_ascii=False)[:300]}")
print()
print("=" * 80)
print("D. 监督盘点")
print(f"  目标域(重建基准) : {R['D_supervision_inventory']['target_domain_rebuilt']}")
print(f"  目标域(旧7类)    : {R['D_supervision_inventory']['target_domain_old7class']}")
print(f"  源域(train_data_2): {R['D_supervision_inventory']['source_domain_train_data_2']}")
an = R["D_supervision_inventory"]["_analysis"]
print(f"  → 目标域可评类: {an['classes_evaluable_on_target']}")
print(f"  → 仅源域有监督: {an['classes_source_only']}")
print(f"  → 全域零样本  : {an['classes_no_supervision_at_all']}")
print()
print("E. 图幅:", json.dumps(R["E_image_sizes"], ensure_ascii=False))
print(f"\n完整结果: {out}")
