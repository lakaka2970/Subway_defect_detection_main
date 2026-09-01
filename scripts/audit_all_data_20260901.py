#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全量数据审计（2026-09-01）
目的：不依赖任何历史报告的数字，从磁盘重新数一遍所有数据资产，
      用于独立验证 8.31/9.01 阶段报告的结论，并找出被遗漏的资产。

输出：docs/plans/9.01全量数据盘点/audit_all_data.json
"""
import os, json, csv, hashlib, sys
from pathlib import Path
from collections import Counter, defaultdict

ROOT = Path(r"E:\Work\Subway_defect_detection_main")
DATA = ROOT / "data"
OUT = ROOT / "docs" / "plans" / "9.01全量数据盘点"
OUT.mkdir(parents=True, exist_ok=True)

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
result = {}

def count_dir(p):
    """返回 (图像数, 标签数, 图像集合, 标签目录是否存在)"""
    p = Path(p)
    if not p.exists():
        return 0, 0, set(), False
    imgs = {f.stem for f in p.iterdir() if f.suffix.lower() in IMG_EXT}
    lab_dir = p / "labels"
    labs = {f.stem for f in lab_dir.iterdir() if f.suffix == ".txt"} if lab_dir.exists() else set()
    return len(imgs), len(labs), imgs, lab_dir.exists()

def read_label_stats(lab_dir, id2name=None):
    """统计标签目录：框数、类直方图、空标签文件数、每图框数分布"""
    lab_dir = Path(lab_dir)
    if not lab_dir.exists():
        return None
    total = 0
    cls = Counter()
    empty = 0
    per_img = Counter()
    files = [f for f in lab_dir.iterdir() if f.suffix == ".txt"]
    for f in files:
        n = 0
        try:
            for line in f.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        c = int(float(parts[0]))
                    except ValueError:
                        continue
                    cls[c] += 1
                    n += 1
        except Exception:
            pass
        total += n
        per_img[n] += 1
        if n == 0:
            empty += 1
    return {
        "n_label_files": len(files),
        "n_boxes": total,
        "n_empty_label_files": empty,
        "class_hist": {str(k): v for k, v in sorted(cls.items())},
        "boxes_per_image_hist": {str(k): v for k, v in sorted(per_img.items())},
    }

# ---------------------------------------------------------------- 1. 资产清单
assets = {}

# 1.1 旧 Defect_dataset（检测车实拍，7 类）
dd = DATA / "Defect_dataset"
n_img, n_lab, stems, has_lab = count_dir(dd / "images")
assets["Defect_dataset"] = {
    "path": str(dd.relative_to(ROOT)),
    "domain": "检测车实拍(目标域)",
    "n_images": n_img,
    "n_labels": n_lab,
    "label_stats": read_label_stats(dd / "labels"),
    "_stems": sorted(stems),
}

# 1.2 重建基准 Defect_dataset_16_rebuilt
rb = DATA / "Defect_dataset_16_rebuilt"
n_img, n_lab, stems_rb, has_lab = count_dir(rb / "images")
assets["Defect_dataset_16_rebuilt"] = {
    "path": str(rb.relative_to(ROOT)),
    "domain": "检测车实拍(目标域) 重建外部基准",
    "n_images": n_img,
    "n_labels": n_lab,
    "label_stats": read_label_stats(rb / "labels"),
    "_stems": sorted(stems_rb),
}

# 1.3 Defect_dataset_2（车间，16 类源）
dd2 = DATA / "Defect_dataset_2" / "Defect_dataset"
n_img, n_lab, stems_d2, has_lab = count_dir(dd2 / "images")
assets["Defect_dataset_2"] = {
    "path": str(dd2.relative_to(ROOT)),
    "domain": "车间静态拍摄(源域)",
    "n_images": n_img,
    "n_labels": n_lab,
    "label_stats": read_label_stats(dd2 / "labels"),
    "_stems": sorted(stems_d2),
}

# 1.4 Normal_dataset（检测车实拍无缺陷）
nd = DATA / "Normal_dataset"
n_img, n_lab, stems_nd, has_lab = count_dir(nd / "images")
assets["Normal_dataset"] = {
    "path": str(nd.relative_to(ROOT)),
    "domain": "检测车实拍(目标域) 无缺陷",
    "n_images": n_img,
    "n_labels": n_lab,
    "label_stats": read_label_stats(nd / "labels"),
    "_stems": sorted(stems_nd),
}

# 1.5 train_data_2（当前唯一训练集）
for name in ["train_data_2", "train_data_3_raw", "train_data_3_raw_2560"]:
    td = DATA / name
    info = {"path": str(td.relative_to(ROOT)), "domain": "车间(源域) 构建训练集", "splits": {}}
    all_stems = set()
    for sp in ["train", "val", "test"]:
        sp_img = td / sp / "images"
        if not sp_img.exists():
            sp_img = td / sp
        n_img, n_lab, stems, has_lab = count_dir(sp_img)
        if n_img == 0 and not (td / sp).exists():
            continue
        labd = td / sp / "labels"
        info["splits"][sp] = {
            "n_images": n_img,
            "n_labels": n_lab,
            "label_stats": read_label_stats(labd) if labd.exists() else None,
        }
        all_stems |= stems
    info["_stems"] = sorted(all_stems)
    assets[name] = info

# 1.6 normal_field_v1_sample
nfs = DATA / "normal_field_v1_sample"
n_img, n_lab, stems_nfs, _ = count_dir(nfs / "images")
assets["normal_field_v1_sample"] = {
    "path": str(nfs.relative_to(ROOT)),
    "domain": "检测车实拍(目标域) 无缺陷切片样本",
    "n_images": n_img,
    "n_labels": n_lab,
    "label_stats": read_label_stats(nfs / "labels"),
    "_stems": sorted(stems_nfs),
}

# ---------------------------------------------------------------- 2. 切片资产
tiles = {}
for t in ["tiles_workshop", "tiles_workshop_val", "tiles_real", "tiles_normal"]:
    td = DATA / t
    if not td.exists():
        tiles[t] = {"exists": False}
        continue
    ti = td / "images"
    tl = td / "labels"
    ni = len([f for f in ti.iterdir() if f.suffix.lower() in IMG_EXT]) if ti.exists() else 0
    nl = len([f for f in tl.iterdir() if f.suffix == ".txt"]) if tl.exists() else 0
    tiles[t] = {
        "exists": True,
        "n_tiles": ni,
        "n_labels": nl,
        "label_stats": read_label_stats(tl) if tl.exists() else None,
    }
result["tiles"] = tiles

# ---------------------------------------------------------------- 3. 切片索引
ti_dir = DATA / "tiles_index"
idx = {}
if ti_dir.exists():
    for f in sorted(ti_dir.iterdir()):
        if f.suffix in {".txt", ".csv", ".json"}:
            if f.suffix == ".txt":
                lines = [l.strip() for l in f.read_text(encoding="utf-8", errors="ignore").splitlines() if l.strip()]
                idx[f.name] = {"type": "txt", "n_lines": len(lines)}
            else:
                idx[f.name] = {"type": f.suffix, "size_bytes": f.stat().st_size}
result["tiles_index"] = idx

# ---------------------------------------------------------------- 4. 重叠/泄漏分析
def seg_of(stem):
    """从文件名提取线路段，如 102903294_K26305_F1B03-146_1_21 -> F1B03"""
    parts = stem.split("_")
    for p in parts:
        if p.startswith("F1") and "-" in p:
            return p.split("-")[0]
    return "UNKNOWN"

sets = {}
for k, v in assets.items():
    if "_stems" in v and v["_stems"]:
        sets[k] = set(v["_stems"])
result["overlap"] = {}
keys = sorted(sets)
for i, a in enumerate(keys):
    for b in keys[i + 1:]:
        inter = sets[a] & sets[b]
        if inter:
            result["overlap"][f"{a} ∩ {b}"] = len(inter)

# 重建基准 vs Normal_dataset 的重叠（报告称 69 张同名）
result["critical_overlap"] = {
    "rebuilt ∩ Normal_dataset": len(set(assets["Defect_dataset_16_rebuilt"]["_stems"]) & sets.get("Normal_dataset", set())),
    "rebuilt ∩ Defect_dataset": len(set(assets["Defect_dataset_16_rebuilt"]["_stems"]) & sets.get("Defect_dataset", set())),
    "Defect_dataset ∩ Normal_dataset": len(sets.get("Defect_dataset", set()) & sets.get("Normal_dataset", set())),
}

# 区段分布
result["segment_distribution"] = {}
for k in ["Defect_dataset", "Defect_dataset_16_rebuilt", "Normal_dataset", "Defect_dataset_2", "train_data_2"]:
    stems = assets.get(k, {}).get("_stems", [])
    result["segment_distribution"][k] = dict(Counter(seg_of(s) for s in stems).most_common())

# ---------------------------------------------------------------- 5. 折分配核对
fold_csv = rb / "fold_assignments.csv"
if fold_csv.exists():
    rows = list(csv.DictReader(fold_csv.open(encoding="utf-8")))
    result["fold_check"] = {
        "n_rows": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
        "per_fold": dict(Counter(r.get("fold", "?") for r in rows)),
        "per_fold_segment": {},
    }
    for fk in sorted(set(r.get("fold", "?") for r in rows)):
        sub = [r for r in rows if r.get("fold") == fk]
        result["fold_check"]["per_fold_segment"][fk] = dict(
            Counter(seg_of(Path(r.get("image", r.get("stem", ""))).stem) for r in sub).most_common()
        )

# ---------------------------------------------------------------- 6. manifest 核对
mf = DATA / "train_data_2" / "manifest.json"
if mf.exists():
    try:
        m = json.loads(mf.read_text(encoding="utf-8"))
        def summarize(obj, depth=0):
            if isinstance(obj, dict):
                return {k: (len(v) if isinstance(v, (list, dict)) else v) for k, v in list(obj.items())[:40]}
            return str(type(obj))
        result["train_data_2_manifest"] = summarize(m)
    except Exception as e:
        result["train_data_2_manifest"] = {"error": str(e)}

# ---------------------------------------------------------------- 7. 类别口径核对
cls_files = {}
for p in [DATA / "train_data_2" / "classes.txt", DATA / "Defect_dataset" / "classes.txt",
          DATA / "train_data_3_raw" / "classes.txt"]:
    if p.exists():
        cls_files[str(p.relative_to(ROOT))] = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
result["classes_files"] = cls_files

# ---------------------------------------------------------------- 8. 输出
for v in assets.values():
    v.pop("_stems", None)
result["assets"] = assets

out_json = OUT / "audit_all_data.json"
out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

# 打印摘要
print("=" * 78)
print("全量数据资产审计")
print("=" * 78)
print(f"{'资产':<34}{'图像':>9}{'标签':>9}{'框数':>9}{'空标签':>8}{'类数':>6}")
print("-" * 78)
for k, v in assets.items():
    ls = v.get("label_stats") or {}
    tot = ls.get("n_boxes", 0)
    if "splits" in v:
        tot = sum((s.get("label_stats") or {}).get("n_boxes", 0) for s in v["splits"].values())
        nimg = sum(s["n_images"] for s in v["splits"].values())
        nlab = sum(s["n_labels"] for s in v["splits"].values())
        emp = sum((s.get("label_stats") or {}).get("n_empty_label_files", 0) for s in v["splits"].values())
        ncls = len(set().union(*[set((s.get("label_stats") or {}).get("class_hist", {})) for s in v["splits"].values()]))
    else:
        nimg, nlab = v.get("n_images", 0), v.get("n_labels", 0)
        emp = ls.get("n_empty_label_files", 0)
        ncls = len(ls.get("class_hist", {}))
    print(f"{k:<34}{nimg:>9}{nlab:>9}{tot:>9}{emp:>8}{ncls:>6}")
print("-" * 78)
print("\n[切片资产]")
for k, v in tiles.items():
    if v.get("exists"):
        ls = v.get("label_stats") or {}
        print(f"  {k:<22} tile={v['n_tiles']:>7}  label={v['n_labels']:>7}  box={ls.get('n_boxes',0):>7}  empty={ls.get('n_empty_label_files',0):>6}")
    else:
        print(f"  {k:<22} 不存在")
print("\n[关键重叠]")
for k, v in result["critical_overlap"].items():
    print(f"  {k:<40} {v}")
print("\n[其他重叠]")
for k, v in sorted(result["overlap"].items(), key=lambda x: -x[1]):
    if v > 0:
        print(f"  {k:<60} {v}")
print("\n[区段分布]")
for k, v in result["segment_distribution"].items():
    print(f"  {k:<30} {v}")
print(f"\n完整结果: {out_json}")
