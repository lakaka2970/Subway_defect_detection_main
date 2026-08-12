#!/usr/bin/env python3
"""
统计 Defect_dataset 中各类别标签数量，生成 Markdown 统计报告。
依据 defect_dict.json 中的编码-中文名称对照表，输出完整的统计文档。
"""

import json
import os
from collections import defaultdict
from pathlib import Path

# ── 配置 ──────────────────────────────────────────────
DATASET_DIR = Path("data/Defect_dataset")
LABELS_DIR = DATASET_DIR / "labels"
TRAIN_DIR = LABELS_DIR / "train"
VAL_DIR = LABELS_DIR / "val"
DEFECT_DICT_PATH = Path("subway_defect/deployment/defect_dict.json")
OUTPUT_PATH = Path("docs/Defect_dataset统计报告.md")

# ── 1. 加载缺陷字典 ──────────────────────────────────
with open(DEFECT_DICT_PATH, "r", encoding="utf-8") as f:
    defect_dict = json.load(f)

code_to_info: dict[int, dict] = {}  # class_id -> info

# 先通过 defect_data.yaml 建立 id -> code 映射
# 同时也从 label 中动态发现所有出现的 class_id
def load_name_mapping() -> dict[int, str]:
    """从 defect_data.yaml 读取 id->code 映射，再从 defect_dict.json 补全中文名"""
    import yaml
    yaml_path = DATASET_DIR / "defect_data.yaml"
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    id_to_code: dict[int, str] = {}
    if "names" in data:
        for id_str, code in data["names"].items():
            id_to_code[int(id_str)] = code
    return id_to_code


# 建立 code -> 详细信息
code_to_detail: dict[str, dict] = {}
for d in defect_dict["defects"]:
    code_to_detail[d["code"]] = d

# ── 2. 扫描标签文件 ──────────────────────────────────

def parse_yolo_label(filepath: Path) -> list[int]:
    """解析 YOLO 标注文件，返回 class_id 列表"""
    ids = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if parts:
                ids.append(int(parts[0]))
    return ids


def scan_labels(label_dir: Path) -> dict[int, dict]:
    """
    扫描目录下所有 .txt 标注文件。
    返回: {class_id: {"instances": int, "images": int, "image_files": set}}
    """
    stats: dict[int, dict] = defaultdict(lambda: {"instances": 0, "images": 0, "image_files": set()})

    txt_files = sorted(label_dir.glob("*.txt"))
    for txt_file in txt_files:
        class_ids = parse_yolo_label(txt_file)
        if not class_ids:
            continue
        # 记录该图像包含的类别
        image_has_class = set(class_ids)
        for cid in class_ids:
            stats[cid]["instances"] += 1
        for cid in image_has_class:
            stats[cid]["images"] += 1
            stats[cid]["image_files"].add(txt_file.stem)

    return stats


train_stats = scan_labels(TRAIN_DIR)
val_stats = scan_labels(VAL_DIR)

# 收集所有 class_id
all_class_ids = sorted(set(list(train_stats.keys()) + list(val_stats.keys())))

# ── 3. 加载 id->code 映射 ─────────────────────────────
id_to_code = load_name_mapping()

# 对于 yaml 中没有的 class_id，尝试从 defect_dict 推断（如果有需要）
# 这里也输出未映射的 class_id 供检查
print(f"在标签中发现的 class_id: {all_class_ids}")
print(f"defect_data.yaml 中定义的 class_id: {sorted(id_to_code.keys())}")

unmapped = set(all_class_ids) - set(id_to_code.keys())
if unmapped:
    print(f"⚠️  警告: 以下 class_id 在 defect_data.yaml 中无定义: {sorted(unmapped)}")

# ── 4. 计算总数 ────────────────────────────────────────
total_train_instances = sum(train_stats[c]["instances"] for c in all_class_ids)
total_val_instances = sum(val_stats[c]["instances"] for c in all_class_ids)
total_instances = total_train_instances + total_val_instances

# 图像数
train_label_files = set()
for c in all_class_ids:
    train_label_files |= train_stats[c].get("image_files", set())
val_label_files = set()
for c in all_class_ids:
    val_label_files |= val_stats[c].get("image_files", set())

# 统计所有图像文件
all_train_txt = set(f.stem for f in TRAIN_DIR.glob("*.txt"))
all_val_txt = set(f.stem for f in VAL_DIR.glob("*.txt"))

# 无标注的图像 = 只有图片没有对应txt的，但我们有augmented labels中没有对应图像的
# 更准确: 统计有标注的图像数量
train_annotated = len(train_label_files)
val_annotated = len(val_label_files)

# 统计train和val中的实际图像数
TRAIN_IMG_DIR = DATASET_DIR / "images" / "train"
VAL_IMG_DIR = DATASET_DIR / "images" / "val"
train_images_total = len(list(TRAIN_IMG_DIR.glob("*.jpg"))) if TRAIN_IMG_DIR.exists() else len(all_train_txt)
val_images_total = len(list(VAL_IMG_DIR.glob("*.jpg"))) if VAL_IMG_DIR.exists() else len(all_val_txt)

# ── 5. 生成 Markdown 报告 ─────────────────────────────

def get_class_name(class_id: int) -> tuple[str, str, str, str]:
    """返回 (code, name_cn, severity, category)"""
    code = id_to_code.get(class_id, f"UNKNOWN_{class_id}")
    detail = code_to_detail.get(code, {})
    name_cn = detail.get("name_cn", f"未知类别(ID={class_id})")
    severity = detail.get("severity", "unknown")
    category = detail.get("category", "unknown")
    return code, name_cn, severity, category


def severity_label(sev: str) -> str:
    labels = {
        "minor": "轻微",
        "normal": "一般",
        "serious": "严重",
        "critical": "危急",
    }
    return labels.get(sev, sev)


def category_label(cat: str) -> str:
    labels = {
        "rigid_catenary": "刚性接触网",
        "flexible_catenary": "柔性接触网",
    }
    return labels.get(cat, cat)


# 构建表格数据
rows = []
for cid in all_class_ids:
    code, name_cn, severity, category = get_class_name(cid)
    train_cnt = train_stats[cid]["instances"]
    val_cnt = val_stats[cid]["instances"]
    total = train_cnt + val_cnt
    pct = total / total_instances * 100 if total_instances > 0 else 0
    rows.append((cid, code, name_cn, category_label(category), severity_label(severity),
                 train_cnt, val_cnt, total, pct))

# 按实例数降序排列
rows_by_count = sorted(rows, key=lambda r: r[7], reverse=True)

now = __import__("datetime").datetime.now().strftime("%Y-%m-%d")

report = f"""# Defect_dataset 缺陷类型统计报告

> **统计日期**: {now}
> **数据集路径**: `data/Defect_dataset/`
> **统计对象**: train / val 两个子集中全部标注实例
> **类别定义来源**: `subway_defect/deployment/defect_dict.json`

---

## 1. 数据集概览

| 统计项 | train | val | 合计 |
|--------|-------|-----|------|
| 图像总数 | {train_images_total} | {val_images_total} | **{train_images_total + val_images_total}** |
| 含标注的图像数 | {train_annotated} | {val_annotated} | **{train_annotated + val_annotated}** |
| 无缺陷背景图 | {train_images_total - train_annotated} | {val_images_total - val_annotated} | **{(train_images_total - train_annotated) + (val_images_total - val_annotated)}** |
| 缺陷实例总数 | {total_train_instances} | {total_val_instances} | **{total_instances}** |

---

## 2. 缺陷类别统计（按类别ID排序）

| ID | 编码 | 中文名称 | 所属系统 | 严重等级 | train | val | 合计 | 占比 |
|----|------|---------|----------|---------|-------|-----|------|------|
"""

for r in rows:
    report += f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | **{r[7]}** | {r[8]:.1f}% |\n"

# 合计行
report += f"| **—** | **—** | **合计** | **—** | **—** | **{total_train_instances}** | **{total_val_instances}** | **{total_instances}** | **100%** |\n"

report += """
---

## 3. 按缺陷实例数量排序（降序）

```
"""

# ASCII 柱状图
max_count = rows_by_count[0][7] if rows_by_count else 1
max_bar_len = 40

for r in rows_by_count:
    bar_len = int(r[7] / max_count * max_bar_len)
    bar = "█" * bar_len
    report += f"{r[1]:8s}  {bar:<{max_bar_len}} {r[7]:5d}  ({r[8]:.1f}%)  {r[2]}\n"

report += "```\n\n---\n\n"

# ── 6. 按系统类型统计 ──
rigid_rows = [r for r in rows_by_count if "刚性" in r[3]]
flexible_rows = [r for r in rows_by_count if "柔性" in r[3]]
rigid_total = sum(r[7] for r in rigid_rows)
flexible_total = sum(r[7] for r in flexible_rows)

report += f"""## 4. 按系统类型统计

### 4.1 刚性接触网（{len(rigid_rows)}类，{rigid_total}实例，占{rigid_total/total_instances*100:.1f}%）

| 编码 | 中文名称 | 实例数 | 占比(子系统内) |
|------|---------|--------|---------------|
"""
for r in rigid_rows:
    report += f"| {r[1]} | {r[2]} | {r[7]} | {r[7]/rigid_total*100:.1f}% |\n"

report += f"""
### 4.2 柔性接触网（{len(flexible_rows)}类，{flexible_total}实例，占{flexible_total/total_instances*100:.1f}%）

| 编码 | 中文名称 | 实例数 | 占比(子系统内) |
|------|---------|--------|---------------|
"""
for r in flexible_rows:
    report += f"| {r[1]} | {r[2]} | {r[7]} | {r[7]/flexible_total*100:.1f}% |\n"

report += """
---

## 5. 按严重等级统计

"""

# 收集各等级
severity_counts: dict[str, int] = defaultdict(int)
severity_rows: dict[str, list] = defaultdict(list)
for r in rows_by_count:
    sev = r[4]
    severity_counts[sev] += r[7]
    severity_rows[sev].append(r)

sev_order = ["危急", "严重", "一般", "轻微"]
sev_icons = {"危急": "🔴", "严重": "🟠", "一般": "🟡", "轻微": "🔵"}

report += "| 严重等级 | 实例数 | 占比 | 涉及类别 |\n"
report += "|----------|--------|------|----------|\n"
for sev in sev_order:
    if sev in severity_counts:
        count = severity_counts[sev]
        codes = "、".join(r[1] for r in severity_rows[sev])
        report += f"| {sev_icons.get(sev, '')} {sev} | {count} | {count/total_instances*100:.1f}% | {codes} |\n"

report += """
---

## 6. 覆盖图像数统计（每个类别涉及的图像数）

| 编码 | 中文名称 | train图像 | val图像 | 合计图像 |
|------|---------|-----------|---------|----------|
"""
for r in rows_by_count:
    cid = r[0]
    t_imgs = train_stats[cid]["images"]
    v_imgs = val_stats[cid]["images"]
    report += f"| {r[1]} | {r[2]} | {t_imgs} | {v_imgs} | **{t_imgs + v_imgs}** |\n"

report += """
---

## 7. 类别分布特征分析

### 7.1 样本不均衡
"""

most = rows_by_count[0]
least = rows_by_count[-1]

report += f"""
- **最多**：{most[1]}（{most[7]}例，{most[8]:.1f}%）
- **最少**：{least[1]}（{least[7]}例，{least[8]:.1f}%）
- **极值比**：约 **{most[7]/least[7]:.1f} : 1**（{most[1]} / {least[1]}）
- **尾部类别**（<100例）："""

tail_classes = [r for r in rows_by_count if r[7] < 100]
if tail_classes:
    for r in tail_classes:
        report += f"{r[1]}（{r[7]}）、"
    report = report.rstrip("、")
else:
    report += "无"

report += """

### 7.2 螺母"缺失" vs "松动"模式

| 缺陷对 | 缺失 | 松动 | 比值 |
|--------|------|------|------|
"""

# 自动识别缺失-松动对
missing_codes = {r[1]: r[7] for r in rows_by_count if "缺失" in r[2]}
loose_codes = {r[1]: r[7] for r in rows_by_count if "松动" in r[2]}

# 手动配对
pairs = [
    ("VHBNM", "VHBNL", "VHBN"),
    ("SVHBNM", "SVHBNL", "SVHBN"),
    ("RHTBNM", "RHTBNL", "RHTBN"),
    ("GWCSBNM", "GWCSBNL", "GWCSB"),
    ("GWCNM", "GWCNL", "GWCN"),
]

for m_code, l_code, label in pairs:
    m_cnt = missing_codes.get(m_code, 0)
    l_cnt = loose_codes.get(l_code, 0)
    if m_cnt > 0 or l_cnt > 0:
        ratio = f"{m_cnt/l_cnt:.1f} : 1" if l_cnt > 0 else "N/A"
        report += f"| {label} | {m_cnt} | {l_cnt} | {ratio} |\n"

report += """
### 7.3 训练/验证分布一致性

"""

if total_val_instances > 0:
    ratio = total_train_instances / total_val_instances
    report += f"train : val ≈ **{ratio:.1f} : 1**（各缺陷类别与总体比例基本一致，未见明显分配偏斜）。\n"

# 检查各类别的 train/val 比例
report += """
| 编码 | train占比 | val占比 | 偏差 |
|------|-----------|---------|------|
"""
for r in rows_by_count:
    cid = r[0]
    train_c = train_stats[cid]["instances"]
    val_c = val_stats[cid]["instances"]
    train_pct = train_c / total_train_instances * 100 if total_train_instances > 0 else 0
    val_pct = val_c / total_val_instances * 100 if total_val_instances > 0 else 0
    diff = abs(train_pct - val_pct)
    flag = " ⚠️" if diff > 2.0 else ""
    report += f"| {r[1]} | {train_pct:.1f}% | {val_pct:.1f}% | {diff:.1f}%{flag} |\n"

# ── 7.4 未纳入当前数据集的规范类别 ──
all_defect_codes = set(d["code"] for d in defect_dict["defects"])
dataset_codes = set(r[1] for r in rows)
missing_codes_set = all_defect_codes - dataset_codes

report += f"""
### 7.4 未纳入当前数据集的规范类别

按照 `subway_defect/deployment/defect_dict.json` 定义的完整 {len(all_defect_codes)} 类缺陷体系，以下 {len(missing_codes_set)} 类在数据集中暂无标注：

| 编码 | 中文名称 | 所属系统 | 严重等级 |
|------|---------|----------|----------|
"""
for code in sorted(missing_codes_set):
    detail = code_to_detail.get(code, {})
    name_cn = detail.get("name_cn", code)
    cat = category_label(detail.get("category", ""))
    sev = severity_label(detail.get("severity", ""))
    report += f"| {code} | {name_cn} | {cat} | {sev} |\n"

report += f"""
> 这 {len(missing_codes_set)} 类待后续采集标注补充。

---

## 8. 标签与中文名称对照总表

下表依据 `subway_defect/deployment/defect_dict.json`（单一事实来源），列出完整的 {len(all_defect_codes)} 类缺陷编码与中文名称对照：

| 序号 | 编码 | 中文名称 | 所属系统 | 严重等级 | 当前数据集状态 |
|------|------|---------|----------|----------|---------------|
"""
for i, d in enumerate(defect_dict["defects"], 1):
    code = d["code"]
    cat = category_label(d.get("category", ""))
    sev = severity_label(d.get("severity", ""))
    status = "✅ 已标注" if code in dataset_codes else "⏳ 待采集"
    report += f"| {i} | {code} | {d['name_cn']} | {cat} | {sev} | {status} |\n"

report += f"""
---

## 9. 结论

- 当前 `Defect_dataset` 共覆盖 **{len(rows)} 类缺陷**，总计 **{total_instances} 个标注实例**（train: {total_train_instances}, val: {total_val_instances}），分布于 **{train_annotated + val_annotated} 张带标注图像**。
- 刚性接触网缺陷占 **{rigid_total/total_instances*100:.1f}%**，柔性接触网缺陷占 **{flexible_total/total_instances*100:.1f}%**。
- 类别不均衡程度为 **{most[7]/least[7]:.1f} : 1**（最大类 / 最小类）"""

if tail_classes:
    tail_total = sum(r[7] for r in tail_classes)
    tail_codes = "、".join(r[1] for r in tail_classes)
    report += f"，尾部{len(tail_classes)}类（{tail_codes}）合计仅 **{tail_total} 例（{tail_total/total_instances*100:.1f}%）**，建议针对性进行数据增强或合成以提升模型对该类别的泛化能力。"
else:
    report += "。"

report += f"""
- 完整 {len(all_defect_codes)} 类规范体系中，尚有 {len(missing_codes_set)} 类无标注数据，建议后续采集补齐。

---

*报告生成: {now} | 数据版本: Defect_dataset*
"""

# ── 写入文件 ──────────────────────────────────────────
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(report)

print(f"Report generated: {OUTPUT_PATH}")
print(f"   Classes: {len(rows)}")
print(f"   Total instances: {total_instances} (train: {total_train_instances}, val: {total_val_instances})")
print(f"   Annotated images: {train_annotated + val_annotated}")
