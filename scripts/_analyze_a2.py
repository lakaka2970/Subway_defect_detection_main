#!/usr/bin/env python3
"""Analyze A2 category 05 and 09 for actionable training data."""
import csv
from pathlib import Path

R = Path("output/7.14训练结果/整理结果")

# Category 05
f = R / "05_模型误报分类或定位错误.csv"
rows = list(csv.DictReader(open(f, encoding="utf-8-sig")))
print(f"05 模型误报: {len(rows)} rows")
has_bbox = sum(1 for r in rows if r.get("bbox", "").strip())
has_gt = sum(1 for r in rows if r.get("gt_class", "").strip())
has_action = sum(1 for r in rows if r.get("annotation_action", "").strip())
has_state = sum(1 for r in rows if r.get("state_label", "").strip())
print(f"  Has bbox: {has_bbox}/{len(rows)}")
print(f"  Has gt_class: {has_gt}/{len(rows)}")
print(f"  Has annotation_action: {has_action}/{len(rows)}")
print(f"  Has state_label: {has_state}/{len(rows)}")

actions = {}
for r in rows:
    a = r.get("annotation_action", "").strip()
    if a:
        actions[a] = actions.get(a, 0) + 1
print(f"  Actions: {actions}")

states = {}
for r in rows:
    s = r.get("state_label", "").strip()
    if s:
        states[s] = states.get(s, 0) + 1
print(f"  States: {states}")

# Error type breakdown with gt_class
for et in ["localization_error", "class_confusion", "structure_fp"]:
    subset = [r for r in rows if r.get("error_type") == et]
    with_gt = sum(1 for r in subset if r.get("gt_class", "").strip())
    with_bbox = sum(1 for r in subset if r.get("bbox", "").strip())
    print(f"  {et}: {len(subset)} total, {with_gt} with gt_class, {with_bbox} with bbox")

# Category 09
f9 = R / "09_EMPTY_LABEL未纳入本轮审查.csv"
rows9 = list(csv.DictReader(open(f9, encoding="utf-8-sig")))
print(f"\n09 EMPTY_LABEL: {len(rows9)} rows")
if rows9:
    print(f"  Columns: {list(rows9[0].keys())[:8]}")
    has_source = sum(1 for r in rows9 if r.get("source_id", "").strip())
    print(f"  Has source_id: {has_source}/{len(rows9)}")

# Summary: what can be used for training
print("\n=== Actionable A2 Data Summary ===")
print(f"  Already merged (02 HN): 60 images")
print(f"  Already merged (03 FN): 38 images")
print(f"  Already merged (04 Rev): 6 images")
print(f"  Category 05 usable:")
loc_err = [r for r in rows if r.get("error_type") == "localization_error" and r.get("gt_class", "").strip()]
cls_conf = [r for r in rows if r.get("error_type") == "class_confusion" and r.get("gt_class", "").strip()]
struct_fp = [r for r in rows if r.get("error_type") == "structure_fp"]
print(f"    localization_error with GT: {len(loc_err)} (can correct bbox)")
print(f"    class_confusion with GT: {len(cls_conf)} (can correct class)")
print(f"    structure_fp: {len(struct_fp)} (can use as hard negatives)")
print(f"  Category 09 (empty labels): {len(rows9)} (potential negatives, needs audit)")