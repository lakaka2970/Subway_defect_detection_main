#!/usr/bin/env python3
"""
Build A2 human review package from outstanding audit items.

Organizes the remaining unreviewed / disputed A2 audit items into a
structured directory for human reviewers:

    output/A2人工审查包/
    ├── README.md                    # Instructions for reviewers
    ├── 01_未完成或结论缺失/          # 75 items — fill in missing conclusions
    │   ├── images/
    │   └── review_sheet.csv
    ├── 02_待完成二审/                # 240 items — second reviewer sign-off
    │   ├── images/
    │   └── review_sheet.csv
    ├── 03_争议隔离/                  # 200 items — dual-reviewer resolution
    │   ├── images/
    │   └── review_sheet.csv
    ├── 04_结论异常需复核/            # 55 items — re-review anomalous conclusions
    │   ├── images/
    │   └── review_sheet.csv
    └── 05_标注修订待执行/            # 6 items — apply annotation corrections
        ├── images/
        └── corrections.csv

Usage::

    python scripts/build_a2_review_package.py
    python scripts/build_a2_review_package.py --dry-run
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
A2_DIR = PROJECT_ROOT / "output" / "7.14训练结果"
FILTERED_IMAGES = A2_DIR / "筛选图片"
RESULTS_DIR = A2_DIR / "整理结果"
OUTPUT_DIR = PROJECT_ROOT / "output" / "A2人工审查包"

# Source CSVs → package subdirectories
PACKAGES = [
    ("08_未完成或结论缺失.csv", "01_未完成或结论缺失",
     "以下样本在A2审查中未获得结论。请逐张查看图片，填写 error_type 和 state_label 字段。\n"
     "error_type 可选值: texture_fp, structure_fp, class_confusion, localization_error, "
     "missing_annotation, ambiguous_state, true_fn\n"
     "state_label 可选值: normal, missing, loose, broken, other"),
    ("10_待完成二审.csv", "02_待完成二审",
     "以下样本已完成一审但缺少二审签字。请作为第二审查员复核一审结论，"
     "在 second_reviewer 和 second_conclusion 字段中填写您的工号和结论。\n"
     "如同意一审结论，second_conclusion 填写 'agree'；如不同意，填写正确结论。"),
    ("06_争议隔离与人工复核.csv", "03_争议隔离",
     "以下样本存在争议（一审与自动审查结论不一致）。需要两名审查员独立判定后取共识。\n"
     "请在 reviewer_1, conclusion_1, reviewer_2, conclusion_2 字段中分别填写。"),
    ("07_结论异常需复核.csv", "04_结论异常需复核",
     "以下样本的审查结论与预期模式不符（如高置信度TP被标为normal）。请重新审查并修正结论。"),
    ("04_标注修订.csv", "05_标注修订待执行",
     "以下6个样本需要标注修订。请确认修订内容后，运行 merge_a2_review_data.py 执行修订。\n"
     "annotation_action 可选值: add_label, correct_bbox, correct_class"),
]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def find_image(review_id: str, event_type: str, crop_id: str) -> Path | None:
    """Find the corresponding filtered image."""
    idx = review_id.replace("A2-", "").lstrip("0") or "0"
    idx_str = idx.zfill(5)

    matches = list(FILTERED_IMAGES.glob(f"{idx_str}_{event_type}_*.jpg"))
    if matches:
        return matches[0]
    matches = list(FILTERED_IMAGES.glob(f"*_{crop_id}.jpg"))
    if matches:
        return matches[0]
    return None


def build_package(dry_run: bool = False) -> None:
    print("=" * 60)
    print("  Build A2 Human Review Package")
    print("=" * 60)
    print(f"  Source:  {RESULTS_DIR}")
    print(f"  Images:  {FILTERED_IMAGES}")
    print(f"  Output:  {OUTPUT_DIR}")
    print(f"  Dry run: {dry_run}")
    print()

    total_items = 0
    total_images = 0

    for csv_name, subdir, instructions in PACKAGES:
        csv_path = RESULTS_DIR / csv_name
        rows = read_csv(csv_path)
        pkg_dir = OUTPUT_DIR / subdir
        img_dir = pkg_dir / "images"

        print(f"  ── {subdir}: {len(rows)} 条 ──")

        if not rows:
            print(f"    [SKIP] {csv_name} not found or empty")
            continue

        if not dry_run:
            pkg_dir.mkdir(parents=True, exist_ok=True)
            img_dir.mkdir(parents=True, exist_ok=True)

        # Copy images
        copied = 0
        missing = 0
        for row in rows:
            review_id = row.get("review_id", "")
            event_type = row.get("event_type", "FP")
            crop_id = row.get("crop_id", "")

            img_path = find_image(review_id, event_type, crop_id)
            if img_path and img_path.exists():
                if not dry_run:
                    shutil.copy2(img_path, img_dir / img_path.name)
                copied += 1
            else:
                missing += 1

        # Copy CSV as review sheet
        if not dry_run:
            sheet_name = "corrections.csv" if "标注修订" in subdir else "review_sheet.csv"
            shutil.copy2(csv_path, pkg_dir / sheet_name)

            # Write README
            readme = pkg_dir / "README.md"
            readme.write_text(
                f"# {subdir}\n\n"
                f"**样本数量**: {len(rows)}\n"
                f"**图片已复制**: {copied}\n"
                f"**图片缺失**: {missing}\n\n"
                f"## 审查说明\n\n{instructions}\n\n"
                f"## 文件说明\n\n"
                f"- `images/` — 待审查的裁剪图片\n"
                f"- `{sheet_name}` — 审查表格（CSV格式，可用Excel打开）\n\n"
                f"## 完成后\n\n"
                f"请将填写完成的 CSV 文件放回此目录，然后通知开发人员执行数据合并。\n",
                encoding="utf-8",
            )

        total_items += len(rows)
        total_images += copied
        print(f"    图片: {copied} copied, {missing} missing")

    # Write top-level README
    if not dry_run:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "README.md").write_text(
            "# A2 人工审查包\n\n"
            f"**生成日期**: 2026-07-25\n"
            f"**总待审样本**: {total_items}\n"
            f"**总图片数**: {total_images}\n\n"
            "## 目录结构\n\n"
            "| 子目录 | 样本数 | 说明 |\n"
            "|--------|:------:|------|\n"
            "| 01_未完成或结论缺失 | 75 | 填写缺失的审查结论 |\n"
            "| 02_待完成二审 | 240 | 第二审查员签字 |\n"
            "| 03_争议隔离 | 200 | 双人独立判定 |\n"
            "| 04_结论异常需复核 | 55 | 重新审查异常结论 |\n"
            "| 05_标注修订待执行 | 6 | 确认标注修订内容 |\n\n"
            "## 审查流程\n\n"
            "1. 打开各子目录中的 `review_sheet.csv`（可用 Excel 打开）\n"
            "2. 对照 `images/` 中的图片逐张审查\n"
            "3. 填写相应字段后保存 CSV\n"
            "4. 通知开发人员执行数据合并和训练集更新\n\n"
            "## 注意事项\n\n"
            "- 审查标准参照 SPECIFICATION.md 中的缺陷定义\n"
            "- 争议样本需两名审查员独立判定，一致率需 ≥90%\n"
            "- 标注修订需确认 bbox 坐标正确后再执行\n",
            encoding="utf-8",
        )

    print(f"\n  {'='*50}")
    print(f"  Total: {total_items} items, {total_images} images")
    if dry_run:
        print(f"  [DRY-RUN] Would create package at: {OUTPUT_DIR}")
    else:
        print(f"  Package created at: {OUTPUT_DIR}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build A2 human review package")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only")
    args = parser.parse_args()
    build_package(dry_run=args.dry_run)