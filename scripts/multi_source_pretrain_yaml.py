#!/usr/bin/env python3
"""
Generate training dataset YAML configs for multi-source pretraining phases.

Reads the dataset directories built by ``multi_source_dataset_builder.py`` and
produces YAML configuration files that correspond to each training phase
defined in the multi-source datasets training plan.

Usage (run on AutoDL instance after dataset builder)::

    # Generate all training configs
    python scripts/multi_source_pretrain_yaml.py

    # Generate configs for specific phases
    python scripts/multi_source_pretrain_yaml.py --phases 2 3 4

    # Specify custom dataset root
    python scripts/multi_source_pretrain_yaml.py --root data/multi_datasets

    # Dry-run: print what would be created
    python scripts/multi_source_pretrain_yaml.py --dry-run

Output files (written to config/train/pretrain/)::

    config/train/pretrain/
    ├── phase2_tiny_pretrain.yaml    # TT100K P2 head warmup (optional)
    ├── phase3_public_defect.yaml     # DeepPCB + NEU-DET + GC10-DET (generic_defect)
    ├── phase4_neck_head_adapt.yaml   # Public → custom domain adaptation
    ├── phase5_main_training.yaml     # Full training on custom crops
    └── phase6_short_finetune.yaml    # Short fine-tune with minimal augmentation
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

try:
    import yaml
except ImportError:
    print("ERROR: pyyaml is required. Install with: pip install pyyaml")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────

DEFAULT_DATASET_ROOT = Path("data/multi_datasets")
CONFIG_OUTPUT_DIR = Path("config/train/pretrain")

CUSTOM_CLASSES = [
    "VHBNM",   # 0
    "VHBNL",   # 1
    "SVHBNM",  # 2
    "SVHBNL",  # 3
    "SVHTNL",  # 4
    "CBHPM",   # 5
    "CBVPM",   # 6
]

# ── Helper ─────────────────────────────────────────────────────────────

def _resolve_paths(root: Path, paths: List[str]) -> List[str]:
    """Convert relative paths under *root* to absolute strings, checking existence."""
    resolved: List[str] = []
    for p in paths:
        abs_path = root / p
        if abs_path.exists():
            resolved.append(str(abs_path.resolve()))
        else:
            print(f"  [WARN] Path does not exist: {abs_path}")
    return resolved


def _write_yaml(path: Path, data: dict, dry_run: bool = False) -> None:
    """Write a YAML file with consistent formatting."""
    if dry_run:
        print(f"  [DRY-RUN] Would write: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    # Use block literal style for multi-line descriptions
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    print(f"  [OK] Wrote: {path}")


# ==========================================================================
# Phase config generators
# ==========================================================================

def generate_phase2_tiny_pretrain(root: Path, dry_run: bool = False) -> Optional[Path]:
    """Phase 2 (optional): P2 small-object head pretraining with TT100K.

    Only generated if ``mixed_tiny_pretrain/`` exists under *root*.
    """
    tiny_dir = root / "mixed_tiny_pretrain"
    data_yaml = tiny_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"  [SKIP] Phase 2: {data_yaml} not found -- TT100K not available")
        return None

    config = {
        "path": str(tiny_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["tiny_object"],
        #
        "# ===== Phase 2: P2 Small-Object Head Warmup =====": "",
        "# 目标": "让新增 P2 检测分支先学会小目标定位",
        "# 数据": "TT100K (可选 DeepPCB 替代)",
        "# 模型": "YOLO11s-P2-EMA-SimAM-Lite",
        "# 输入": "1024 或 1280",
        "# 训练": "50–80 epochs",
        "# 输出": "weights/p2_tiny_pretrain.pt",
        "epochs": 80,
        "imgsz": 1024,
        "batch": 16,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.02,
        "warmup_epochs": 5,
        "cos_lr": True,
        "mosaic": 0.2,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.4,
        "degrees": 3.0,
        "translate": 0.1,
        "scale": 0.5,
        "close_mosaic": 20,
    }

    out_path = CONFIG_OUTPUT_DIR / "phase2_tiny_pretrain.yaml"
    _write_yaml(out_path, config, dry_run)
    return out_path


def generate_phase3_public_defect(root: Path, dry_run: bool = False) -> Optional[Path]:
    """Phase 3: Public industrial defect pretraining (generic_defect).

    Merges DeepPCB + NEU-DET + GC10-DET + optional Insulator Defect.
    Uses ``mixed_pretrain/data.yaml`` if available, otherwise builds paths manually.
    """
    mixed_dir = root / "mixed_pretrain"
    data_yaml = mixed_dir / "data.yaml"

    # Collect available public dataset paths
    public_dir = root / "public"
    train_paths: List[str] = []
    val_paths: List[str] = []

    priority_order = ["deeppcb", "gc10_det", "neu_det", "insulator_defect"]
    for key in priority_order:
        ds_dir = public_dir / key
        train_p = ds_dir / "images" / "train"
        val_p = ds_dir / "images" / "val"
        if train_p.is_dir():
            train_paths.append(str(train_p.resolve()))
        if val_p.is_dir():
            val_paths.append(str(val_p.resolve()))

    if not train_paths:
        print(f"  [SKIP] Phase 3: No public datasets found under {public_dir}")
        return None

    # If mixed_pretrain already exists (from builder), use it as single source
    if data_yaml.exists():
        config = {
            "# ===== Phase 3: Public Industrial Defect Pretraining =====": "",
            "# 目标": "让 backbone/neck/P2/P3 学习工业异常纹理",
            "# 初始化": "p2_tiny_pretrain.pt 或 COCO yolo11s.pt",
            "# 数据": "DeepPCB + NEU-DET + GC10-DET + 可选 Insulator (generic_defect)",
            "# 使用方式": f"data: {data_yaml}",
            "path": str(mixed_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["generic_defect"],
            #
            "epochs": 120,
            "imgsz": 1024,
            "batch": 16,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.02,
            "warmup_epochs": 10,
            "weight_decay": 0.0005,
            "cos_lr": True,
            "mosaic": 0.2,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.0,
            "degrees": 5.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 1.0,
            "perspective": 0.0002,
            "close_mosaic": 30,
            #
            "# 训练策略": "",
            "#   前 10 epoch": "冻结 backbone 前半部分，只训 neck + head",
            "#   第 11 epoch 后": "解冻全部，backbone 使用较低学习率",
            "# 输出": "weights/public_defect_pretrain.pt",
            "# 验收": "mAP50 不必追求极致；重点关注训练稳定性、P2/P3 有效性",
            "freeze": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }
    else:
        # Fallback: use individual paths
        if not val_paths:
            val_paths = train_paths  # use same for val if no separate val
        config = {
            "# ===== Phase 3: Public Industrial Defect Pretraining =====": "",
            "# 目标": "让 backbone/neck/P2/P3 学习工业异常纹理",
            "# 初始化": "p2_tiny_pretrain.pt 或 COCO yolo11s.pt",
            "# 数据": "DeepPCB + NEU-DET + GC10-DET (generic_defect)",
            "train": train_paths,
            "val": val_paths[:1] if val_paths else train_paths[:1],
            "nc": 1,
            "names": ["generic_defect"],
            #
            "epochs": 120,
            "imgsz": 1024,
            "batch": 16,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "lrf": 0.02,
            "warmup_epochs": 10,
            "weight_decay": 0.0005,
            "cos_lr": True,
            "mosaic": 0.2,
            "mixup": 0.0,
            "copy_paste": 0.0,
            "erasing": 0.0,
            "degrees": 5.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 1.0,
            "perspective": 0.0002,
            "close_mosaic": 30,
            "freeze": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        }

    out_path = CONFIG_OUTPUT_DIR / "phase3_public_defect.yaml"
    _write_yaml(out_path, config, dry_run)
    return out_path


def generate_phase4_neck_head_adapt(root: Path, dry_run: bool = False) -> Optional[Path]:
    """Phase 4: Domain adaptation -- public defect → custom contact-net 7 classes.

    Uses subway_crops for training. If subway data doesn't exist, creates a template.
    """
    subway_dir = root / "subway_crops"
    train_dir = subway_dir / "train"
    val_dir = subway_dir / "val"

    # Check if we can find custom data elsewhere
    if not train_dir.is_dir():
        # Try project's own dataset
        project_data = Path("data/Defect_dataset")
        if project_data.is_dir():
            train_dir = project_data / "images" / "train"
            val_dir = project_data / "images" / "val"

    if not train_dir.is_dir():
        print(f"  [WARN] Phase 4: subway_crops/train not found at {train_dir}")
        print(f"  [WARN] Phase 4: Generating template -- update paths before training")
        train_path = str(subway_dir.resolve() / "train" / "images")
        val_path = str(subway_dir.resolve() / "val" / "images")
    else:
        train_path = str(train_dir.resolve())
        val_path = str(val_dir.resolve()) if val_dir.is_dir() else train_path

    config = {
        "# ===== Phase 4: Custom Contact-Net Domain Adaptation =====": "",
        "# 目标": "从公开缺陷域切换到真实接触网 7 类",
        "# 初始化": "public_defect_pretrain.pt",
        "# 数据": "subway_crops (1024/1280 原生分辨率 ROI crop)",
        "# 类别": "7 类接触网缺陷",
        "path": str(subway_dir.resolve()) if subway_dir.is_dir() else str(root.resolve()),
        "train": train_path,
        "val": val_path,
        "nc": 7,
        "names": CUSTOM_CLASSES,
        #
        "epochs": 50,
        "imgsz": 1024,
        "batch": 16,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 1.0,
        "warmup_epochs": 3,
        "cos_lr": False,
        "mosaic": 0.1,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.5,
        "hsv_v": 0.5,
        "degrees": 3.0,
        "translate": 0.1,
        "scale": 0.4,
        #
        "# 冻结策略": "冻结 backbone 前 60% 层，训练 neck + P2/P3/P4/P5 + attention",
        "freeze": [0, 1, 2, 3, 4, 5, 6, 7],
        "# 验收": "mAP50 > 0.35, mAP50-95 > 0.25, Recall 明显提升, 各类别 AP != 0",
    }

    out_path = CONFIG_OUTPUT_DIR / "phase4_neck_head_adapt.yaml"
    _write_yaml(out_path, config, dry_run)
    return out_path


def generate_phase5_main_training(root: Path, dry_run: bool = False) -> Optional[Path]:
    """Phase 5: Main training on custom crops at native resolution."""
    subway_dir = root / "subway_crops"
    train_dir = subway_dir / "train"
    val_dir = subway_dir / "val"

    if not train_dir.is_dir():
        # Try project's own dataset
        project_data = Path("data/Defect_dataset")
        if project_data.is_dir():
            train_dir = project_data / "images" / "train"
            val_dir = project_data / "images" / "val"

    if not train_dir.is_dir():
        train_path = str(subway_dir.resolve() / "train" / "images")
        val_path = str(subway_dir.resolve() / "val" / "images")
    else:
        train_path = str(train_dir.resolve())
        val_path = str(val_dir.resolve()) if val_dir.is_dir() else train_path

    config = {
        "# ===== Phase 5: Main Training -- Native Resolution Crops =====": "",
        "# 目标": "小目标尺度适应训练 -- 主训练阶段",
        "# 初始化": "Phase 4 best.pt",
        "# 模型": "YOLO11s-P2-EMA-SimAM-Lite",
        "# 输入": "1280 原生 crop",
        "train": train_path,
        "val": val_path,
        "nc": 7,
        "names": CUSTOM_CLASSES,
        #
        "epochs": 120,
        "imgsz": 1280,
        "batch": 12,
        "optimizer": "AdamW",
        "lr0": 0.0008,
        "lrf": 0.02,
        "warmup_epochs": 8,
        "warmup_momentum": 0.5,
        "weight_decay": 0.0005,
        "cos_lr": True,
        "patience": 40,
        #
        "mosaic": 0.2,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "hsv_h": 0.015,
        "hsv_s": 0.6,
        "hsv_v": 0.5,
        "degrees": 5.0,
        "translate": 0.12,
        "scale": 0.5,
        "shear": 1.0,
        "perspective": 0.0003,
        "close_mosaic": 40,
        #
        "# 多尺度策略": "不建议 multi_scale:true；推荐固定 1280 或自定义 [1024,1280,1536]",
        "# 输出": "weights/phase5_main.pt",
    }

    out_path = CONFIG_OUTPUT_DIR / "phase5_main_training.yaml"
    _write_yaml(out_path, config, dry_run)
    return out_path


def generate_phase6_short_finetune(root: Path, dry_run: bool = False) -> Optional[Path]:
    """Phase 6: Short fine-tune with minimal augmentation."""
    subway_dir = root / "subway_crops"
    train_dir = subway_dir / "train"
    val_dir = subway_dir / "val"

    if not train_dir.is_dir():
        project_data = Path("data/Defect_dataset")
        if project_data.is_dir():
            train_dir = project_data / "images" / "train"
            val_dir = project_data / "images" / "val"

    if not train_dir.is_dir():
        train_path = str(subway_dir.resolve() / "train" / "images")
        val_path = str(subway_dir.resolve() / "val" / "images")
    else:
        train_path = str(train_dir.resolve())
        val_path = str(val_dir.resolve()) if val_dir.is_dir() else train_path

    config = {
        "# ===== Phase 6: Short Fine-Tune -- Minimal Augmentation =====": "",
        "# 目标": "真实分布短微调，早停防退化",
        "# 初始化": "Phase 5 best.pt",
        "train": train_path,
        "val": val_path,
        "nc": 7,
        "names": CUSTOM_CLASSES,
        #
        "epochs": 30,
        "imgsz": 1280,
        "batch": 12,
        "optimizer": "AdamW",
        "lr0": 0.00003,
        "lrf": 1.0,
        "cos_lr": False,
        "patience": 8,
        #
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "degrees": 1.0,
        "translate": 0.05,
        "scale": 0.2,
        "shear": 0.0,
        "perspective": 0.0,
        "hsv_h": 0.005,
        "hsv_s": 0.2,
        "hsv_v": 0.2,
        #
        "# 冻结": "冻结 backbone 前 70%，只微调 neck + head + attention",
        "freeze": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "# 保存策略": "每 epoch 保存；选择 best_mAP50-95.pt 和 best_F2.pt",
        "# 注意": "不要默认使用 last.pt",
    }

    out_path = CONFIG_OUTPUT_DIR / "phase6_short_finetune.yaml"
    _write_yaml(out_path, config, dry_run)
    return out_path


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    global CONFIG_OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Generate training YAML configs for multi-source pretraining phases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/multi_source_pretrain_yaml.py
  python scripts/multi_source_pretrain_yaml.py --phases 2 3 4
  python scripts/multi_source_pretrain_yaml.py --root data/multi_datasets
  python scripts/multi_source_pretrain_yaml.py --dry-run
""",
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--phases", type=int, nargs="*", choices=range(2, 7),
        help="Specific phases to generate configs for (2-6). Default: all available.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print configs without writing files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=CONFIG_OUTPUT_DIR,
        help=f"Output directory for YAML files (default: {CONFIG_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Allow overriding the global output dir
    CONFIG_OUTPUT_DIR = args.output_dir

    print()
    print("=" * 60)
    print("  Multi-Source Pretrain Config Generator")
    print("=" * 60)
    print(f"  Dataset root: {args.root}")
    print(f"  Output dir:   {CONFIG_OUTPUT_DIR}")
    print(f"  Dry run:      {args.dry_run}")
    print()

    phases_to_run = args.phases or [2, 3, 4, 5, 6]
    results: Dict[int, Optional[Path]] = {}

    generators = {
        2: ("Phase 2: P2 Tiny-Object Head Pretrain (TT100K)", generate_phase2_tiny_pretrain),
        3: ("Phase 3: Public Defect Pretrain (DeepPCB+NEU+GC10)", generate_phase3_public_defect),
        4: ("Phase 4: Custom Neck/Head Domain Adaptation", generate_phase4_neck_head_adapt),
        5: ("Phase 5: Main Training (Native Resolution Crops)", generate_phase5_main_training),
        6: ("Phase 6: Short Fine-Tune (Minimal Augmentation)", generate_phase6_short_finetune),
    }

    for phase_num in sorted(phases_to_run):
        if phase_num not in generators:
            continue
        title, gen_func = generators[phase_num]
        print(f"\n{'-' * 60}")
        print(f"  {title}")
        print(f"{'-' * 60}")
        result = gen_func(args.root, args.dry_run)
        results[phase_num] = result

    # Summary
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    for phase_num in sorted(results):
        path = results[phase_num]
        if path:
            print(f"  Phase {phase_num}: {path}")
        else:
            print(f"  Phase {phase_num}: SKIPPED (data not available)")

    print()
    print("  Suggested training order:")
    print("    1. Phase 2 (if TT100K available) -> weights/p2_tiny_pretrain.pt")
    print("    2. Phase 3 -> weights/public_defect_pretrain.pt")
    print("    3. Phase 4 -> weights/neck_head_adapt.pt")
    print("    4. Phase 5 -> weights/main.pt")
    print("    5. Phase 6 -> weights/best_finetune.pt")
    print()
    print("  Usage with custom train script:")
    print("    from config.train.pretrain import phase3_public_defect")
    print("    model.train(data=phase3_public_defect, ...)")
    print()


if __name__ == "__main__":
    main()
