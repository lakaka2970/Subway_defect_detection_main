#!/usr/bin/env python3
"""
Generate training dataset YAML configs for unified training stages.

Reads the dataset directories built by ``multi_source_dataset_builder.py`` and
produces YAML configuration files that correspond to each training stage.

Usage (run on AutoDL instance after dataset builder)::

    # Generate all available training configs
    python scripts/multi_source_pretrain_yaml.py

    # Generate configs for specific stages
    python scripts/multi_source_pretrain_yaml.py --stages p2 1 2

    # Specify custom dataset root
    python scripts/multi_source_pretrain_yaml.py --root data/multi_datasets

    # Dry-run: print what would be created
    python scripts/multi_source_pretrain_yaml.py --dry-run

Output files (written to config/train/pretrain/)::

    config/train/pretrain/
    ├── stage_p2_tiny_pretrain.yaml     # (optional) TT100K P2 head warmup
    ├── stage1_public_pretrain.yaml     # DeepPCB + NEU-DET + GC10-DET
    ├── stage2_domain_adapt.yaml        # Public → custom contact-net adaptation
    ├── stage3_main_training.yaml       # Full training on custom crops
    ├── stage4_short_finetune.yaml      # Short fine-tune, minimal augmentation
    └── stage5_hard_negative.yaml       # Hard negative mining + calibration

Training flow (recommended)::

    python scripts/train_pipeline.py --stages 1 2 3 4
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


# ── YAML writer with real comments ─────────────────────────────────────

def _write_yaml(
    path: Path,
    data: dict,
    header: Optional[str] = None,
    dry_run: bool = False,
) -> None:
    """Write a YAML file with a leading comment block.

    Args:
        path: Output file path.
        data: Clean config dict (no pseudo-comment keys).
        header: Optional multi-line header string written as YAML comments
            (each line prefixed with ``# ``).
        dry_run: If True, print what would be written instead of writing.
    """
    if dry_run:
        print(f"  [DRY-RUN] Would write: {path}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        if header:
            for line in header.strip().split("\n"):
                line = line.strip()
                f.write(f"# {line}\n" if line else "#\n")
            f.write("\n")
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"  [OK] {path.name}")


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


# ==========================================================================
# Stage config generators
# ==========================================================================

def generate_stage_p2_tiny_pretrain(root: Path, dry_run: bool = False) -> Optional[Path]:
    """Stage P2 (optional): P2 small-object head pretraining with TT100K.

    Only generated if ``mixed_tiny_pretrain/`` exists under *root*.
    """
    tiny_dir = root / "mixed_tiny_pretrain"
    data_yaml = tiny_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"  [SKIP] Stage P2: {data_yaml} not found -- TT100K not available")
        return None

    header = """\
Stage P2 (optional): TT100K P2 Small-Object Head Warmup
=======================================================
Target: Train the newly added P2 detection branch on tiny objects first.
Dataset: TT100K (optional DeepPCB as substitute).
Model: YOLO11s-P2-EMA-SimAM (only for P2 four-scale models).
Input: 1024.
Training: 80 epochs.
Output: weights/stage_p2_tiny_pretrain.pt.

If you are NOT using a P2 model, skip this stage.
"""

    config = {
        "path": str(tiny_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "nc": 1,
        "names": ["tiny_object"],
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

    out_path = CONFIG_OUTPUT_DIR / "stage_p2_tiny_pretrain.yaml"
    _write_yaml(out_path, config, header=header, dry_run=dry_run)
    return out_path


def generate_stage1_public_pretrain(root: Path, dry_run: bool = False) -> Optional[Path]:
    """Stage 1: Public industrial defect pretraining (generic_defect).

    Merges DeepPCB + NEU-DET + GC10-DET + optional Insulator Defect.
    """
    mixed_dir = root / "mixed_pretrain"
    data_yaml = mixed_dir / "data.yaml"

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
        print(f"  [SKIP] Stage 1: No public datasets found under {public_dir}")
        return None

    header = """\
Stage 1: Public Industrial Defect Pretraining
=============================================
Target: Let backbone/neck/P2/P3 learn industrial anomaly textures.
Init: stage_p2_tiny_pretrain.pt (if P2 model) or COCO yolo11s.pt.
Datasets: DeepPCB + NEU-DET + GC10-DET + optional Insulator.
Classes: 1 (generic_defect) — model learns "where is the anomaly", not "what kind".

Training strategy:
  - First 10 epochs: freeze backbone first half, train neck + head only.
  - After epoch 10: unfreeze all, lower LR for backbone.
Output: weights/stage1_public_pretrain.pt.
Acceptance: mAP50 doesn't need to be extreme; focus on training stability and P2/P3 effectiveness.
"""

    if data_yaml.exists():
        config = {
            "path": str(mixed_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "nc": 1,
            "names": ["generic_defect"],
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
    else:
        if not val_paths:
            val_paths = train_paths
        config = {
            "train": train_paths,
            "val": val_paths[:1] if val_paths else train_paths[:1],
            "nc": 1,
            "names": ["generic_defect"],
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

    out_path = CONFIG_OUTPUT_DIR / "stage1_public_pretrain.yaml"
    _write_yaml(out_path, config, header=header, dry_run=dry_run)
    return out_path


def generate_stage2_domain_adapt(root: Path, dry_run: bool = False) -> Optional[Path]:
    """Stage 2: Domain adaptation — public defect → custom contact-net 7 classes."""
    subway_dir = root / "subway_crops"
    train_dir = subway_dir / "train"
    val_dir = subway_dir / "val"

    if not train_dir.is_dir():
        project_data = Path("data/Defect_dataset")
        if project_data.is_dir():
            train_dir = project_data / "images" / "train"
            val_dir = project_data / "images" / "val"

    if not train_dir.is_dir():
        print(f"  [WARN] Stage 2: subway_crops/train not found at {train_dir}")
        print(f"  [WARN] Stage 2: Generating template — update paths before training")
        train_path = str(subway_dir.resolve() / "train" / "images")
        val_path = str(subway_dir.resolve() / "val" / "images")
    else:
        train_path = str(train_dir.resolve())
        val_path = str(val_dir.resolve()) if val_dir.is_dir() else train_path

    header = """\
Stage 2: Custom Contact-Net Domain Adaptation
=============================================
Target: Transfer from public defect domain to real contact-net 7 class defects.
Init: stage1_public_pretrain.pt.
Data: subway_crops (1024 native resolution ROI crops).
Classes: 7 contact-net defect classes.

Strategy: Freeze backbone first 60% layers, train neck + P3/P4/P5 + attention.
Acceptance: mAP50 > 0.35, mAP50-95 > 0.25, all per-class AP > 0.
"""

    config = {
        "path": str(subway_dir.resolve()) if subway_dir.is_dir() else str(root.resolve()),
        "train": train_path,
        "val": val_path,
        "nc": 7,
        "names": CUSTOM_CLASSES,
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
        "freeze": [0, 1, 2, 3, 4, 5, 6, 7],
    }

    out_path = CONFIG_OUTPUT_DIR / "stage2_domain_adapt.yaml"
    _write_yaml(out_path, config, header=header, dry_run=dry_run)
    return out_path


def generate_stage3_main_training(root: Path, dry_run: bool = False) -> Optional[Path]:
    """Stage 3: Main training on custom crops at native resolution."""
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

    header = """\
Stage 3: Main Training — Native Resolution Crops
================================================
Target: Small-object scale adaptation — the main training stage.
Init: Stage 2 best.pt.
Model: YOLO11s-P2-EMA-SimAM (recommended) or yolo11s-EMA-SimAM.
Input: 1280 native crop (fixed; NO multi_scale ±50% to protect small objects).
Classes: 7 contact-net defect classes.

Key differences from legacy C2:
  - AdamW instead of low-LR SGD.
  - No erasing/copy_paste (protects small objects).
  - mosaic=0.2 (less distortion for small targets).
  - close_mosaic=40 (33% of training on clean real images).
  - patience=40 (prevents overfitting in later epochs).
Output: weights/stage3_main.pt.
"""

    config = {
        "train": train_path,
        "val": val_path,
        "nc": 7,
        "names": CUSTOM_CLASSES,
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
        "flipud": 0.0,
        "fliplr": 0.5,
        "close_mosaic": 40,
        "auto_augment": "randaugment",
    }

    out_path = CONFIG_OUTPUT_DIR / "stage3_main_training.yaml"
    _write_yaml(out_path, config, header=header, dry_run=dry_run)
    return out_path


def generate_stage4_short_finetune(root: Path, dry_run: bool = False) -> Optional[Path]:
    """Stage 4: Short fine-tune with minimal augmentation."""
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

    header = """\
Stage 4: Short Fine-Tune — Minimal Augmentation
===============================================
Target: Converge to real distribution with early stopping to prevent degradation.
Init: Stage 3 best.pt.
Classes: 7 contact-net defect classes.

Strategy:
  - Freeze backbone first 70%, fine-tune only neck + head + attention.
  - Zero augmentation (mosaic=0, erasing=0).
  - Very low LR (3e-5).
  - patience=8 — stops early when overfitting begins.
  - Save every epoch; select best_mAP50-95.pt.

DO NOT default to last.pt — always use best_mAP50-95.pt for deployment.
Output: weights/stage4_best_finetune.pt.
"""

    config = {
        "train": train_path,
        "val": val_path,
        "nc": 7,
        "names": CUSTOM_CLASSES,
        "epochs": 30,
        "imgsz": 1280,
        "batch": 12,
        "optimizer": "AdamW",
        "lr0": 0.00003,
        "lrf": 1.0,
        "cos_lr": False,
        "patience": 8,
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
        "freeze": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "save_period": 1,
    }

    out_path = CONFIG_OUTPUT_DIR / "stage4_short_finetune.yaml"
    _write_yaml(out_path, config, header=header, dry_run=dry_run)
    return out_path


def generate_stage5_hard_negative(root: Path, dry_run: bool = False) -> Optional[Path]:
    """Stage 5 (optional): Hard negative mining + per-class threshold calibration.

    Uses Stage 4 model to collect false-positive crops from training/validation
    sets, then retrains with minimal augmentation to reduce false alarms.
    """
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

    header = """\
Stage 5 (optional): Hard Negative Mining + Threshold Calibration
================================================================
Target: Reduce false positives and calibrate per-class confidence thresholds.
Init: Stage 4 best.pt.
Classes: 7 contact-net defect classes.

Strategy:
  - Use Stage 4 model to collect false-positive crops from train + val sets.
  - Add hard negative crops to training set.
  - Retrain with zero augmentation, very low LR (2e-5).
  - Per-class threshold search (precision/recall dual objective).
  - Freeze backbone first 70%.

Only run this stage if Stage 4 precision is below target (90%).
Output: weights/stage5_calibrated.pt.
"""

    config = {
        "train": train_path,
        "val": val_path,
        "nc": 7,
        "names": CUSTOM_CLASSES,
        "epochs": 30,
        "imgsz": 1280,
        "batch": 12,
        "optimizer": "AdamW",
        "lr0": 0.00002,
        "lrf": 1.0,
        "cos_lr": False,
        "patience": 10,
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "freeze": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        "save_period": 1,
    }

    out_path = CONFIG_OUTPUT_DIR / "stage5_hard_negative.yaml"
    _write_yaml(out_path, config, header=header, dry_run=dry_run)
    return out_path


# ==========================================================================
# Main
# ==========================================================================

def main() -> None:
    global CONFIG_OUTPUT_DIR

    parser = argparse.ArgumentParser(
        description="Generate training YAML configs for unified training stages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/multi_source_pretrain_yaml.py
  python scripts/multi_source_pretrain_yaml.py --stages p2 1 2
  python scripts/multi_source_pretrain_yaml.py --root data/multi_datasets
  python scripts/multi_source_pretrain_yaml.py --dry-run
""",
    )
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_DATASET_ROOT,
        help=f"Dataset root directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--stages", type=str, nargs="*", choices=["p2", "1", "2", "3", "4", "5"],
        help="Specific stages to generate configs for (p2, 1-5). Default: all available.",
    )
    parser.add_argument(
        "--phases", type=str, nargs="*", dest="stages_deprecated",
        help="Deprecated: use --stages instead.",
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

    # Handle --phases deprecation
    stages_to_run: List[str] = args.stages if args.stages else (
        args.stages_deprecated if args.stages_deprecated else None
    )

    CONFIG_OUTPUT_DIR = args.output_dir

    print()
    print("=" * 60)
    print("  Unified Stage Config Generator")
    print("=" * 60)
    print(f"  Dataset root: {args.root}")
    print(f"  Output dir:   {CONFIG_OUTPUT_DIR}")
    print(f"  Dry run:      {args.dry_run}")
    print()

    generators = {
        "p2": ("Stage P2 (optional): P2 TT100K Tiny-Object Head Warmup", generate_stage_p2_tiny_pretrain),
        "1":  ("Stage 1: Public Defect Pretraining (DeepPCB+NEU+GC10)", generate_stage1_public_pretrain),
        "2":  ("Stage 2: Custom Domain Adaptation (7 classes)", generate_stage2_domain_adapt),
        "3":  ("Stage 3: Main Training (1280 native crops)", generate_stage3_main_training),
        "4":  ("Stage 4: Short Fine-Tune (minimal augmentation)", generate_stage4_short_finetune),
        "5":  ("Stage 5 (optional): Hard Negative Mining + Calibration", generate_stage5_hard_negative),
    }

    # Default: all available stages
    stages_to_run = stages_to_run or list(generators.keys())

    results: Dict[str, Optional[Path]] = {}

    for stage_key in stages_to_run:
        if stage_key not in generators:
            print(f"  [SKIP] Unknown stage: {stage_key}")
            continue
        title, gen_func = generators[stage_key]
        print(f"\n{'-' * 60}")
        print(f"  {title}")
        print(f"{'-' * 60}")
        result = gen_func(args.root, args.dry_run)
        results[stage_key] = result

    # Summary
    print(f"\n{'=' * 60}")
    print("  Summary")
    print(f"{'=' * 60}")
    for stage_key in sorted(results, key=lambda k: (k.isdigit(), k)):
        path = results[stage_key]
        if path:
            print(f"  Stage {stage_key}: {path}")
        else:
            print(f"  Stage {stage_key}: SKIPPED (data not available)")

    print()
    print("  Recommended training flow:")
    print("    1. Stage 1 -> weights/stage1_public_pretrain.pt")
    print("    2. Stage 2 -> weights/stage2_domain_adapt.pt")
    print("    3. Stage 3 -> weights/stage3_main.pt")
    print("    4. Stage 4 -> weights/stage4_best_finetune.pt")
    print()
    print("  One-command training:")
    print("    python scripts/train_pipeline.py --stages 1 2 3 4")
    print()


if __name__ == "__main__":
    main()
