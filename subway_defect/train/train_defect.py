#!/usr/bin/env python3
"""Train the defect detection model — multi-stage training pipeline.

Supports two modes:

**Legacy 3-stage** (backward compatible)::

    C1 — Head Warmup (frozen backbone, 50 epochs)
    C2 — Full Training (heavy augmentation, 200 epochs)
    C3 — Fine-Tune (mild augmentation, 50 epochs)

**Modern 5-stage** (recommended)::

    Stage 0 — Data Sanity Check (optional)
    Stage 1 — Neck + Head Warmup (frozen backbone early, AdamW)
    Stage 2 — Small-Object Scale Adaptation (full unfreeze, 1280 crops)
    Stage 3 — Short Fine-Tune (20-40 epochs, early stopping)
    Stage 4 — Hard Negative Mining + Threshold Calibration

Output layout::

    output/<timestamp>/
        stage0_sanity/         (if run)
        stage1_warmup/          or  c1_warmup/
        stage2_adaptation/      or  c2_full/
        stage3_finetune/        or  c3_finetune/
        stage4_hard_negative/   (if run)

Training hyperparameters are loaded from ``config/train/<stage>.yaml``
(legacy) or ``config/train/pretrain/<stage>.yaml`` (modern).
Hardware tuning (batch/workers/cache) is applied automatically at runtime.

Usage::

    # Legacy 3-stage with COCO pretrain
    python -m subway_defect.train.train_defect \\
        --data datasets/defects/defect_data.yaml --coco_pretrain --device 0

    # Modern 5-stage with P2 model
    python -m subway_defect.train.train_defect \\
        --data data/subway_crops/subway_crops.yaml \\
        --model subway_defect/models/yolo11s-P2-EMA-SimAM.yaml \\
        --coco_pretrain --device 0 --stages 1 2 3 --pretrain-config-dir

    # Single stage only
    train-defect --data ... --pretrained <ckpt.pt> --stages 3

    # Include hard negative mining
    train-defect --data ... --stages 1 2 3 4
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from subway_defect import PROJECT_ROOT
from subway_yolo import YOLO

from subway_defect.train.configs import (
    COCO_PRETRAINED,
    HardwareProfile,
    apply_hardware_profile,
    load_train_config,
)

logger = logging.getLogger(__name__)

# ── Stage definitions ─────────────────────────────────────────────────

# Legacy 3-stage (backward compatible)
_LEGACY_STAGES: Dict[str, Dict] = {
    "c1": {"label": "C1: Head Warmup (50 epochs, frozen backbone)", "vram_gb": 4.0,
           "config_key": "warmup", "name": "c1_warmup"},
    "c2": {"label": "C2: Full Training (200 epochs, heavy augmentation)", "vram_gb": 8.0,
           "config_key": "full", "name": "c2_full"},
    "c3": {"label": "C3: Fine-Tune (50 epochs, mild augmentation)", "vram_gb": 6.0,
           "config_key": "finetune", "name": "c3_finetune"},
}

# Unified training stages (recommended)
_UNIFIED_STAGES: Dict[str, Dict] = {
    "0":  {"label": "Stage 0 (optional): Data Sanity Check",       "vram_gb": 4.0,
           "config_key": None, "name": "stage0_sanity"},
    "p2": {"label": "Stage P2 (optional): TT100K P2 Head Warmup",  "vram_gb": 6.0,
           "config_key": "stage_p2_tiny_pretrain", "name": "stage_p2"},
    "1":  {"label": "Stage 1: Public Defect Pretraining",          "vram_gb": 8.0,
           "config_key": "stage1_public_pretrain", "name": "stage1_public"},
    "2":  {"label": "Stage 2: Custom Domain Adaptation",           "vram_gb": 6.0,
           "config_key": "stage2_domain_adapt", "name": "stage2_adapt"},
    "3":  {"label": "Stage 3: Main Training (1280 native)",        "vram_gb": 10.0,
           "config_key": "stage3_main_training", "name": "stage3_main"},
    "4":  {"label": "Stage 4: Short Fine-Tune",                    "vram_gb": 8.0,
           "config_key": "stage4_short_finetune", "name": "stage4_finetune"},
    "5":  {"label": "Stage 5 (optional): Hard Negative Mining",    "vram_gb": 8.0,
           "config_key": "stage5_hard_negative", "name": "stage5_hard_neg"},
}

# Legacy alias for backward compatibility
_MODERN_STAGES = _UNIFIED_STAGES

from subway_defect.classes import TRAIN_CLASSES as DEFECT_CLASS_NAMES
# Keep the local alias for backward compatibility
# DEFECT_CLASS_NAMES = TRAIN_CLASSES


# ── Helpers ──────────────────────────────────────────────────────────

def _check_gpu_memory(required_gb: float = 6.0) -> None:
    """Warn if GPU free memory is below *required_gb* before training starts."""
    try:
        import torch
    except ImportError:
        logger.warning("PyTorch not available — skipping GPU memory check")
        return
    try:
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info(0)
        free_gb = free / (1024 ** 3)
        if free_gb < required_gb:
            warnings.warn(
                f"GPU free memory is low ({free_gb:.1f} GB / {total / (1024 ** 3):.1f} GB "
                f"total). Training may OOM. Close other GPU processes or reduce --batch.",
                stacklevel=2,
            )
    except RuntimeError as exc:
        logger.warning("GPU memory check failed (CUDA runtime error): %s", exc)


def _cleanup_gpu(model_obj) -> None:
    """Release GPU resources held by a model instance."""
    try:
        import torch
        del model_obj
        torch.cuda.empty_cache()
    except Exception:
        pass


def _resolve_pretrained(args, coco_pretrain_map: dict) -> Optional[str]:
    """Resolve pretrained weights path with fallback to ``weights/``.

    Returns the resolved path string, or ``None`` if explicit pretrained
    was requested but not found.
    """
    pretrained = args.pretrained
    if args.coco_pretrain and not pretrained:
        for key, weight in coco_pretrain_map.items():
            if key in str(args.model):
                pretrained = weight
                logger.info("Auto COCO pretrain: %s", weight)
                break
        if not pretrained:
            pretrained = "yolo11s.pt"
            logger.info("Default COCO pretrain: %s", pretrained)

    if not pretrained:
        return None

    pt_path = Path(pretrained)
    if pt_path.exists():
        return str(pt_path)

    # Try weights/ directory
    yolo_w = Path("weights") / pt_path.name
    if yolo_w.exists():
        logger.info("Found pretrained weights: %s", yolo_w)
        return str(yolo_w)

    logger.error("Pretrained weights not found: %s", pt_path.name)
    logger.error("       Place the file in weights/ or use --pretrained <path>")
    return None


def _validate_model_path(model: str) -> None:
    """Exit early if the model YAML file does not exist."""
    if model.endswith((".yaml", ".yml")) and not Path(model).exists():
        logger.error("Model config not found: %s", model)
        sys.exit(1)


def _apply_overrides(config: dict, args) -> dict:
    """Apply CLI argument overrides to a training config."""
    if args.workers is not None:
        config["workers"] = args.workers
    if args.batch is not None:
        config["batch"] = args.batch
    if args.no_amp:
        config["amp"] = False
    return config


def _load_stage_config(stage_key: str, use_pretrain_dir: bool = False) -> dict:
    """Load training config for a stage from YAML.

    Args:
        stage_key: Stage identifier (``"warmup"``, ``"stage1_neck_head_warmup"``, etc.).
        use_pretrain_dir: If True, load from ``config/train/pretrain/``.

    Returns:
        Config dict.
    """
    if use_pretrain_dir:
        from subway_defect.train.configs import _CONFIG_DIR, _safe_load_yaml
        path = _CONFIG_DIR / "train" / "pretrain" / f"{stage_key}.yaml"
        config = _safe_load_yaml(path)
        if config:
            return config
        logger.warning("Pretrain config not found: %s — falling back to legacy", path)

    # Fall back to legacy config
    return load_train_config(
        stage_key.replace("stage1_", "").replace("stage2_", "").replace(
            "stage3_", "").replace("stage4_", "")
    ) if not use_pretrain_dir else {}


# ── Stage runners ────────────────────────────────────────────────────

def _run_training_stage(
    stage_key: str,
    stage_info: Dict,
    base_config: dict,
    profile: "HardwareProfile",
    args,
    ckpt_in: Path,
    class_names: List[str],
) -> Path:
    """Run a single training stage and return the path to best.pt.

    Each stage may use a **different dataset** — the stage config YAML can
    specify its own ``data`` / ``nc`` / ``names`` fields. These take
    precedence over the CLI ``--data`` argument, enabling multi-source
    pretraining flows like:

        COCO pretrain → public_defect (generic_defect) → subway_crops (7 classes)

    Args:
        stage_key: Stage identifier string.
        stage_info: Stage metadata dict (label, vram_gb, config_key, name).
        base_config: Base config dict (data, device, project) — ``data`` here
            is the CLI default, overridden if the stage YAML has its own.
        profile: :class:`HardwareProfile` for batch/worker tuning.
        args: Parsed CLI arguments.
        ckpt_in: Path to the checkpoint/model to load.
        class_names: List of class name strings (used as fallback).

    Returns:
        Path to ``best.pt`` produced by this stage.
    """
    label = stage_info["label"]
    config_key = stage_info["config_key"]
    stage_name = stage_info["name"]

    logger.info("=" * 66)
    logger.info("  %s", label)
    logger.info("=" * 66)

    # Load stage-specific config
    use_pretrain = getattr(args, "pretrain_config_dir", False)
    if config_key:
        stage_config = _load_stage_config(config_key, use_pretrain)
    else:
        stage_config = {}

    if not stage_config:
        logger.warning("No config for stage %s — using defaults from base", stage_key)
        stage_config = {}

    # ── Per-stage dataset resolution ──────────────────────────────
    # Stage YAML can specify its own ``data``, ``nc``, ``names`` — these
    # take priority over the CLI --data argument. This allows each
    # pretraining stage to use a completely different dataset.
    stage_data = stage_config.pop("data", None) if "data" in stage_config else None
    stage_nc = stage_config.pop("nc", None) if "nc" in stage_config else None
    stage_names = stage_config.pop("names", None) if "names" in stage_config else None

    # Build merged config: stage YAML ← base (data/device/project) ← stage YAML overrides
    config = {**stage_config, **base_config}
    if stage_data is not None:
        config["data"] = stage_data
        logger.info("  [per-stage dataset] %s", stage_data)
    if stage_nc is not None:
        config["nc"] = stage_nc
    if stage_names is not None:
        config["names"] = stage_names
        # If the stage defines different class names, use those for logging
        if isinstance(stage_names, list) and len(stage_names) > 0:
            class_names = stage_names

    config = apply_hardware_profile(config, profile, args.model)
    _apply_overrides(config, args)
    _check_gpu_memory(required_gb=stage_info.get("vram_gb", 6.0))

    # ── Register callbacks ────────────────────────────────────────
    from subway_defect.train.callbacks import register_all_callbacks
    model = YOLO(str(ckpt_in))
    cbs = register_all_callbacks(model, class_names=class_names)
    dynamics, metrics_logger, ckpt_mgr, report, hard_examples = cbs

    try:
        model.train(name=stage_name, **config)
        save_dir = Path(model.trainer.save_dir) if hasattr(model, "trainer") else Path(".")
        best = save_dir / "weights" / "best.pt"
        if not best.exists():
            # Some YOLO versions save best.pt directly in save_dir
            alt_best = save_dir / "best.pt"
            if alt_best.exists():
                best = alt_best
    finally:
        _cleanup_gpu(model)

    logger.info("  %s complete  →  %s", label, best)
    if ckpt_mgr:
        ckpt_summary = ckpt_mgr.summary()
        logger.info("  Checkpoint summary: %s", json.dumps(ckpt_summary, indent=2))

    return best


def _run_sanity_check(
    args,
    class_names: List[str],
    base_config: dict,
) -> None:
    """Stage 0: Data sanity check — label visualization + statistics.

    Not a training stage — outputs a data report to the run directory.
    """
    logger.info("=" * 66)
    logger.info("  Stage 0: Data Sanity Check")
    logger.info("=" * 66)

    import random

    from subway_yolo.data.utils import check_det_dataset

    data_path = args.data
    run_dir = Path(base_config["project"]) / "stage0_sanity"
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Dataset statistics ─────────────────────────────────────────
    try:
        data_info = check_det_dataset(data_path)
        logger.info("Dataset info: %s images, %s classes",
                     data_info.get("train", "?"), data_info.get("nc", "?"))
    except Exception as exc:
        logger.warning("Could not load dataset info: %s", exc)
        data_info = {}

    # ── Label stats ────────────────────────────────────────────────
    label_dir = None
    for key in ("train", "val"):
        path_str = data_info.get(key, "")
        if path_str:
            p = Path(path_str)
            # labels are usually parallel to images
            lbl = p.parent.parent / "labels" / p.name
            if lbl.is_dir():
                label_dir = lbl
                break

    stats: Dict[str, Dict] = {name: {"count": 0, "areas": [], "aspect_ratios": []}
                              for name in class_names}

    if label_dir and label_dir.is_dir():
        label_files = list(label_dir.glob("*.txt"))
        logger.info("Analyzing %s label files...", len(label_files))
        for lf in label_files[:500]:  # sample up to 500
            for line in lf.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    cls_id = int(parts[0])
                    w = float(parts[3])
                    h = float(parts[4])
                except (ValueError, IndexError):
                    continue
                if cls_id < len(class_names):
                    stats[class_names[cls_id]]["count"] += 1
                    stats[class_names[cls_id]]["areas"].append(w * h)
                    stats[class_names[cls_id]]["aspect_ratios"].append(
                        w / h if h > 0 else 1.0
                    )

    # Print statistics
    lines = ["", "  Per-class label statistics:", f"  {'Class':<12s} {'Count':>6s}  "
             f"{'AvgArea':>8s}  {'AvgAR':>7s}  {'MinArea':>8s}  {'MaxArea':>8s}",
             f"  {'-'*60}"]
    for name in class_names:
        s = stats.get(name, {"count": 0, "areas": [], "aspect_ratios": []})
        areas = s["areas"]
        ars = s["aspect_ratios"]
        lines.append(
            f"  {name:<12s} {s['count']:>6d}  "
            f"{sum(areas)/len(areas):>8.4f}" if areas else f"  {name:<12s} {s['count']:>6d}  "
            f"{'N/A':>8s}"
        )
        if areas:
            lines[-1] += f"  {sum(ars)/len(ars):>7.3f}  {min(areas):>8.4f}  {max(areas):>8.4f}"

    for line in lines:
        logger.info(line)

    # Save stats
    (run_dir / "label_statistics.json").write_text(
        json.dumps({k: {"count": v["count"],
                        "avg_area": sum(v["areas"]) / len(v["areas"]) if v["areas"] else 0,
                        "avg_aspect_ratio": sum(v["aspect_ratios"]) / len(v["aspect_ratios"])
                        if v["aspect_ratios"] else 0}
                    for k, v in stats.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Overfit test (20 images) ────────────────────────────────────
    if hasattr(args, "sanity_overfit") and args.sanity_overfit:
        logger.info("Running overfit test on 20 images...")
        # This would create a small subset and train to near-100% mAP
        # Implementation depends on whether we can subset on-the-fly
        logger.info("  (overfit test requires manual setup — skipping auto-run)")

    logger.info("  Sanity check complete → %s", run_dir)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train defect detector (legacy 3-stage or modern 5-stage)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Legacy 3-stage
  train-defect --data datasets/defects/defect_data.yaml --coco_pretrain --device 0

  # Modern 5-stage with P2 model
  train-defect --data data/subway_crops/subway_crops.yaml \\
      --model subway_defect/models/yolo11s-P2-EMA-SimAM.yaml \\
      --coco_pretrain --device 0 --stages 0 1 2 3 --pretrain-config-dir

  # Single stage
  train-defect --data ... --pretrained <ckpt.pt> --stages 3

  # Stage 3 only (short fine-tune)
  train-defect --data ... --pretrained <ckpt.pt> --stages 3
""",
    )
    # ── Required ──
    parser.add_argument("--data", required=True, help="Path to dataset YAML file")

    # ── Model ──
    parser.add_argument(
        "--model", default="subway_defect/models/yolo11s-EMA-SimAM.yaml",
        help="Model YAML or .pt path",
    )

    # ── Training mode ──
    parser.add_argument(
        "--stages", type=str, nargs="*", default=None,
        help="Stages to run: 'c1 c2 c3' (legacy) or '0 1 2 3 4' (modern). "
             "Default: c1 c2 c3 (legacy 3-stage)",
    )
    parser.add_argument(
        "--pretrain-config-dir", action="store_true",
        help="Load stage configs from config/train/pretrain/ (modern 5-stage)",
    )
    parser.add_argument(
        "--coco_pretrain", action="store_true",
        help="Auto-load matching COCO pretrained weights",
    )
    parser.add_argument("--pretrained", default=None,
                        help="Path to .pt weights (overrides --coco_pretrain)")

    # ── Hardware ──
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="defect_detector")
    parser.add_argument("--workers", type=int, default=None,
                        help="Override DataLoader worker count")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP (debug only)")
    parser.add_argument("--vram", type=float, default=None,
                        help="Manually specify GPU VRAM in GB")

    # ── Misc ──
    parser.add_argument("--sanity-overfit", action="store_true",
                        help="Run 20-image overfit test in Stage 0")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing training")

    args = parser.parse_args()

    _validate_model_path(args.model)

    # ── Hardware detection ─────────────────────────────────────────
    profile = HardwareProfile.detect()
    if args.vram is not None:
        profile.vram_gb = args.vram
        logger.info("VRAM manually set to %.1f GB", args.vram)

    # ── Resolve stage plan ─────────────────────────────────────────
    unified_stage_keys = {"0", "p2", "1", "2", "3", "4", "5"}
    legacy_stage_keys = {"c1", "c2", "c3"}

    if args.stages is None:
        # Default: legacy 3-stage (backward compatible)
        logger.warning(
            "No --stages specified — using legacy C1/C2/C3. "
            "For the recommended unified pipeline, use: --stages 1 2 3 4 --pretrain-config-dir"
        )
        stage_plan = [("c1", True), ("c2", True), ("c3", True)]
        stage_map = _LEGACY_STAGES
        use_unified = False
    elif all(s in legacy_stage_keys for s in args.stages):
        # Explicit legacy stage names
        logger.warning(
            "Legacy C1/C2/C3 stage names detected. "
            "Consider migrating to unified stages: --stages 1 2 3 4 --pretrain-config-dir"
        )
        stage_plan = [(s, True) for s in args.stages]
        stage_map = _LEGACY_STAGES
        use_unified = False
        for s in ("c1", "c2", "c3"):
            if s not in args.stages:
                stage_plan.append((s, False))
    else:
        # Unified stages
        stage_plan = [(s, True) for s in args.stages if s in _UNIFIED_STAGES]
        stage_map = _UNIFIED_STAGES
        use_unified = True
        # Fill in skipped for display
        for s in sorted(unified_stage_keys, key=lambda k: (k.isdigit(), k)):
            if s not in args.stages:
                stage_plan.append((s, False))

    # ── Generate timestamp & run directory ─────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "output" / timestamp
    base = {"data": args.data, "device": args.device, "project": str(run_dir)}

    # ── Resolve pretrained weights ─────────────────────────────────
    pretrained = _resolve_pretrained(args, COCO_PRETRAINED)
    if args.pretrained and pretrained is None:
        sys.exit(1)

    # ── Log plan ───────────────────────────────────────────────────
    logger.info("Training plan:")
    for stage_key, enabled in stage_plan:
        info = stage_map[stage_key]
        status = "RUN" if enabled else "SKIP"
        logger.info("  [%s] %s — %s", status, stage_key, info["label"])
    logger.info("  Output: %s", run_dir)
    logger.info("  Model:  %s", args.model)
    logger.info("  Mode:   %s", "unified stages" if use_unified else "legacy 3-stage")

    if args.dry_run:
        logger.info("[DRY-RUN] Exiting without training")
        return

    # ── Execute stages ─────────────────────────────────────────────
    ckpt: Path = Path(pretrained) if pretrained else Path(args.model)

    for stage_key, enabled in stage_plan:
        if not enabled:
            continue

        info = stage_map[stage_key]

        if stage_key == "0" and use_unified:
            # Stage 0: Sanity check (no training)
            _run_sanity_check(args, DEFECT_CLASS_NAMES, base)
        else:
            ckpt = _run_training_stage(
                stage_key=stage_key,
                stage_info=info,
                base_config=base,
                profile=profile,
                args=args,
                ckpt_in=ckpt,
                class_names=DEFECT_CLASS_NAMES,
            )

    logger.info("=" * 66)
    logger.info("  Training complete. Final model: %s", ckpt)
    logger.info("  Output directory: %s", run_dir)


if __name__ == "__main__":
    main()
