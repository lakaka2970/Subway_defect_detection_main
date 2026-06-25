#!/usr/bin/env python3
"""Train the defect detection model — Stage C (3 sub-stages).

C1 — Head warmup (frozen backbone, 50 epochs)
C2 — Full training (heavy augmentation, 200 epochs)
C3 — Fine-tune (mild augmentation, 50 epochs)

Output layout::

    output/<timestamp>/
        c1_warmup/
        c2_full/
        c3_finetune/

Training hyperparameters are loaded from ``config/train/<stage>.yaml``.
Hardware tuning (batch/workers/cache) is applied automatically at runtime.

Stages run sequentially by default. Use --skip_* flags to control:

    # C1 only (head warmup, verify mAP50 > 0.30)
    train-defect --data ... --coco_pretrain --skip_full --skip_finetune

    # C2 only (full training from C1 checkpoint)
    train-defect --data ... --pretrained <c1_best.pt> --skip_warmup --skip_finetune

    # C3 only (fine-tune from C2 checkpoint)
    train-defect --data ... --pretrained <c2_best.pt> --skip_warmup --skip_full

    # Full pipeline (C1 → C2 → C3)
    train-defect --data ... --coco_pretrain

Usage:
    python -m subway_defect.train.train_defect --data datasets/defects/defect_data.yaml --device 0
"""

import argparse
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

from subway_yolo import YOLO

from subway_defect.train.configs import (
    COCO_PRETRAINED,
    HardwareProfile,
    apply_hardware_profile,
    load_train_config,
)

logger = logging.getLogger(__name__)

# ── Per-stage constants ─────────────────────────────────────────────
_STAGE_CONFIGS = {
    "warmup":   {"label": "C1: Head Warmup (50 epochs, frozen backbone)", "vram_gb": 4.0},
    "full":     {"label": "C2: Full Training (200 epochs, heavy augmentation)", "vram_gb": 8.0},
    "finetune": {"label": "C3: Fine-Tune (50 epochs, mild augmentation)", "vram_gb": 6.0},
}


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
                f"GPU free memory is low ({free_gb:.1f} GB / {total / (1024 ** 3):.1f} GB total). "
                f"Training may OOM. Close other GPU processes or reduce --batch.",
                stacklevel=2,
            )
    except RuntimeError as e:
        logger.warning("GPU memory check failed (CUDA runtime error): %s", e)


def _cleanup_gpu(model_obj) -> None:
    """Release GPU resources held by a model instance."""
    try:
        import torch
        del model_obj
        torch.cuda.empty_cache()
    except Exception:
        pass


def _resolve_pretrained(args, coco_pretrain_map: dict) -> Optional[str]:
    """Resolve pretrained weights path with fallback to yolo_weights/.

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

    # Try yolo_weights/ directory
    yolo_w = Path("yolo_weights") / pt_path.name
    if yolo_w.exists():
        logger.info("Found pretrained weights: %s", yolo_w)
        return str(yolo_w)

    logger.error("Pretrained weights not found: %s", pt_path.name)
    logger.error("       Place the file in yolo_weights/ or use --pretrained <path>")
    return None  # Caller must check for None and exit


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


def _run_stage(stage_key: str, config: dict, profile, args, ckpt_in: Path) -> Path:
    """Run a single training stage and return the path to best.pt.

    Args:
        stage_key: One of ``"warmup"``, ``"full"``, ``"finetune"``.
        config: Base training config (from YAML).
        profile: :class:`HardwareProfile` used for batch/worker tuning.
        args: Parsed CLI arguments.
        ckpt_in: Path to the checkpoint to load for this stage
            (pretrained .pt or model .yaml for the first stage).

    Returns:
        Path to ``best.pt`` produced by this stage.
    """
    info = _STAGE_CONFIGS[stage_key]
    logger.info("=" * 60)
    logger.info("Stage %s", info["label"])
    logger.info("=" * 60)

    config = apply_hardware_profile({**load_train_config(stage_key), **config}, profile, args.model)
    _apply_overrides(config, args)
    _check_gpu_memory(required_gb=info["vram_gb"])

    model = YOLO(str(ckpt_in))
    try:
        model.train(name=f"c{list(_STAGE_CONFIGS).index(stage_key) + 1}_{stage_key}", **config)
        best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    finally:
        _cleanup_gpu(model)

    return best


def main():
    parser = argparse.ArgumentParser(description="Train defect detector")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="subway_defect/models/yolo11s-EMA-SimAM.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="defect_detector")
    parser.add_argument("--pretrained", default=None,
                        help="Path to .pt weights (e.g. yolo11s.pt for COCO pretrain)")
    parser.add_argument("--coco_pretrain", action="store_true",
                        help="Auto-load matching COCO pretrained weights")
    parser.add_argument("--skip_warmup", action="store_true",
                        help="Skip C1 warmup (use when loading pretrained weights)")
    parser.add_argument("--skip_full", action="store_true",
                        help="Skip C2 full training stage")
    parser.add_argument("--skip_finetune", action="store_true",
                        help="Skip C3 fine-tune stage")

    # Hardware overrides
    parser.add_argument("--workers", type=int, default=None,
                        help="Override DataLoader worker count (default: auto-detect)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size (default: auto-detect from VRAM)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP (debug only)")
    parser.add_argument("--vram", type=float, default=None,
                        help="Manually specify GPU VRAM in GB (override auto-detection)")
    args = parser.parse_args()

    _validate_model_path(args.model)

    # ── Hardware detection ──
    profile = HardwareProfile.detect()
    if args.vram is not None:
        profile.vram_gb = args.vram
        logger.info("VRAM manually set to %.1f GB", args.vram)

    # Generate timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(__file__).resolve().parents[2] / "output" / timestamp

    base = {"data": args.data, "device": args.device, "project": str(run_dir)}

    # Resolve pretrained weights
    pretrained = _resolve_pretrained(args, COCO_PRETRAINED)
    if args.pretrained and pretrained is None:
        sys.exit(1)

    # ── Stage execution (C1 → C2 → C3) ──
    stage_plan = [
        ("warmup",   not args.skip_warmup),
        ("full",     not args.skip_full),
        ("finetune", not args.skip_finetune),
    ]

    ckpt: Path = Path(pretrained) if pretrained else Path(args.model)
    for stage_key, enabled in stage_plan:
        if not enabled:
            logger.info("Skipping %s", _STAGE_CONFIGS[stage_key]["label"])
            continue
        ckpt = _run_stage(stage_key, base, profile, args, ckpt)

    logger.info("=" * 60)
    logger.info("Training complete. Final model: %s", ckpt)


if __name__ == "__main__":
    main()
