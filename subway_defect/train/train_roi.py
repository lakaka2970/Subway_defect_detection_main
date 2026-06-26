#!/usr/bin/env python3
"""Train the ROI proposer (Stage B) — YOLO11n on structural regions.

Auto-detects GPU VRAM / CPU cores / RAM and tunes batch size, DataLoader
workers, and cache strategy accordingly.

Usage:
    python -m subway_defect.train.train_roi --data datasets/roi/roi_data.yaml --device 0
    python -m subway_defect.train.train_roi --data data/Defect_dataset/defect_data.yaml --pretrained yolo11n.pt
"""

import argparse
import logging
import sys
from pathlib import Path

from subway_yolo import YOLO

from subway_defect.train.configs import (
    ROI_TRAIN_CONFIG,
    HardwareProfile,
    apply_hardware_profile,
)

logger = logging.getLogger(__name__)


def _check_gpu_memory(required_gb: float = 4.0) -> None:
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
        free, _ = torch.cuda.mem_get_info(0)
        free_gb = free / (1024 ** 3)
        if free_gb < required_gb:
            import warnings
            warnings.warn(
                f"GPU free memory is low ({free_gb:.1f} GB). "
                f"Training may OOM. Close other GPU processes or reduce --batch.",
                stacklevel=2,
            )
    except RuntimeError as e:
        logger.warning("GPU memory check failed (CUDA runtime error): %s", e)


def main():
    parser = argparse.ArgumentParser(description="Train ROI proposer (Stage B)")
    parser.add_argument("--data", default="datasets/roi/roi_data.yaml")
    parser.add_argument("--model", default="yolo11n.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="roi_proposer")
    parser.add_argument("--pretrained", default=None,
                        help="COCO pretrained weights (e.g. yolo11n.pt)")

    # Hardware overrides
    parser.add_argument("--workers", type=int, default=None,
                        help="Override DataLoader worker count (default: auto-detect)")
    parser.add_argument("--batch", type=int, default=None,
                        help="Override batch size (default: auto-detect from VRAM)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP (debug only)")
    args = parser.parse_args()

    # ── Hardware detection ──
    profile = HardwareProfile.detect()

    config = {**ROI_TRAIN_CONFIG}
    config = apply_hardware_profile(config, profile, args.model)
    config["data"] = args.data
    config["epochs"] = args.epochs
    config["device"] = args.device
    config["name"] = args.name

    if args.workers is not None:
        config["workers"] = args.workers
    if args.batch is not None:
        config["batch"] = args.batch
    if args.no_amp:
        config["amp"] = False

    # Verify pretrained file exists — check explicit path, then weights/
    if args.pretrained:
        pt_path = Path(args.pretrained)
        if not pt_path.exists():
            yolo_w = Path("weights") / pt_path.name
            if yolo_w.exists():
                args.pretrained = str(yolo_w)
                logger.info("Found pretrained weights: %s", args.pretrained)
            else:
                logger.error("Pretrained weights not found: %s", pt_path.name)
                logger.error("       Place the file in weights/ or use --pretrained <path>")
                sys.exit(1)

    _check_gpu_memory(required_gb=4.0)

    # If pretrained specified, load from weights; otherwise build from yaml
    model_file = args.pretrained or args.model
    model = YOLO(model_file)
    results = model.train(**config)
    logger.info("ROI training complete. Best model: %s", model.trainer.save_dir)


if __name__ == "__main__":
    main()
