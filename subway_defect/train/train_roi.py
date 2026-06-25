#!/usr/bin/env python3
"""Train the ROI proposer (Stage B) — YOLO11n on structural regions.

Auto-detects GPU VRAM / CPU cores / RAM and tunes batch size, DataLoader
workers, and cache strategy accordingly.

Usage:
    python -m subway_defect.train.train_roi --data datasets/roi/roi_data.yaml --device 0
    python -m subway_defect.train.train_roi --data data/Defect_dataset/defect_data.yaml --pretrained yolo11n.pt
"""

import argparse
from pathlib import Path

from subway_yolo import YOLO

from subway_defect.train.configs import (
    ROI_TRAIN_CONFIG,
    HardwareProfile,
    apply_hardware_profile,
)


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

    # Verify pretrained file exists
    if args.pretrained and not Path(args.pretrained).exists():
        print(f"ERROR: Pretrained weights not found: {args.pretrained}")
        print(f"       Place the file in the project root or use --pretrained <path>")
        return

    # If pretrained specified, load from weights; otherwise build from yaml
    model_file = args.pretrained or args.model
    model = YOLO(model_file)
    results = model.train(**config)
    print(f"ROI training complete. Best model: {model.trainer.save_dir}")


if __name__ == "__main__":
    main()
