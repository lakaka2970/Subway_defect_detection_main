#!/usr/bin/env python3
"""Train the ROI proposer (Stage B) — YOLO11n on structural regions.

Usage:
    python train/train_roi.py --data datasets/roi/roi_data.yaml --device 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

from train.configs import ROI_TRAIN_CONFIG


def main():
    parser = argparse.ArgumentParser(description="Train ROI proposer")
    parser.add_argument("--data", default="datasets/roi/roi_data.yaml")
    parser.add_argument("--model", default="yolo11n.yaml")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="roi_proposer")
    parser.add_argument("--pretrained", default=None,
                        help="COCO pretrained weights (e.g. yolo11n.pt)")
    args = parser.parse_args()

    config = {
        **ROI_TRAIN_CONFIG,
        "data": args.data,
        "epochs": args.epochs,
        "device": args.device,
        "name": args.name,
    }

    # If pretrained specified, load from weights; otherwise build from yaml
    model_file = args.pretrained or args.model
    model = YOLO(model_file)
    results = model.train(**config)
    print(f"ROI training complete. Best model: {results.save_dir}")


if __name__ == "__main__":
    main()
