#!/usr/bin/env python3
"""Train the defect detection model — Stage C (3 sub-stages).

C1 — Head warmup (frozen backbone, 50 epochs)
C2 — Full training (heavy augmentation, 200 epochs)
C3 — Fine-tune (mild augmentation, 50 epochs)

Usage:
    python train/train_defect.py --data datasets/defects/defect_data.yaml --device 0
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

from train.configs import (
    COCO_PRETRAINED,
    DEFECT_FINETUNE_CONFIG,
    DEFECT_FULL_TRAIN_CONFIG,
    DEFECT_WARMUP_CONFIG,
)


def main():
    parser = argparse.ArgumentParser(description="Train defect detector")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="models/yolo11s-EMA-SimAM.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="defect_detector")
    parser.add_argument("--pretrained", default=None,
                        help="Path to .pt weights (e.g. yolo11s.pt for COCO pretrain)")
    parser.add_argument("--coco_pretrain", action="store_true",
                        help="Auto-load matching COCO pretrained weights")
    parser.add_argument("--skip_warmup", action="store_true",
                        help="Skip C1 warmup (use when loading pretrained weights)")
    parser.add_argument("--skip_finetune", action="store_true",
                        help="Skip C3 fine-tune stage")
    args = parser.parse_args()

    base = {"data": args.data, "device": args.device}

    # Resolve pretrained weights
    pretrained = args.pretrained
    if args.coco_pretrain and not pretrained:
        # Auto-detect from model name
        for key, weight in COCO_PRETRAINED.items():
            if key in str(args.model):
                pretrained = weight
                print(f"Auto COCO pretrain: {weight}")
                break
        if not pretrained:
            pretrained = "yolo11s.pt"
            print(f"Default COCO pretrain: {pretrained}")

    # ── C1: Warmup ──
    if not args.skip_warmup:
        print("=" * 60)
        print("Stage C1: Head Warmup (50 epochs, frozen backbone)")
        print("=" * 60)
        c1 = {**DEFECT_WARMUP_CONFIG, **base}
        model_file = pretrained or args.model
        model = YOLO(model_file)
        model.train(name=f"{args.name}_c1_warmup", **c1)
        ckpt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    else:
        print("Skipping C1 warmup")
        ckpt = Path(pretrained) if pretrained else Path(args.model)

    # ── C2: Full Training ──
    print("=" * 60)
    print("Stage C2: Full Training (200 epochs, heavy augmentation)")
    print("=" * 60)
    c2 = {**DEFECT_FULL_TRAIN_CONFIG, **base}
    model2 = YOLO(str(ckpt))
    model2.train(name=f"{args.name}_c2_full", **c2)
    ckpt2 = Path(model2.trainer.save_dir) / "weights" / "best.pt"

    # ── C3: Fine-Tune ──
    if not args.skip_finetune:
        print("=" * 60)
        print("Stage C3: Fine-Tune (50 epochs, mild augmentation)")
        print("=" * 60)
        c3 = {**DEFECT_FINETUNE_CONFIG, **base}
        model3 = YOLO(str(ckpt2))
        model3.train(name=f"{args.name}_c3_finetune", **c3)
        final = Path(model3.trainer.save_dir) / "weights" / "best.pt"
    else:
        final = ckpt2

    print("=" * 60)
    print(f"Training complete. Final model: {final}")


if __name__ == "__main__":
    main()
