#!/usr/bin/env python3
"""Train the defect detection model — Stage C (3 sub-stages).

C1 — Head warmup (frozen backbone, 50 epochs)
C2 — Full training (heavy augmentation, 200 epochs)
C3 — Fine-tune (mild augmentation, 50 epochs)

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
import warnings
from datetime import datetime
from pathlib import Path

from subway_yolo import YOLO

from subway_defect.train.configs import (
    COCO_PRETRAINED,
    DEFECT_FINETUNE_CONFIG,
    DEFECT_FULL_TRAIN_CONFIG,
    DEFECT_WARMUP_CONFIG,
    HardwareProfile,
    apply_hardware_profile,
)


def _check_gpu_memory(required_gb: float = 6.0) -> None:
    """Warn if GPU free memory is below *required_gb* before training starts."""
    try:
        import torch
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
    except Exception:
        pass


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
    args = parser.parse_args()

    # ── Hardware detection ──
    profile = HardwareProfile.detect()

    # Generate timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use absolute path so output goes to <project_root>/output/ instead of runs/
    output_root = Path(__file__).resolve().parents[2] / "output"

    base = {
        "data": args.data,
        "device": args.device,
        "project": str(output_root),
    }

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

    # Verify pretrained file exists (avoid silent hang on download)
    if pretrained and not Path(pretrained).exists():
        print(f"ERROR: Pretrained weights not found: {pretrained}")
        print(f"       Place the file in the project root or use --pretrained <path>")
        return

    # -- C1: Warmup --
    if not args.skip_warmup:
        print("=" * 60)
        print("Stage C1: Head Warmup (50 epochs, frozen backbone)")
        print("=" * 60)
        c1 = {**DEFECT_WARMUP_CONFIG, **base}
        c1 = apply_hardware_profile(c1, profile, args.model)
        if args.workers is not None:
            c1["workers"] = args.workers
        if args.batch is not None:
            c1["batch"] = args.batch
        if args.no_amp:
            c1["amp"] = False

        _check_gpu_memory(required_gb=4.0)

        model_file = pretrained or args.model
        model = YOLO(model_file)
        model.train(name=f"{args.name}_{timestamp}_c1_warmup", **c1)
        ckpt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    else:
        print("Skipping C1 warmup")
        ckpt = Path(pretrained) if pretrained else Path(args.model)

    # -- C2: Full Training --
    if not args.skip_full:
        print("=" * 60)
        print("Stage C2: Full Training (200 epochs, heavy augmentation)")
        print("=" * 60)
        c2 = {**DEFECT_FULL_TRAIN_CONFIG, **base}
        c2 = apply_hardware_profile(c2, profile, args.model)
        if args.workers is not None:
            c2["workers"] = args.workers
        if args.batch is not None:
            c2["batch"] = args.batch
        if args.no_amp:
            c2["amp"] = False

        _check_gpu_memory(required_gb=8.0)

        model2 = YOLO(str(ckpt))
        model2.train(name=f"{args.name}_{timestamp}_c2_full", **c2)
        ckpt2 = Path(model2.trainer.save_dir) / "weights" / "best.pt"
    else:
        print("Skipping C2 full training")
        ckpt2 = ckpt

    # -- C3: Fine-Tune --
    if not args.skip_finetune:
        print("=" * 60)
        print("Stage C3: Fine-Tune (50 epochs, mild augmentation)")
        print("=" * 60)
        c3 = {**DEFECT_FINETUNE_CONFIG, **base}
        # Inherit workers/cache from C2, but use finetune"s own smaller batch
        c3["workers"] = c2.get("workers", profile.recommended_workers)
        c3["cache"] = c2.get("cache", profile.recommended_cache)
        if args.workers is not None:
            c3["workers"] = args.workers
        if args.batch is not None:
            c3["batch"] = args.batch
        if args.no_amp:
            c3["amp"] = False

        _check_gpu_memory(required_gb=6.0)

        model3 = YOLO(str(ckpt2))
        model3.train(name=f"{args.name}_{timestamp}_c3_finetune", **c3)
        final = Path(model3.trainer.save_dir) / "weights" / "best.pt"
    else:
        final = ckpt2

    print("=" * 60)
    print(f"Training complete. Final model: {final}")


if __name__ == "__main__":
    main()
