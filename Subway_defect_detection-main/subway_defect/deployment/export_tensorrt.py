#!/usr/bin/env python3
"""
Export YOLO models to TensorRT for GPU-accelerated inference.

Supports FP16 (default) and INT8 (with calibration) precision modes.
FP16 is recommended for vehicle-side deployment (10s constraint).

Usage:
    python -m subway_defect.deployment.export_tensorrt --model runs/train/weights/best.pt --fp16

    # INT8 export with calibration
    python -m subway_defect.deployment.export_tensorrt --model runs/train/weights/best.pt --int8 \\
        --calibration_data datasets/calibration/
"""

import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def export_fp16(model_path: str, output_path: str = None,
                imgsz: int = 1024, workspace: int = 4):
    """Export model to TensorRT FP16 engine.

    Args:
        model_path: Path to .pt weights.
        output_path: Output .engine path. Default: same name, .engine ext.
        imgsz: Input image size for static shape.
        workspace: GPU workspace size in GB.
    """
    model = YOLO(model_path)
    if output_path is None:
        output_path = str(Path(model_path).with_suffix(".engine"))
    model.export(
        format="engine",
        imgsz=imgsz,
        half=True,           # FP16
        workspace=workspace,
        device="0",
    )
    print(f"FP16 engine saved to {output_path}")
    return output_path


def export_int8(model_path: str, calibration_data: str,
                output_path: str = None, imgsz: int = 1024,
                workspace: int = 4):
    """Export model to TensorRT INT8 engine with calibration.

    Args:
        model_path: Path to .pt weights.
        calibration_data: Directory of calibration images (200-500).
        output_path: Output .engine path.
        imgsz: Input image size for static shape.
        workspace: GPU workspace size in GB.
    """
    model = YOLO(model_path)
    if output_path is None:
        stem = Path(model_path).stem
        output_path = str(Path(model_path).with_name(f"{stem}_int8.engine"))
    model.export(
        format="engine",
        imgsz=imgsz,
        int8=True,
        data=calibration_data,
        workspace=workspace,
        device="0",
    )
    print(f"INT8 engine saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export YOLO to TensorRT")
    parser.add_argument("--model", required=True, help="Path to .pt weights")
    parser.add_argument("--fp16", action="store_true", help="Export FP16 engine")
    parser.add_argument("--int8", action="store_true", help="Export INT8 engine")
    parser.add_argument("--calibration_data", default="datasets/calibration/")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--workspace", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.fp16:
        export_fp16(args.model, args.output, args.imgsz, args.workspace)
    elif args.int8:
        export_int8(args.model, args.calibration_data, args.output,
                    args.imgsz, args.workspace)
    else:
        print("Specify --fp16 or --int8")
        sys.exit(1)


if __name__ == "__main__":
    main()
