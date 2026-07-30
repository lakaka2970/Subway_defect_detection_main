#!/usr/bin/env python3
"""Comprehensive model analysis: per-class AP + TTA evaluation.

Usage:
  python scripts/analyze_model_performance.py --model output/.../stage_4/weights/best.pt --data data/subway_crops/subway_crops.yaml
  python scripts/analyze_model_performance.py --model output/.../stage_5/weights/best.pt --data data/subway_crops/subway_crops.yaml --tta
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subway_yolo import YOLO


def run_val(model_path: str, data_yaml: str, imgsz: int = 1280,  device: str = "0",
            split: str = "val", tta: bool = False) -> dict:
    """Run validation and extract per-class metrics."""
    model = YOLO(model_path)
    t0 = time.time()

    kwargs = dict(
        data=data_yaml,
        imgsz=imgsz,
        device=device,
        split=split,
        plots=False,
        save_json=False,
        verbose=True,
    )

    if tta:
        kwargs["augment"] = True  # Ultralytics TTA: fliplr + scale augmentation

    results = model.val(**kwargs)

    elapsed = time.time() - t0

    # Extract per-class AP
    per_class = {}
    if hasattr(results, 'ap_class_index') and results.ap_class_index is not None:
        names = results.names if hasattr(results, 'names') else {}
        for i, cls_id in enumerate(results.ap_class_index):
            cls_name = names.get(int(cls_id), f"class_{cls_id}")
            ap50_val = results.ap50[i] if hasattr(results, 'ap50') and i < len(results.ap50) else 0.0
            ap_val = results.ap[i] if hasattr(results, 'ap') and i < len(results.ap) else 0.0
            per_class[cls_name] = {
                "ap50": round(float(ap50_val), 4),
                "ap50_95": round(float(ap_val), 4),
            }

    summary = {
        "model": model_path,
        "data": data_yaml,
        "split": split,
        "imgsz": imgsz,
        "tta": tta,
        "elapsed_seconds": round(elapsed, 1),
        "mAP50": round(float(results.box.map50), 4),
        "mAP50_95": round(float(results.box.map), 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
        "fitness": round(float(results.fitness), 4) if hasattr(results, 'fitness') else None,
        "per_class": per_class,
    }

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze YOLO model performance")
    parser.add_argument("--model", required=True, type=str, help="Path to model weights")
    parser.add_argument("--data", required=True, type=str, help="Path to data YAML")
    parser.add_argument("--imgsz", default=1280, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--split", default="val", help="Dataset split: train/val/test")
    parser.add_argument("--tta", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--output", default=None, type=str, help="Output JSON path")
    args = parser.parse_args()

    summary = run_val(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        device=args.device,
        split=args.split,
        tta=args.tta,
    )

    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        model_name = Path(args.model).stem
        tag = "tta" if args.tta else "standard"
        out_path = Path(args.model).parents[2] / f"analysis_{model_name}_{tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print summary
    print()
    print("=" * 60)
    print(f"  Model: {args.model}")
    print(f"  TTA: {args.tta}")
    print(f"  mAP50: {summary['mAP50']:.4f}  |  mAP50-95: {summary['mAP50_95']:.4f}")
    print(f"  Precision: {summary['precision']:.4f}  |  Recall: {summary['recall']:.4f}")
    print(f"  Time: {summary['elapsed_seconds']:.1f}s")
    print("=" * 60)
    print()
    print("Per-Class AP:")
    print(f"{'Class':<12s} {'AP50':>8s} {'AP50-95':>8s}")
    print("-" * 30)
    for cls_name, vals in sorted(summary["per_class"].items(),
                                  key=lambda x: x[1]["ap50"]):
        marker = " ⚠ LOW" if vals["ap50"] < 0.30 else ""
        print(f"{cls_name:<12s} {vals['ap50']:>8.4f} {vals['ap50_95']:>8.4f}{marker}")

    print(f"\n  Output: {out_path}")


if __name__ == "__main__":
    main()
