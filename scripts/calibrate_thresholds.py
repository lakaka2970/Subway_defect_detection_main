#!/usr/bin/env python3
"""
Per-class confidence threshold calibration (Stage 5 post-step).

Runs inference on a validation set, collects per-class (confidence, is_TP)
pairs, and searches for the optimal confidence threshold for each class
that maximises F2-score while respecting minimum Precision and Recall
constraints.

This addresses the core issue identified in Run A1: different defect types
have very different Precision/Recall trade-offs (e.g. SVHBNM Precision=0.168
at default threshold, but VHBNM is fine). Per-class thresholds let each class
operate at its optimal operating point.

Workflow:
  1. python scripts/collect_hard_negatives.py (collect FP crops)
  2. python scripts/calibrate_thresholds.py (per-class threshold search)
  3. Deploy with per-class thresholds in inference pipeline

Output::

    data/calibrated_thresholds/
    ├── thresholds.json       (recommended per-class thresholds)
    ├── pr_curves.json        (full PR data for plotting)
    └── calibration_report.txt (human-readable report)

Usage::

    python scripts/calibrate_thresholds.py \\
        --model weights/stage4_best_finetune.pt \\
        --data data/subway_crops/subway_crops.yaml

    python scripts/calibrate_thresholds.py \\
        --model weights/stage5_calibrated.pt \\
        --data data/subway_crops/subway_crops.yaml \\
        --target-precision 0.90 --target-recall 0.80
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Use Ultralytics' native box_iou — guaranteed consistent with model.val()
try:
    from ultralytics.utils.metrics import box_iou as _ultra_box_iou
    import torch
    HAS_ULTRA_IOU = True
except ImportError:
    HAS_ULTRA_IOU = False

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_OUTPUT = Path("data/calibrated_thresholds")
DEFAULT_CONF_START = 0.05
DEFAULT_CONF_END = 0.95
DEFAULT_CONF_STEP = 0.01

# Class names (must match training config)
CLASS_NAMES = [
    "VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
]
NC = len(CLASS_NAMES)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _box_iou(
    box1: Tuple[float, float, float, float],
    box2: Tuple[float, float, float, float],
) -> float:
    """Compute IoU between two YOLO-format boxes (xc, yc, w, h) in normalised coords."""
    def to_corners(b: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        xc, yc, w, h = b
        return (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2)

    x1a, y1a, x2a, y2a = to_corners(box1)
    x1b, y1b, x2b, y2b = to_corners(box2)
    inter_x1 = max(x1a, x1b)
    inter_y1 = max(y1a, y1b)
    inter_x2 = min(x2a, x2b)
    inter_y2 = min(y2a, y2b)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = (x2a - x1a) * (y2a - y1a)
    area_b = (x2b - x1b) * (y2b - y1b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _load_yolo_labels(label_dir: Path) -> Dict[str, List[Tuple[int, float, float, float, float]]]:
    """Load YOLO-format ground-truth labels keyed by image stem."""
    labels: Dict[str, List[Tuple[int, float, float, float, float]]] = {}
    for lbl_file in sorted(label_dir.glob("*.txt")):
        stem = lbl_file.stem
        entries: List[Tuple[int, float, float, float, float]] = []
        for line in lbl_file.read_text(encoding="utf-8").strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                cls_id = int(parts[0])
                vals = [float(p) for p in parts[1:5]]
                entries.append((cls_id, vals[0], vals[1], vals[2], vals[3]))
            except (ValueError, IndexError):
                continue
        if entries:
            labels[stem] = entries
    return labels


def _box_iou_pixel(
    box1: np.ndarray, box2: np.ndarray,
) -> float:
    """Compute IoU between two boxes in pixel (x1,y1,x2,y2) format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_b = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _f_beta_score(precision: float, recall: float, beta: float = 2.0) -> float:
    """Compute F-beta score. beta=2 weights Recall higher than Precision."""
    if precision <= 0 or recall <= 0:
        return 0.0
    beta_sq = beta ** 2
    return (1 + beta_sq) * (precision * recall) / (beta_sq * precision + recall)


# ── Main logic ───────────────────────────────────────────────────────────────

def calibrate_thresholds(
    model_weights: Path,
    data_yaml: Path,
    output_dir: Path = DEFAULT_OUTPUT,
    conf_start: float = DEFAULT_CONF_START,
    conf_end: float = DEFAULT_CONF_END,
    conf_step: float = DEFAULT_CONF_STEP,
    target_precision: float = 0.85,
    target_recall: float = 0.80,
    iou_threshold: float = 0.50,
    device: str = "0",
    imgsz: int = 1280,
) -> Dict:
    """Run inference and calibrate per-class confidence thresholds.

    Args:
        model_weights: Path to trained YOLO model.
        data_yaml: Path to dataset YAML config.
        output_dir: Output directory for calibration files.
        conf_start: Lower bound of threshold search range.
        conf_end: Upper bound of threshold search range.
        conf_step: Step size for threshold search.
        target_precision: Minimum acceptable Precision.
        target_recall: Minimum acceptable Recall.
        iou_threshold: IoU threshold for TP matching.
        device: CUDA device.
        imgsz: Inference image size.

    Returns:
        Calibration result dict.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics is required. Install with: pip install ultralytics")
        sys.exit(1)

    print("=" * 60)
    print("  Per-Class Threshold Calibrator")
    print("=" * 60)
    print(f"  Model          : {model_weights}")
    print(f"  Data           : {data_yaml}")
    print(f"  Target Precision: {target_precision}")
    print(f"  Target Recall   : {target_recall}")
    print(f"  IoU threshold   : {iou_threshold}")
    print(f"  Search range    : [{conf_start}, {conf_end}] step={conf_step}")
    print()

    # ── Load model ─────────────────────────────────────────────────────
    print("Loading model...")
    model = YOLO(str(model_weights))

    # ── Read data.yaml ─────────────────────────────────────────────────
    import yaml as _yaml
    with open(data_yaml, encoding="utf-8") as f:
        ds_cfg = _yaml.safe_load(f)

    ds_path = Path(ds_cfg.get("path", "."))
    val_img_rel = ds_cfg.get("val", "images/val")
    val_img_dir = ds_path / val_img_rel
    if not val_img_dir.is_dir():
        val_img_dir = data_yaml.parent / val_img_rel
    if not val_img_dir.is_dir():
        print(f"ERROR: Validation image directory not found: {val_img_dir}")
        sys.exit(1)

    val_lbl_dir = val_img_dir.parent / "labels"
    if not val_lbl_dir.is_dir():
        # Try alternate layout: dataset_root/labels/split_name/
        val_lbl_dir = val_img_dir.parent.parent / "labels" / val_img_dir.parent.name
    if not val_lbl_dir.is_dir():
        print(f"ERROR: Label directory not found")
        sys.exit(1)

    gt_labels = _load_yolo_labels(val_lbl_dir)
    print(f"  GT labels loaded: {len(gt_labels)} images")

    # ── Count GT boxes per class ───────────────────────────────────────
    gt_per_class: Dict[int, int] = defaultdict(int)
    for entries in gt_labels.values():
        for cls_id, *_ in entries:
            if cls_id < NC:
                gt_per_class[cls_id] += 1
    print(f"  GT boxes per class: {dict(gt_per_class)}")

    # ── Run inference and collect (conf, cls, is_TP) ───────────────────
    print(f"\nRunning inference on {val_img_dir}...")
    results = model.predict(
        source=str(val_img_dir),
        imgsz=imgsz,
        conf=conf_start,  # very low — we collect everything
        iou=0.50,
        device=device,
        verbose=False,
        stream=True,
    )

    # Per-class: list of (confidence, is_TP) tuples
    per_class_preds: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)
    total_dets = 0
    total_tp = 0
    stem_matched = 0
    stem_missed = 0

    for result in tqdm(results, desc="Collecting predictions", unit="img"):
        if result is None or result.boxes is None:
            continue

        stem = Path(result.path).stem
        gts = gt_labels.get(stem)
        if gts is None:
            # No GT labels for this image — all detections are FPs
            stem_missed += 1
        else:
            stem_matched += 1

        if gts is None:
            gts = []

        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        # Get image dimensions for pixel-space matching
        img_h, img_w = result.orig_shape

        # Convert GT boxes from normalised (xc,yc,w,h) → pixel (x1,y1,x2,y2)
        gt_xyxy_list = []
        gt_cls_list = []
        for gt_cls, gt_xc, gt_yc, gt_w, gt_h in gts:
            if gt_cls >= NC:
                continue
            x1 = max(0.0, (gt_xc - gt_w / 2) * img_w)
            y1 = max(0.0, (gt_yc - gt_h / 2) * img_h)
            x2 = min(img_w, (gt_xc + gt_w / 2) * img_w)
            y2 = min(img_h, (gt_yc + gt_h / 2) * img_h)
            gt_xyxy_list.append([x1, y1, x2, y2])
            gt_cls_list.append(gt_cls)

        if not gt_xyxy_list:
            # No valid GT for this image — all detections are FPs
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                if cls_id < NC:
                    per_class_preds[cls_id].append((conf, False))
                    total_dets += 1
            continue

        # Get prediction boxes in pixel xyxy format
        # boxes.xyxy is already in pixel coordinates of orig_shape
        pred_xyxy = boxes.xyxy.cpu().numpy()  # (N, 4) in pixel coords
        pred_cls = boxes.cls.cpu().numpy().astype(int)  # (N,)
        pred_conf = boxes.conf.cpu().numpy()  # (N,)

        # Build GT tensor for box_iou
        gt_tensor = np.array(gt_xyxy_list, dtype=np.float32)  # (M, 4)

        # Track which GTs have been matched (prevent double-counting)
        gt_matched = [False] * len(gt_xyxy_list)

        # Process detections sorted by confidence (high → low)
        det_order = np.argsort(-pred_conf)
        for idx in det_order:
            cls_id = int(pred_cls[idx])
            conf = float(pred_conf[idx])
            total_dets += 1

            if cls_id >= NC:
                continue

            is_tp = False
            best_iou = 0.0
            best_gt = -1

            # Find best unmatched GT of the same class
            for gt_idx, gt_cls in enumerate(gt_cls_list):
                if gt_matched[gt_idx]:
                    continue
                if gt_cls != cls_id:
                    continue

                # Compute IoU between this pred box and GT box (both in pixel xyxy)
                pred_box = pred_xyxy[idx:idx+1]  # (1, 4)
                gt_box = gt_tensor[gt_idx:gt_idx+1]  # (1, 4)

                if HAS_ULTRA_IOU:
                    iou = float(_ultra_box_iou(
                        torch.from_numpy(pred_box),
                        torch.from_numpy(gt_box),
                    )[0, 0])
                else:
                    iou = _box_iou_pixel(
                        pred_box[0], gt_box[0],
                    )

                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx

            if best_iou >= iou_threshold and best_gt >= 0:
                is_tp = True
                gt_matched[best_gt] = True
                total_tp += 1

            per_class_preds[cls_id].append((conf, is_tp))

    print(f"  Total detections collected: {total_dets}")
    print(f"  Total TPs: {total_tp}")
    print(f"  Images with GT: {stem_matched}, without GT: {stem_missed}")

    # ── Per-class threshold search ─────────────────────────────────────
    thresholds_range = np.arange(conf_start, conf_end + conf_step, conf_step)
    calibration: Dict[str, Dict] = {}
    pr_curve_data: Dict[str, Dict] = {}

    for cls_id in sorted(per_class_preds):
        if cls_id >= NC:
            continue
        cls_name = CLASS_NAMES[cls_id]
        preds = per_class_preds[cls_id]
        gt_count = gt_per_class.get(cls_id, 0)
        if gt_count == 0:
            print(f"\n  {cls_name}: No GT boxes — skipping")
            continue

        preds.sort(key=lambda x: x[0], reverse=True)  # sort by confidence desc

        best_threshold = 0.5  # fallback default
        best_f2 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        pr_points: List[Dict] = []

        for thresh in thresholds_range:
            tp = sum(1 for c, is_tp in preds if c >= thresh and is_tp)
            fp = sum(1 for c, is_tp in preds if c >= thresh and not is_tp)

            precision = tp / max(1, tp + fp)
            recall = tp / gt_count if gt_count > 0 else 0.0
            f2 = _f_beta_score(precision, recall, beta=2.0)

            pr_points.append({
                "threshold": round(float(thresh), 3),
                "tp": tp, "fp": fp,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f2": round(f2, 4),
            })

            # Select best threshold: satisfy both targets, then maximize F2
            if precision >= target_precision and recall >= target_recall and f2 > best_f2:
                best_f2 = f2
                best_threshold = float(thresh)
                best_precision = precision
                best_recall = recall

        # If no threshold satisfies both targets, pick the best F2 overall
        if best_f2 == 0.0:
            for pt in pr_points:
                if pt["f2"] > best_f2:
                    best_f2 = pt["f2"]
                    best_threshold = pt["threshold"]
                    best_precision = pt["precision"]
                    best_recall = pt["recall"]

        calibration[cls_name] = {
            "recommended_threshold": round(best_threshold, 3),
            "precision": round(best_precision, 4),
            "recall": round(best_recall, 4),
            "f2_score": round(best_f2, 4),
            "gt_count": gt_count,
            "detections_at_threshold": sum(1 for c, _ in preds if c >= best_threshold),
            "meets_targets": best_precision >= target_precision and best_recall >= target_recall,
        }
        pr_curve_data[cls_name] = {
            "gt_count": gt_count,
            "total_detections": len(preds),
            "points": pr_points,
        }

        status = "✓" if calibration[cls_name]["meets_targets"] else "✗"
        print(f"\n  {status} {cls_name}: threshold={best_threshold:.2f}  "
              f"P={best_precision:.3f}  R={best_recall:.3f}  F2={best_f2:.3f}")

    # ── Save outputs ───────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)

    thresholds_path = output_dir / "thresholds.json"
    thresholds_path.write_text(
        json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    pr_curves_path = output_dir / "pr_curves.json"
    pr_curve_data["_config"] = {
        "model": str(model_weights),
        "data": str(data_yaml),
        "target_precision": target_precision,
        "target_recall": target_recall,
        "iou_threshold": iou_threshold,
        "search_range": [conf_start, conf_end, conf_step],
    }
    pr_curves_path.write_text(
        json.dumps(pr_curve_data, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    # ── Human-readable report ──────────────────────────────────────────
    report_lines = [
        "=" * 60,
        "Per-Class Confidence Threshold Calibration Report",
        "=" * 60,
        f"Model: {model_weights}",
        f"Data:  {data_yaml}",
        f"Target: Precision ≥ {target_precision}, Recall ≥ {target_recall}",
        "",
        f"{'Class':<12s} {'Thresh':>8s} {'Prec':>8s} {'Rec':>8s} {'F2':>8s} {'GT':>6s} {'Status':>8s}",
        "-" * 66,
    ]
    for cls_name in CLASS_NAMES:
        if cls_name not in calibration:
            continue
        c = calibration[cls_name]
        status = "PASS" if c["meets_targets"] else "BELOW"
        report_lines.append(
            f"{cls_name:<12s} {c['recommended_threshold']:>8.3f} "
            f"{c['precision']:>8.4f} {c['recall']:>8.4f} {c['f2_score']:>8.4f} "
            f"{c['gt_count']:>6d} {status:>8s}"
        )
    report_lines.extend([
        "",
        "Usage in inference:",
        "  # Python",
        "  from subway_defect.pipeline.two_stage import TwoStagePipeline",
        f"  per_class_conf = {json.dumps({k: v['recommended_threshold'] for k, v in calibration.items()}, indent=4)}",
        "  pipeline = TwoStagePipeline(..., per_class_conf=per_class_conf)",
        "",
        f"Full data: {output_dir}/",
    ])

    report_path = output_dir / "calibration_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\n{'=' * 60}")
    print(f"  Calibration files saved to {output_dir}/")

    return calibration


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Per-class confidence threshold calibration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/calibrate_thresholds.py --model weights/stage4_best_finetune.pt
  python scripts/calibrate_thresholds.py --model weights/stage4_best_finetune.pt --target-precision 0.90
""",
    )
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to trained YOLO model (.pt)",
    )
    parser.add_argument(
        "--data", type=Path,
        default=Path("data/subway_crops/subway_crops.yaml"),
        help="Path to dataset YAML config",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--target-precision", type=float, default=0.85,
        help="Minimum acceptable Precision (default: 0.85)",
    )
    parser.add_argument(
        "--target-recall", type=float, default=0.80,
        help="Minimum acceptable Recall (default: 0.80)",
    )
    parser.add_argument(
        "--conf-start", type=float, default=DEFAULT_CONF_START,
        help=f"Lower bound of threshold search (default: {DEFAULT_CONF_START})",
    )
    parser.add_argument(
        "--conf-end", type=float, default=DEFAULT_CONF_END,
        help=f"Upper bound of threshold search (default: {DEFAULT_CONF_END})",
    )
    parser.add_argument(
        "--conf-step", type=float, default=DEFAULT_CONF_STEP,
        help=f"Step size for threshold search (default: {DEFAULT_CONF_STEP})",
    )
    parser.add_argument(
        "--iou-threshold", type=float, default=0.50,
        help="IoU threshold for TP matching (default: 0.50)",
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="CUDA device (default: 0)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=1280,
        help="Inference image size (default: 1280)",
    )
    args = parser.parse_args()

    calibrate_thresholds(
        model_weights=args.model,
        data_yaml=args.data,
        output_dir=args.output,
        conf_start=args.conf_start,
        conf_end=args.conf_end,
        conf_step=args.conf_step,
        target_precision=args.target_precision,
        target_recall=args.target_recall,
        iou_threshold=args.iou_threshold,
        device=args.device,
        imgsz=args.imgsz,
    )


if __name__ == "__main__":
    main()
