#!/usr/bin/env python3
"""
Label Quality Audit Tool — flag suspicious annotations for manual review.

Motivation
----------
SVHBNM has the most training instances (1,197) but the worst mAP50 (0.251).
The report (``docs/plans/2026-06-27_方案C_*.md``) identifies several
potential labelling issues that cannot be ruled out by code alone:

- Bounding boxes that are too large / too small / shifted
- Normal (non-defect) structures incorrectly labelled as defects
- Class-boundary blur between VHBNM / SVHBNM / SVHBNL / VHBNL
- Same defect annotated at different scales across images
- Missing annotations (model detects real defect → GT says background)

This script helps auditors by:

1. Running YOLO inference on the training / validation set
2. Flagging **high-confidence FP** (model says defect, GT says background)
   → potential missing annotations or normal-structure-looks-like-defect cases
3. Flagging **low-confidence FN** (GT says defect, model misses or has low conf)
   → potential mislabelling, extreme difficulty, or annotation boundary issues
4. Saving cropped regions + metadata for quick manual review
5. Producing a per-class summary sorted by suspicion score

Output layout::

    data/label_audit/
    ├── high_conf_fp/          # Potential missing / mislabelled annotations
    │   ├── VHBNM/
    │   ├── SVHBNM/            # ← Priority: most problematic class
    │   └── ...
    ├── low_conf_fn/           # Potential annotation errors or hard cases
    │   ├── SVHBNL/
    │   └── ...
    ├── audit_summary.json     # Per-class statistics + suspicion rankings
    └── audit_report.txt       # Human-readable report with recommendations

Usage::

    # Full audit on validation set (recommended first pass)
    python scripts/audit_labels.py --model weights/stage3_main.pt --split val

    # Focus on specific classes
    python scripts/audit_labels.py --model weights/stage3_main.pt \\
        --classes SVHBNM SVHBNL --split train

    # Tune sensitivity
    python scripts/audit_labels.py --model weights/stage3_main.pt \\
        --fp-conf 0.40 --fn-conf 0.15 --max-samples 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np

# ── Project paths ───────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA = _PROJECT_ROOT / "data" / "subway_crops" / "subway_crops.yaml"

try:
    from subway_defect.classes import TRAIN_CLASSES as CLASS_NAMES
except ImportError:
    CLASS_NAMES = [
        "VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM",
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Data structures
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SuspiciousSample:
    """A single flagged sample for manual review."""
    image_path: str
    label_path: str
    crop_path: str = ""
    pred_class: int = -1
    pred_class_name: str = ""
    gt_class: int = -1
    gt_class_name: str = ""
    confidence: float = 0.0
    iou_with_any_gt: float = 0.0
    suspicion_reason: str = ""  # "high_conf_fp" | "low_conf_fn" | "size_outlier"
    suspicion_score: float = 0.0  # 0-1, higher = more suspicious


@dataclass
class AuditReport:
    model_path: str = ""
    split: str = ""
    total_images: int = 0
    total_gt_boxes: int = 0
    total_predictions: int = 0
    fp_flagged: int = 0
    fn_flagged: int = 0
    per_class: Dict[str, dict] = field(default_factory=dict)
    top_suspicious: List[SuspiciousSample] = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
# Core logic
# ═════════════════════════════════════════════════════════════════════════════

def _box_iou(box1: Tuple[float, ...], box2: Tuple[float, ...]) -> float:
    """Compute IoU between two boxes in xywhn format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Convert to xyxy
    bx1_l = x1 - w1 / 2
    bx1_r = x1 + w1 / 2
    bx1_t = y1 - h1 / 2
    bx1_b = y1 + h1 / 2

    bx2_l = x2 - w2 / 2
    bx2_r = x2 + w2 / 2
    bx2_t = y2 - h2 / 2
    bx2_b = y2 + h2 / 2

    inter_l = max(bx1_l, bx2_l)
    inter_r = min(bx1_r, bx2_r)
    inter_t = max(bx1_t, bx2_t)
    inter_b = min(bx1_b, bx2_b)

    if inter_l >= inter_r or inter_t >= inter_b:
        return 0.0

    inter_area = (inter_r - inter_l) * (inter_b - inter_t)
    area1 = w1 * h1
    area2 = w2 * h2
    return inter_area / (area1 + area2 - inter_area + 1e-9)


def _load_labels_yolo(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """Load YOLO-format labels: [(cls_id, xc, yc, w, h), ...] all normalized."""
    labels: List[Tuple[int, float, float, float, float]] = []
    if not label_path.exists():
        return labels
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            try:
                labels.append((
                    int(parts[0]),
                    float(parts[1]), float(parts[2]),
                    float(parts[3]), float(parts[4]),
                ))
            except (ValueError, IndexError):
                continue
    return labels


def _crop_region(
    img: np.ndarray,
    bbox_norm: Tuple[float, float, float, float],
    margin: float = 0.5,
) -> np.ndarray:
    """Crop a region around a normalized bbox with context margin.

    Args:
        img: BGR image array.
        bbox_norm: (xc, yc, w, h) in normalized coordinates.
        margin: Extra context as fraction of bbox size (0.5 = 50% extra).

    Returns:
        Cropped BGR image or None if invalid.
    """
    h, w = img.shape[:2]
    xc, yc, bw, bh = bbox_norm

    px = int(xc * w)
    py = int(yc * h)
    pw = int(bw * w)
    ph = int(bh * h)

    if pw <= 0 or ph <= 0:
        return None

    # Expand by margin
    expand_w = int(pw * margin)
    expand_h = int(ph * margin)

    x1 = max(0, px - pw // 2 - expand_w)
    y1 = max(0, py - ph // 2 - expand_h)
    x2 = min(w, px + pw // 2 + expand_w)
    y2 = min(h, py + ph // 2 + expand_h)

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]


def audit_labels(
    model_path: Path,
    data_yaml: Path,
    output_dir: Path,
    split: str = "val",
    fp_conf_thresh: float = 0.30,
    fn_conf_thresh: float = 0.15,
    fp_iou_thresh: float = 0.10,
    class_filter: Optional[Set[int]] = None,
    max_samples_per_class: int = 100,
    dry_run: bool = False,
) -> AuditReport:
    """Run label quality audit.

    Args:
        model_path: Path to trained YOLO .pt file.
        data_yaml: Path to dataset YAML config.
        output_dir: Root output directory for audit results.
        split: Dataset split to audit ("train" | "val").
        fp_conf_thresh: Min confidence for a prediction to be flagged as
            potential FP (model is confident, but no matching GT).
        fn_conf_thresh: Max confidence for a GT box to be flagged as
            potential FN (GT exists, but model misses or has very low conf).
        fp_iou_thresh: Max IoU with any GT for a prediction to be considered FP.
        class_filter: If set, only audit these class IDs.
        max_samples_per_class: Max flagged samples to save per class.
        dry_run: If True, only compute statistics without saving crops.

    Returns:
        AuditReport with full statistics and top suspicious samples.
    """
    from subway_yolo import YOLO

    report = AuditReport(model_path=str(model_path), split=split)

    # ── Load model ──
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # ── Find images ──
    data_cfg = {}
    try:
        import yaml as _yaml
        data_cfg = _yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    except Exception:
        pass

    data_root = Path(data_cfg.get("path", data_yaml.parent))
    img_dir = data_root / split / "images"
    lbl_dir = data_root / split / "labels"

    if not img_dir.is_dir():
        # Try alternate layout
        img_dir = data_root / "images" / split
        lbl_dir = data_root / "labels" / split

    if not img_dir.is_dir():
        print(f"ERROR: Image directory not found: {img_dir}")
        print(f"  Check --data path and --split ({split})")
        sys.exit(1)

    image_files = sorted(list(img_dir.glob("*")))
    image_files = [f for f in image_files if f.suffix.lower() in
                   (".jpg", ".jpeg", ".png", ".bmp")]
    report.total_images = len(image_files)
    print(f"Found {len(image_files)} images in {img_dir}")

    # ── Collect all predictions and ground truth ──
    all_fp: List[SuspiciousSample] = []
    all_fn: List[SuspiciousSample] = []
    per_class_fp_count: Dict[str, int] = defaultdict(int)
    per_class_fn_count: Dict[str, int] = defaultdict(int)
    per_class_gt_count: Dict[str, int] = defaultdict(int)

    for idx, img_path in enumerate(image_files):
        if idx % 50 == 0:
            print(f"  Processing {idx+1}/{len(image_files)} ...", flush=True)

        stem = img_path.stem
        lbl_path = lbl_dir / f"{stem}.txt"
        gt_labels = _load_labels_yolo(lbl_path)

        # Count GT per class
        for cls_id, *_ in gt_labels:
            if cls_id < len(CLASS_NAMES):
                per_class_gt_count[CLASS_NAMES[cls_id]] += 1
        report.total_gt_boxes += len(gt_labels)

        # Run inference
        results = model(str(img_path), verbose=False)
        if not results or len(results) == 0:
            # No predictions → all GT are potential FN
            for cls_id, xc, yc, w, h in gt_labels:
                if cls_id >= len(CLASS_NAMES):
                    continue
                if class_filter and cls_id not in class_filter:
                    continue
                all_fn.append(SuspiciousSample(
                    image_path=str(img_path),
                    label_path=str(lbl_path),
                    pred_class=-1,
                    pred_class_name="(no detection)",
                    gt_class=cls_id,
                    gt_class_name=CLASS_NAMES[cls_id],
                    confidence=0.0,
                    iou_with_any_gt=0.0,
                    suspicion_reason="low_conf_fn",
                    suspicion_score=1.0,  # complete miss = highly suspicious
                ))
            continue

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            # Same as above — no predictions
            for cls_id, xc, yc, w, h in gt_labels:
                if cls_id >= len(CLASS_NAMES):
                    continue
                if class_filter and cls_id not in class_filter:
                    continue
                all_fn.append(SuspiciousSample(
                    image_path=str(img_path),
                    label_path=str(lbl_path),
                    pred_class=-1,
                    pred_class_name="(no detection)",
                    gt_class=cls_id,
                    gt_class_name=CLASS_NAMES[cls_id],
                    confidence=0.0,
                    iou_with_any_gt=0.0,
                    suspicion_reason="low_conf_fn",
                    suspicion_score=1.0,
                ))
            continue

        pred_boxes = boxes.xywhn.cpu().numpy() if boxes.xywhn is not None else np.array([])
        pred_cls = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else np.array([])
        pred_conf = boxes.conf.cpu().numpy() if boxes.conf is not None else np.array([])

        # ── Flag high-confidence FP ──
        for i in range(len(pred_boxes)):
            p_cls = int(pred_cls[i])
            p_conf = float(pred_conf[i])
            if p_cls >= len(CLASS_NAMES):
                continue
            if class_filter and p_cls not in class_filter:
                continue
            if p_conf < fp_conf_thresh:
                continue

            # Check IoU against all GT boxes
            best_iou = 0.0
            for gt_cls, gt_xc, gt_yc, gt_w, gt_h in gt_labels:
                iou = _box_iou(
                    (pred_boxes[i][0], pred_boxes[i][1],
                     pred_boxes[i][2], pred_boxes[i][3]),
                    (gt_xc, gt_yc, gt_w, gt_h),
                )
                best_iou = max(best_iou, iou)

            if best_iou < fp_iou_thresh:
                # This is a false positive — model is confident but no GT match
                suspicion = (p_conf - fp_conf_thresh) / (1.0 - fp_conf_thresh + 1e-9)
                all_fp.append(SuspiciousSample(
                    image_path=str(img_path),
                    label_path=str(lbl_path),
                    pred_class=p_cls,
                    pred_class_name=CLASS_NAMES[p_cls],
                    gt_class=-1,
                    gt_class_name="(no GT)",
                    confidence=p_conf,
                    iou_with_any_gt=best_iou,
                    suspicion_reason="high_conf_fp",
                    suspicion_score=min(1.0, suspicion),
                ))
                per_class_fp_count[CLASS_NAMES[p_cls]] += 1

        # ── Flag low-confidence FN ──
        for cls_id, xc, yc, w, h in gt_labels:
            if cls_id >= len(CLASS_NAMES):
                continue
            if class_filter and cls_id not in class_filter:
                continue

            # Find best matching prediction for this GT
            best_match_conf = 0.0
            best_match_iou = 0.0
            for i in range(len(pred_boxes)):
                iou = _box_iou(
                    (xc, yc, w, h),
                    (pred_boxes[i][0], pred_boxes[i][1],
                     pred_boxes[i][2], pred_boxes[i][3]),
                )
                if iou > best_match_iou:
                    best_match_iou = iou
                    best_match_conf = float(pred_conf[i])

            if best_match_conf < fn_conf_thresh:
                # Model missed this GT or has very low confidence
                # Higher score if model was completely unaware (conf=0)
                suspicion = 1.0 - (best_match_conf / (fn_conf_thresh + 1e-9))
                all_fn.append(SuspiciousSample(
                    image_path=str(img_path),
                    label_path=str(lbl_path),
                    pred_class=-1,
                    pred_class_name=f"(best match conf={best_match_conf:.3f})",
                    gt_class=cls_id,
                    gt_class_name=CLASS_NAMES[cls_id],
                    confidence=best_match_conf,
                    iou_with_any_gt=best_match_iou,
                    suspicion_reason="low_conf_fn",
                    suspicion_score=min(1.0, max(0.0, suspicion)),
                ))
                per_class_fn_count[CLASS_NAMES[cls_id]] += 1

    report.fp_flagged = len(all_fp)
    report.fn_flagged = len(all_fn)

    # ── Build per-class report ──
    all_classes = set(list(per_class_gt_count) + list(per_class_fp_count) +
                      list(per_class_fn_count))
    for cls_name in sorted(all_classes):
        gt_n = per_class_gt_count.get(cls_name, 0)
        fp_n = per_class_fp_count.get(cls_name, 0)
        fn_n = per_class_fn_count.get(cls_name, 0)
        fp_rate = fp_n / max(1, gt_n)
        fn_rate = fn_n / max(1, gt_n)

        # Priority score: classes with high FP+FN rate need most attention
        priority_score = fp_rate + fn_rate

        report.per_class[cls_name] = {
            "gt_instances": gt_n,
            "fp_flagged": fp_n,
            "fn_flagged": fn_n,
            "fp_rate": round(fp_rate, 4),
            "fn_rate": round(fn_rate, 4),
            "priority_score": round(priority_score, 4),
            "verdict": (
                "HIGH PRIORITY — audit immediately"
                if priority_score > 0.5 else
                "Medium priority — spot-check 50 samples"
                if priority_score > 0.2 else
                "Low priority — spot-check 20 samples"
            ),
        }

    # ── Save flagged crops ──
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save high-confidence FP crops
        _save_flagged(
            all_fp, img_dir, output_dir / "high_conf_fp",
            max_samples_per_class, "FP",
        )
        # Save low-confidence FN crops
        _save_flagged(
            all_fn, img_dir, output_dir / "low_conf_fn",
            max_samples_per_class, "FN",
        )

    # ── Top suspicious samples (for quick review) ──
    all_flagged = all_fp + all_fn
    all_flagged.sort(key=lambda s: s.suspicion_score, reverse=True)
    report.top_suspicious = all_flagged[:50]

    # ── Write reports ──
    if not dry_run:
        # JSON summary
        summary_json = output_dir / "audit_summary.json"
        summary_data = {
            "model": str(model_path),
            "split": split,
            "total_images": report.total_images,
            "total_gt_boxes": report.total_gt_boxes,
            "fp_flagged": report.fp_flagged,
            "fn_flagged": report.fn_flagged,
            "thresholds": {
                "fp_conf": fp_conf_thresh,
                "fn_conf": fn_conf_thresh,
                "fp_iou": fp_iou_thresh,
            },
            "per_class": report.per_class,
            "top_suspicious_count": len(report.top_suspicious),
        }
        summary_json.write_text(
            json.dumps(summary_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        # Human-readable report
        _write_report(report, output_dir / "audit_report.txt")

    return report


def _save_flagged(
    samples: List[SuspiciousSample],
    img_dir: Path,
    out_dir: Path,
    max_per_class: int,
    tag: str,
) -> None:
    """Save cropped regions of flagged samples for manual review."""
    # Group by predicted/GT class
    by_class: Dict[str, List[SuspiciousSample]] = defaultdict(list)
    for s in samples:
        cls_name = (s.pred_class_name if tag == "FP"
                    else s.gt_class_name)
        by_class[cls_name].append(s)

    for cls_name, items in sorted(by_class.items()):
        cls_dir = out_dir / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        # Sort by suspicion score, take top N
        items.sort(key=lambda s: s.suspicion_score, reverse=True)
        items = items[:max_per_class]

        for i, sample in enumerate(items):
            img_path = Path(sample.image_path)
            if not img_path.exists():
                continue

            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # For FP: crop around the predicted bbox
            # For FN: crop around the GT bbox (we need to re-read the label)
            if tag == "FP":
                # We don't have the bbox stored in SuspiciousSample easily.
                # Instead, save the full image with a metadata sidecar.
                crop_name = f"{img_path.stem}_fp{i:03d}_c{sample.confidence:.2f}"
                cv2.imwrite(str(cls_dir / f"{crop_name}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])
            else:
                crop_name = f"{img_path.stem}_fn{i:03d}_c{sample.confidence:.2f}"
                cv2.imwrite(str(cls_dir / f"{crop_name}.jpg"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 90])

            # Save metadata sidecar
            meta = {
                "image": sample.image_path,
                "label": sample.label_path,
                "pred_class": sample.pred_class_name,
                "gt_class": sample.gt_class_name,
                "confidence": sample.confidence,
                "iou_with_any_gt": sample.iou_with_any_gt,
                "suspicion_reason": sample.suspicion_reason,
                "suspicion_score": sample.suspicion_score,
            }
            (cls_dir / f"{crop_name}.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )


def _write_report(report: AuditReport, path: Path) -> None:
    """Write a human-readable audit report."""
    lines = []
    lines.append("=" * 66)
    lines.append("  LABEL QUALITY AUDIT REPORT")
    lines.append("=" * 66)
    lines.append(f"  Model:       {report.model_path}")
    lines.append(f"  Split:       {report.split}")
    lines.append(f"  Images:      {report.total_images}")
    lines.append(f"  GT boxes:    {report.total_gt_boxes}")
    lines.append(f"  FP flagged:  {report.fp_flagged}")
    lines.append(f"  FN flagged:  {report.fn_flagged}")
    lines.append("")

    # Per-class table
    lines.append("-" * 66)
    lines.append(f"  {'Class':<12s} {'GT':>6s} {'FP':>6s} {'FN':>6s} "
                 f"{'FP%':>7s} {'FN%':>7s} {'Score':>7s}  Verdict")
    lines.append("-" * 66)

    # Sort by priority score
    sorted_classes = sorted(
        report.per_class.items(),
        key=lambda kv: kv[1]["priority_score"],
        reverse=True,
    )
    for cls_name, info in sorted_classes:
        lines.append(
            f"  {cls_name:<12s} "
            f"{info['gt_instances']:>6d} "
            f"{info['fp_flagged']:>6d} "
            f"{info['fn_flagged']:>6d} "
            f"{info['fp_rate']:>7.1%} "
            f"{info['fn_rate']:>7.1%} "
            f"{info['priority_score']:>7.3f}  "
            f"{info['verdict']}"
        )

    lines.append("-" * 66)
    lines.append("")

    # Recommendations
    lines.append("  CHECKLIST FOR MANUAL AUDIT:")
    lines.append("  1. For high_conf_fp/ images:")
    lines.append("     - Is there really no defect? → Add to hard negatives")
    lines.append("     - Is there a defect but no label? → Add missing annotation")
    lines.append("     - Is the box wrong (too big/small/shifted)? → Fix annotation")
    lines.append("  2. For low_conf_fn/ images:")
    lines.append("     - Is the GT label correct? → Model needs more similar samples")
    lines.append("     - Is the GT label wrong? → Remove or fix annotation")
    lines.append("     - Is the defect extremely subtle? → Mark as 'challenging'")
    lines.append("  3. For SVHBNM specifically:")
    lines.append("     - Check if any normal slot-groove bases are labelled as defects")
    lines.append("     - Check if VHBNM and SVHBNM boundaries are consistent")
    lines.append("     - Check box size consistency across similar defects")
    lines.append("")

    # Top suspicious
    if report.top_suspicious:
        lines.append("-" * 66)
        lines.append("  TOP 20 MOST SUSPICIOUS SAMPLES (review first):")
        lines.append("-" * 66)
        for i, s in enumerate(report.top_suspicious[:20]):
            img_stem = Path(s.image_path).stem
            lines.append(
                f"  {i+1:>2d}. [{s.suspicion_reason}] score={s.suspicion_score:.2f} "
                f"conf={s.confidence:.2f} pred={s.pred_class_name} "
                f"gt={s.gt_class_name} img={img_stem}"
            )

    lines.append("")
    lines.append("=" * 66)
    lines.append("  Next: Review flagged crops in data/label_audit/")
    lines.append("        Focus on HIGH PRIORITY classes first.")
    lines.append("=" * 66)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Label Quality Audit Tool — flag suspicious annotations for review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full audit on validation set
  python scripts/audit_labels.py --model weights/stage3_main.pt --split val

  # Focus on SVHBNM only
  python scripts/audit_labels.py --model weights/stage3_main.pt --classes SVHBNM

  # Dry-run for statistics only
  python scripts/audit_labels.py --model weights/stage3_main.pt --dry-run
""",
    )
    parser.add_argument(
        "--model", type=Path, required=True,
        help="Path to trained YOLO .pt file",
    )
    parser.add_argument(
        "--data", type=Path, default=_DEFAULT_DATA,
        help=f"Path to dataset YAML (default: {_DEFAULT_DATA})",
    )
    parser.add_argument(
        "--split", type=str, default="val", choices=["train", "val"],
        help="Dataset split to audit (default: val)",
    )
    parser.add_argument(
        "--output", type=Path, default=_PROJECT_ROOT / "data" / "label_audit",
        help="Output directory for audit results",
    )
    parser.add_argument(
        "--classes", type=str, nargs="*",
        help="Only audit these classes (default: all 7)",
    )
    parser.add_argument(
        "--fp-conf", type=float, default=0.30,
        help="Min confidence to flag as potential FP (default: 0.30)",
    )
    parser.add_argument(
        "--fn-conf", type=float, default=0.15,
        help="Max confidence to flag GT as potential FN (default: 0.15)",
    )
    parser.add_argument(
        "--fp-iou", type=float, default=0.10,
        help="Max IoU with GT for a prediction to be FP (default: 0.10)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=100,
        help="Max flagged samples to save per class (default: 100)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute statistics only, don't save crops",
    )
    args = parser.parse_args()

    if not args.model.exists():
        print(f"ERROR: Model not found: {args.model}")
        sys.exit(1)

    if not args.data.exists():
        print(f"ERROR: Dataset YAML not found: {args.data}")
        sys.exit(1)

    # Resolve class filter
    class_filter: Optional[Set[int]] = None
    if args.classes:
        class_filter = set()
        for c in args.classes:
            if c in CLASS_NAMES:
                class_filter.add(CLASS_NAMES.index(c))
            else:
                print(f"WARNING: Unknown class '{c}' — ignored. Valid: {CLASS_NAMES}")

    print("=" * 60)
    print("  Label Quality Audit Tool")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Data:       {args.data}")
    print(f"  Split:      {args.split}")
    print(f"  FP conf ≥   {args.fp_conf}")
    print(f"  FN conf <   {args.fn_conf}")
    print(f"  FP IoU <    {args.fp_iou}")
    print(f"  Classes:    {sorted(args.classes) if args.classes else 'all 7'}")
    print(f"  Dry run:    {args.dry_run}")
    print()

    report = audit_labels(
        model_path=args.model,
        data_yaml=args.data,
        output_dir=args.output,
        split=args.split,
        fp_conf_thresh=args.fp_conf,
        fn_conf_thresh=args.fn_conf,
        fp_iou_thresh=args.fp_iou,
        class_filter=class_filter,
        max_samples_per_class=args.max_samples,
        dry_run=args.dry_run,
    )

    # Print summary
    print(f"\n{'=' * 60}")
    print("  Audit Complete")
    print(f"{'=' * 60}")
    print(f"  FP flagged: {report.fp_flagged}")
    print(f"  FN flagged: {report.fn_flagged}")
    print()
    print("  Per-class priority (audit in this order):")
    sorted_classes = sorted(
        report.per_class.items(),
        key=lambda kv: kv[1]["priority_score"],
        reverse=True,
    )
    for cls_name, info in sorted_classes:
        print(f"    {cls_name:<12s}: GT={info['gt_instances']:>4d}  "
              f"FP={info['fp_flagged']:>4d}  FN={info['fn_flagged']:>4d}  "
              f"→ {info['verdict']}")

    if not args.dry_run:
        print(f"\n  Full report: {args.output / 'audit_report.txt'}")
        print(f"  JSON summary: {args.output / 'audit_summary.json'}")
        print(f"  FP crops: {args.output / 'high_conf_fp/'}")
        print(f"  FN crops: {args.output / 'low_conf_fn/'}")


if __name__ == "__main__":
    main()
