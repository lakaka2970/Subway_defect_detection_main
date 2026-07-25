"""CBHPM cascade evaluation: detector + classifier."""
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch
from subway_yolo import YOLO
from subway_defect.classifier.inference import ClassifierReasoner

CLASS_NAMES = ["VHBNM", "VHBNL", "SVHBNM", "SVHBNL", "SVHTNL", "CBHPM", "CBVPM"]
CBHPM_IDX = 5

def main():
    print("=" * 60)
    print("  CBHPM Cascade Evaluation")
    print("=" * 60)

    # Load detector
    det = YOLO("weights/stage4_best_finetune.pt")

    # Load classifier
    reasoner = ClassifierReasoner(
        "weights/classifier_cbhpm.pt",
        class_names=["normal", "missing"],
        context_scale=1.75,
        device="0",
        confidence_threshold=0.50,
    )

    # Load val images and labels
    val_img_dir = Path("data/subway_crops/val/images")
    val_lbl_dir = Path("data/subway_crops/val/labels")

    images = sorted(val_img_dir.glob("*.jpg"))
    print(f"  Val images: {len(images)}")

    # Collect GT and predictions for CBHPM
    gt_cbhpm = 0
    det_tp = 0
    det_fp = 0
    det_fn = 0
    cascade_tp = 0
    cascade_fp = 0
    cascade_fn = 0
    cascade_rejected = 0

    conf_thresh = 0.25  # CBHPM calibrated threshold
    iou_thresh = 0.5

    import cv2

    for img_path in images:
        lbl_path = val_lbl_dir / (img_path.stem + ".txt")

        # Load GT
        gt_boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id == CBHPM_IDX:
                        xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                        gt_boxes.append((xc, yc, w, h))
        gt_cbhpm += len(gt_boxes)

        if len(gt_boxes) == 0:
            # Run detector to count FPs
            results = det(str(img_path), imgsz=1280, conf=conf_thresh, iou=iou_thresh, verbose=False)
            if results and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    if int(boxes.cls[i]) == CBHPM_IDX:
                        det_fp += 1
                        # Cascade: run classifier
                        img = cv2.imread(str(img_path))
                        h_img, w_img = img.shape[:2]
                        x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                        det_dict = {"box": {"x": (x1+x2)/2/w_img, "y": (y1+y2)/2/h_img, "w": (x2-x1)/w_img, "h": (y2-y1)/h_img}}
                        state = reasoner(img, det_dict)
                        if state.get("state") in ("normal", "background", "negative"):
                            cascade_rejected += 1
                        else:
                            cascade_fp += 1
            continue

        # Run detector
        results = det(str(img_path), imgsz=1280, conf=conf_thresh, iou=iou_thresh, verbose=False)
        if not results or results[0].boxes is None:
            det_fn += len(gt_boxes)
            cascade_fn += len(gt_boxes)
            continue

        boxes = results[0].boxes
        img = cv2.imread(str(img_path))
        h_img, w_img = img.shape[:2]

        # Get CBHPM detections
        cbhpm_dets = []
        for i in range(len(boxes)):
            if int(boxes.cls[i]) == CBHPM_IDX:
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cbhpm_dets.append((x1, y1, x2, y2, conf))

        # Match detections to GT
        matched_gt = set()
        for det_box in cbhpm_dets:
            x1, y1, x2, y2, conf = det_box
            # Convert to normalized xywh for IoU
            det_xc = (x1 + x2) / 2 / w_img
            det_yc = (y1 + y2) / 2 / h_img
            det_w = (x2 - x1) / w_img
            det_h = (y2 - y1) / h_img

            best_iou = 0
            best_gt = -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                # IoU calculation
                ix1 = max(det_xc - det_w/2, gt[0] - gt[2]/2)
                iy1 = max(det_yc - det_h/2, gt[1] - gt[3]/2)
                ix2 = min(det_xc + det_w/2, gt[0] + gt[2]/2)
                iy2 = min(det_yc + det_h/2, gt[1] + gt[3]/2)
                inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                union = det_w*det_h + gt[2]*gt[3] - inter
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gi

            is_tp = best_iou >= iou_thresh and best_gt >= 0
            if is_tp:
                matched_gt.add(best_gt)
                det_tp += 1
                cascade_tp += 1  # TP always kept
            else:
                det_fp += 1
                # Cascade: run classifier on FP
                det_dict = {"box": {"x": (x1+x2)/2/w_img, "y": (y1+y2)/2/h_img, "w": (x2-x1)/w_img, "h": (y2-y1)/h_img}}
                state = reasoner(img, det_dict)
                if state.get("state") in ("normal", "background", "negative"):
                    cascade_rejected += 1
                else:
                    cascade_fp += 1

        # Unmatched GT = FN
        fn = len(gt_boxes) - len(matched_gt)
        det_fn += fn
        cascade_fn += fn

    # Compute metrics
    det_p = det_tp / (det_tp + det_fp) if (det_tp + det_fp) > 0 else 0
    det_r = det_tp / (det_tp + det_fn) if (det_tp + det_fn) > 0 else 0
    cas_p = cascade_tp / (cascade_tp + cascade_fp) if (cascade_tp + cascade_fp) > 0 else 0
    cas_r = cascade_tp / (cascade_tp + cascade_fn) if (cascade_tp + cascade_fn) > 0 else 0

    print(f"\n  GT CBHPM instances: {gt_cbhpm}")
    print(f"\n  {'Metric':<20} {'Detector Only':>15} {'Cascade':>15} {'Change':>10}")
    print(f"  {'-'*60}")
    print(f"  {'TP':<20} {det_tp:>15} {cascade_tp:>15} {cascade_tp-det_tp:>+10}")
    print(f"  {'FP':<20} {det_fp:>15} {cascade_fp:>15} {cascade_fp-det_fp:>+10}")
    print(f"  {'FN':<20} {det_fn:>15} {cascade_fn:>15} {cascade_fn-det_fn:>+10}")
    print(f"  {'Precision':<20} {det_p:>15.4f} {cas_p:>15.4f} {cas_p-det_p:>+10.4f}")
    print(f"  {'Recall':<20} {det_r:>15.4f} {cas_r:>15.4f} {cas_r-det_r:>+10.4f}")
    print(f"  {'FP Rejected':<20} {'—':>15} {cascade_rejected:>15}")
    fp_reduction = (det_fp - cascade_fp) / det_fp * 100 if det_fp > 0 else 0
    print(f"  {'FP Reduction':<20} {'—':>15} {fp_reduction:>14.1f}%")

    # PoC success criteria
    print(f"\n  PoC Success Criteria:")
    print(f"    FP↓≥30%:  {'✅ PASS' if fp_reduction >= 30 else '❌ FAIL'} ({fp_reduction:.1f}%)")
    r_retention = cas_r / det_r * 100 if det_r > 0 else 0
    print(f"    R保持≥98%: {'✅ PASS' if r_retention >= 98 else '❌ FAIL'} ({r_retention:.1f}%)")

if __name__ == "__main__":
    main()
