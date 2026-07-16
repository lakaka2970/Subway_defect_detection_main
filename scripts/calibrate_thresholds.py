#!/usr/bin/env python3
"""Calibrate per-class confidence thresholds for a YOLO detector.

The script evaluates predictions on a YOLO-format validation split and writes:

- thresholds.json: recommended threshold and P/R/F2 per class
- pr_curves.json: all sampled threshold points per class

It is intentionally independent from training internals so it can be run after
any Stage 4/5 checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subway_yolo import YOLO


IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def xywhn_to_xyxy(box: List[float]) -> List[float]:
    x, y, w, h = box
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def iou(a: List[float], b: List[float]) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def resolve_split_path(data_yaml: Path, split: str) -> Path:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(data.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_value = data.get(split) or data.get("val")
    if split_value is None:
        raise ValueError(f"{data_yaml} does not define '{split}' or 'val'")
    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = (root / split_path).resolve()
    return split_path


def load_names(data_yaml: Path) -> Dict[int, str]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = data.get("names")
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {i: str(v) for i, v in enumerate(names)}
    raise ValueError(f"{data_yaml} does not contain YOLO class names")


def iter_images(image_dir: Path) -> Iterable[Path]:
    if image_dir.is_file():
        yield image_dir
        return
    for p in sorted(image_dir.rglob("*")):
        if p.suffix.lower() in IMG_SUFFIXES:
            yield p


def label_candidates(image_path: Path) -> List[Path]:
    parts = list(image_path.parts)
    candidates = []
    for i, part in enumerate(parts):
        if part == "images":
            label_parts = parts[:]
            label_parts[i] = "labels"
            candidates.append(Path(*label_parts).with_suffix(".txt"))
    candidates.append(image_path.parent.parent / "labels" / f"{image_path.stem}.txt")
    candidates.append(image_path.with_suffix(".txt"))
    return candidates


def load_gts(image_path: Path) -> List[Dict]:
    label_path = next((p for p in label_candidates(image_path) if p.exists()), None)
    if label_path is None:
        return []
    gts = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        coords = [float(v) for v in parts[1:5]]
        gts.append({"cls": cls, "box": xywhn_to_xyxy(coords)})
    return gts


def collect_predictions(
    model_path: Path,
    image_dir: Path,
    imgsz: int,
    device: str,
    min_conf: float,
    iou_threshold: float,
) -> Tuple[List[Dict], Dict[int, int]]:
    model = YOLO(str(model_path))
    rows: List[Dict] = []
    gt_count: Dict[int, int] = {}

    images = list(iter_images(image_dir))
    # A full Python list is interpreted as one giant in-memory batch, while a
    # directory stream is single-image in this vendored version.  Explicit
    # four-image chunks preserve batching without unbounded allocation.
    def iter_batched_results():
        for start in range(0, len(images), 4):
            batch_paths = images[start:start + 4]
            batch_results = model.predict(
                source=[str(path) for path in batch_paths], conf=min_conf,
                imgsz=imgsz, device=device, batch=len(batch_paths),
                stream=False, verbose=False,
            )
            yield from zip(batch_paths, batch_results)

    for image_path, result in iter_batched_results():
        gts = load_gts(image_path)
        for gt in gts:
            gt_count[gt["cls"]] = gt_count.get(gt["cls"], 0) + 1

        boxes = result.boxes
        if boxes is None or len(boxes.cls) == 0:
            continue

        used = set()
        xyxyn = boxes.xyxyn.cpu().numpy()
        cls_arr = boxes.cls.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        order = sorted(range(len(conf_arr)), key=lambda i: float(conf_arr[i]), reverse=True)
        for pred_i in order:
            pred_cls = int(cls_arr[pred_i])
            pred_box = [float(v) for v in xyxyn[pred_i]]
            best_gt = -1
            best_iou = 0.0
            for gt_i, gt in enumerate(gts):
                if gt_i in used or gt["cls"] != pred_cls:
                    continue
                score = iou(pred_box, gt["box"])
                if score > best_iou:
                    best_iou = score
                    best_gt = gt_i
            is_tp = best_gt >= 0 and best_iou >= iou_threshold
            if is_tp:
                used.add(best_gt)
            rows.append({
                "class_id": pred_cls,
                "confidence": float(conf_arr[pred_i]),
                "tp": bool(is_tp),
                "iou": best_iou,
                "image": str(image_path),
            })

    return rows, gt_count


def precision_recall_f2(tp: int, fp: int, gt: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / gt if gt else 0.0
    beta2 = 4.0
    denom = beta2 * precision + recall
    f2 = (1 + beta2) * precision * recall / denom if denom else 0.0
    return precision, recall, f2


def calibrate(
    rows: List[Dict],
    gt_count: Dict[int, int],
    names: Dict[int, str],
    target_precision: float,
    target_recall: float,
) -> Tuple[Dict, Dict]:
    thresholds = {}
    curves = {}
    for class_id, class_name in names.items():
        class_rows = [r for r in rows if r["class_id"] == class_id]
        gt = gt_count.get(class_id, 0)
        candidates = sorted({round(r["confidence"], 4) for r in class_rows}, reverse=True)
        if not candidates:
            candidates = [0.5]

        points = []
        best = None
        # One descending pass gives the same threshold sets without rescanning
        # every detection for every candidate (the former O(N^2) path becomes
        # prohibitive for hard-negative-heavy models at conf=0.001).
        ordered = sorted(class_rows, key=lambda r: r["confidence"], reverse=True)
        cursor = tp = fp = 0
        for threshold in candidates:
            while cursor < len(ordered) and ordered[cursor]["confidence"] >= threshold:
                if ordered[cursor]["tp"]:
                    tp += 1
                else:
                    fp += 1
                cursor += 1
            precision, recall, f2 = precision_recall_f2(tp, fp, gt)
            point = {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f2_score": f2,
                "detections": tp + fp,
                "tp": tp,
                "fp": fp,
            }
            points.append(point)
            if best is None:
                best = point
            meets = precision >= target_precision and recall >= target_recall
            best_meets = best["precision"] >= target_precision and best["recall"] >= target_recall
            if (meets and not best_meets) or (meets == best_meets and f2 > best["f2_score"]):
                best = point

        # Two operating points are required by the frozen plan.  Auto-report
        # prioritises precision and then recall.  Human-review uses the F2
        # optimum (recall weighted 2x).  Impossible targets are reported
        # honestly rather than marked as passing.
        auto_candidates = [p for p in points if p["precision"] >= target_precision]
        auto = max(auto_candidates, key=lambda p: (p["recall"], p["f2_score"], p["threshold"])) \
            if auto_candidates else max(points, key=lambda p: (p["precision"], p["recall"], p["f2_score"]))
        human = max(points, key=lambda p: (p["f2_score"], p["recall"], p["precision"]))

        curves[class_name] = points
        thresholds[class_name] = {
            "auto_report": {
                "conf": auto["threshold"], "precision": auto["precision"],
                "recall": auto["recall"], "f2_score": auto["f2_score"],
                "tp": auto["tp"], "fp": auto["fp"],
                "meets_precision_target": auto["precision"] >= target_precision,
            },
            "human_review": {
                "conf": human["threshold"], "precision": human["precision"],
                "recall": human["recall"], "f2_score": human["f2_score"],
                "tp": human["tp"], "fp": human["fp"],
                "meets_recall_target": human["recall"] >= target_recall,
            },
            "recommended_threshold": best["threshold"],
            "precision": best["precision"],
            "recall": best["recall"],
            "f2_score": best["f2_score"],
            "gt_count": gt,
            "detections_at_threshold": best["detections"],
            "tp": best["tp"],
            "fp": best["fp"],
            "meets_targets": best["precision"] >= target_precision and best["recall"] >= target_recall,
        }
    return thresholds, curves


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate YOLO per-class thresholds")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", default=1280, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-conf", default=0.05, type=float)
    parser.add_argument("--iou", default=0.5, type=float)
    parser.add_argument("--target-precision", default=0.90, type=float)
    parser.add_argument("--target-recall", default=0.90, type=float)
    parser.add_argument("--output", default=None, type=Path)
    args = parser.parse_args()

    data_yaml = args.data.resolve()
    image_dir = resolve_split_path(data_yaml, args.split)
    names = load_names(data_yaml)
    rows, gt_count = collect_predictions(
        args.model,
        image_dir,
        args.imgsz,
        args.device,
        args.min_conf,
        args.iou,
    )
    thresholds, curves = calibrate(
        rows,
        gt_count,
        names,
        args.target_precision,
        args.target_recall,
    )

    output_dir = args.output
    if output_dir is None:
        output_dir = args.model.resolve().parents[1] / "calibrated_thresholds"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "thresholds.json").write_text(
        json.dumps(thresholds, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "pr_curves.json").write_text(
        json.dumps(curves, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "per_class_report.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["class", "workpoint", "conf", "precision", "recall", "f2", "tp", "fp", "target_met"])
        for class_name, item in thresholds.items():
            for workpoint, target_key in (("auto_report", "meets_precision_target"),
                                          ("human_review", "meets_recall_target")):
                point = item[workpoint]
                writer.writerow([class_name, workpoint, point["conf"], point["precision"],
                                 point["recall"], point["f2_score"], point["tp"], point["fp"],
                                 point[target_key]])
    summary_lines = [
        f"model: {args.model}", f"data: {args.data}",
        f"target_precision: {args.target_precision}", f"target_recall: {args.target_recall}",
        "",
    ]
    for class_name, item in thresholds.items():
        auto, human = item["auto_report"], item["human_review"]
        summary_lines.append(
            f"{class_name}: auto conf={auto['conf']:.4f} P={auto['precision']:.4f} "
            f"R={auto['recall']:.4f} met={auto['meets_precision_target']}; "
            f"review conf={human['conf']:.4f} P={human['precision']:.4f} "
            f"R={human['recall']:.4f} met={human['meets_recall_target']}"
        )
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"Images: {len(list(iter_images(image_dir)))}")
    print(f"Detections: {len(rows)}")
    print(f"Output: {output_dir}")
    for class_name, item in thresholds.items():
        print(
            f"{class_name}: th={item['recommended_threshold']:.3f} "
            f"P={item['precision']:.3f} R={item['recall']:.3f} "
            f"F2={item['f2_score']:.3f} GT={item['gt_count']}"
        )


if __name__ == "__main__":
    main()
