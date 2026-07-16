#!/usr/bin/env python3
"""Evaluate a checkpoint on a frozen split with one comparable protocol."""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from calibrate_thresholds import (
    calibrate, collect_predictions, iter_images, load_names, resolve_split_path,
)
from subway_yolo import YOLO


def source_id(path: str) -> str:
    stem = Path(path).stem
    stem = re.sub(r"_[pn]\d+$", "", stem)
    return re.sub(r"_\d+_\d+$", "", stem)


def bootstrap(values: list[float], seed: int = 714, n: int = 1000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return [means[int(.025 * n)], means[min(n - 1, int(.975 * n))]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--min-conf", type=float, default=.001)
    parser.add_argument("--iou", type=float, default=.5)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data.resolve()), split=args.split, imgsz=args.imgsz,
        conf=args.min_conf, iou=args.iou, device=args.device, batch=args.batch,
        plots=True, save_json=True, project=str(args.output.parent), name=args.output.name,
        exist_ok=True, verbose=False,
    )
    result_dict = {k: float(v) for k, v in getattr(metrics, "results_dict", {}).items()}
    names = load_names(args.data.resolve())
    maps = list(getattr(getattr(metrics, "box", None), "maps", []))
    ap50s = list(getattr(getattr(metrics, "box", None), "ap50", []))
    per_class_ap = {names[i]: float(maps[i]) for i in range(min(len(maps), len(names)))}
    per_class_ap50 = {names[i]: float(ap50s[i]) for i in range(min(len(ap50s), len(names)))}

    image_dir = resolve_split_path(args.data.resolve(), args.split)
    rows, gt_count = collect_predictions(
        args.model, image_dir, args.imgsz, args.device, args.min_conf, args.iou,
    )
    thresholds, curves = calibrate(rows, gt_count, names, .90, .80)

    operating = {i: thresholds[name]["human_review"]["conf"] for i, name in names.items()}
    kept = [row for row in rows if row["confidence"] >= operating.get(row["class_id"], 1.0)]
    tp = sum(row["tp"] for row in kept)
    fp = len(kept) - tp
    total_gt = sum(gt_count.values())
    fn = total_gt - tp
    image_count = len(list(iter_images(image_dir)))

    per_class = {}
    for class_id, name in names.items():
        points = curves[name]
        p_at_r40 = max((p["precision"] for p in points if p["recall"] >= .40), default=0.0)
        p_at_r60 = max((p["precision"] for p in points if p["recall"] >= .60), default=0.0)
        r_at_p50 = max((p["recall"] for p in points if p["precision"] >= .50), default=0.0)
        per_class[name] = {
            "gt": gt_count.get(class_id, 0), "ap50": per_class_ap50.get(name),
            "ap50_95": per_class_ap.get(name),
            "p_at_r40": p_at_r40, "p_at_r60": p_at_r60, "r_at_p50": r_at_p50,
            "f2_operating_point": thresholds[name]["human_review"],
        }

    source_rows = defaultdict(lambda: Counter(tp=0, fp=0, detections=0))
    for row in kept:
        source = source_id(row["image"])
        source_rows[source]["detections"] += 1
        source_rows[source]["tp" if row["tp"] else "fp"] += 1
    source_precision = []
    with (args.output / "per_source_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["source_id", "tp", "fp", "detections", "precision"])
        for source, counts in sorted(source_rows.items()):
            precision = counts["tp"] / counts["detections"] if counts["detections"] else 0.0
            source_precision.append(precision)
            writer.writerow([source, counts["tp"], counts["fp"], counts["detections"], precision])

    with (args.output / "tp_fp_samples.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["class_id", "confidence", "tp", "iou", "image"])
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "model": str(args.model), "data": str(args.data), "protocol": vars(args),
        "ultralytics_metrics": result_dict, "per_class": per_class,
        "f2_workpoint": {
            "tp": tp, "fp": fp, "fn": fn,
            "fp_per_1000_crops": fp * 1000 / max(image_count, 1),
            "fn_per_1000_gt": fn * 1000 / max(total_gt, 1),
        },
        "source_precision_bootstrap_95ci": bootstrap(source_precision),
        "image_count": image_count, "gt_count": total_gt,
    }
    (args.output / "evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
