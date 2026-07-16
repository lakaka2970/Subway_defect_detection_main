#!/usr/bin/env python3
"""Apply frozen per-class thresholds once on a held-out test split."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from calibrate_thresholds import collect_predictions, load_names, resolve_split_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--min-conf", type=float, default=.001)
    parser.add_argument("--iou", type=float, default=.5)
    args = parser.parse_args()

    names = load_names(args.data.resolve())
    thresholds = json.loads(args.thresholds.read_text(encoding="utf-8"))
    image_dir = resolve_split_path(args.data.resolve(), args.split)
    rows, gt_count = collect_predictions(
        args.model, image_dir, args.imgsz, args.device, args.min_conf, args.iou,
    )
    report = {}
    for workpoint in ("auto_report", "human_review"):
        kept = [
            row for row in rows
            if row["confidence"] >= thresholds[names[row["class_id"]]][workpoint]["conf"]
        ]
        per_class = {}
        total = Counter(tp=0, fp=0, fn=0)
        for class_id, name in names.items():
            selected = [row for row in kept if row["class_id"] == class_id]
            tp = sum(row["tp"] for row in selected)
            fp = len(selected) - tp
            fn = gt_count.get(class_id, 0) - tp
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / gt_count.get(class_id, 0) if gt_count.get(class_id, 0) else 0.0
            per_class[name] = {"threshold": thresholds[name][workpoint]["conf"],
                               "tp": tp, "fp": fp, "fn": fn,
                               "precision": precision, "recall": recall}
            total.update(tp=tp, fp=fp, fn=fn)
        total["precision"] = total["tp"] / max(total["tp"] + total["fp"], 1)
        total["recall"] = total["tp"] / max(total["tp"] + total["fn"], 1)
        report[workpoint] = {"overall": dict(total), "per_class": per_class}

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "calibrated_test_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
