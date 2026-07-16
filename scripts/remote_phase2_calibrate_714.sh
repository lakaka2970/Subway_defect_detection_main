#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main

RUN='output/7.14训练结果'
PY=/root/miniconda3/bin/python

declare -A MODELS=(
  [old_s4]='output/20260704_121435/stage_4/weights/best.pt'
  [new_s4]='output/20260709_full_multisource/stage_4/weights/best.pt'
  [new_s5]='output/20260709_full_multisource/stage_5/weights/best.pt'
)

for name in old_s4 new_s4 new_s5; do
  model=${MODELS[$name]}
  min_conf=0.01
  # The legacy checkpoint floods NMS under the revised labels at 0.001 and
  # causes per-batch truncation, so use a safe floor only for that baseline.
  if [ "$name" = "old_s4" ]; then
    min_conf=0.05
  fi
  out="$RUN/phase2/$name"
  mkdir -p "$out"
  "$PY" scripts/calibrate_thresholds.py \
    --model "$model" --data data/subway_crops/calibration.yaml --split val \
    --imgsz 1280 --device 0 --min-conf "$min_conf" --iou 0.5 \
    --target-precision 0.80 --target-recall 0.40 --output "$out/calibration" \
    > "$out/calibration.log" 2>&1
  "$PY" scripts/evaluate_calibrated_thresholds.py \
    --model "$model" --data data/subway_crops/test.yaml --split val \
    --thresholds "$out/calibration/thresholds.json" --output "$out/test" \
    --imgsz 1280 --device 0 --min-conf "$min_conf" --iou 0.5 \
    > "$out/test.log" 2>&1
done
