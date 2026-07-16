#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/projects/Subway_defect_detection_main
mkdir -p output/logs

if [ -d output/20260709_full_multisource/stage_3 ]; then
  suffix="$(date +%H%M%S)"
  mv output/20260709_full_multisource/stage_3 "output/20260709_full_multisource/stage_3_bad_nan_${suffix}"
fi

rm -f data/subway_crops/train/labels.cache data/subway_crops/val/labels.cache

LOG="output/logs/20260709_stage3_to_5_after_nan_fix.log"
PIDFILE="output/logs/20260709_stage3_to_5_after_nan_fix.pid"

setsid /root/miniconda3/bin/python scripts/train_pipeline.py \
  --stages 3 4 5 \
  --model yolo11m-EMA-SimAM \
  --device 0 \
  --batch 4 \
  --output output/20260709_full_multisource \
  --pretrained output/20260709_full_multisource/stage_2/weights/best.pt \
  >"${LOG}" 2>&1 < /dev/null &

echo $! > "${PIDFILE}"
echo "started pid $(cat "${PIDFILE}")"
echo "log ${LOG}"
