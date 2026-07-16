#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/projects/Subway_defect_detection_main
mkdir -p output/logs
rm -f weights/yolo11m.pt

LOG="output/logs/20260709_full_stage1a_to_5.log"
PIDFILE="output/logs/20260709_full_stage1a_to_5.pid"

setsid /root/miniconda3/bin/python scripts/train_pipeline.py \
  --stages 1a 1b 2 3 4 5 \
  --model yolo11m-EMA-SimAM \
  --device 0 \
  --output output/20260709_full_multisource \
  --pretrained output/20260704_121435/stage_1a/weights/best.pt \
  >"${LOG}" 2>&1 < /dev/null &

echo $! > "${PIDFILE}"
echo "started pid $(cat "${PIDFILE}")"
echo "log ${LOG}"
