#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main

date '+%F %T'
echo MODEL_SEARCH
if command -v rg >/dev/null 2>&1; then
  rg -n 'Transformer|Attention|MHSA|C2fAttn|SimAM|EMA' ultralytics subway_yolo config models scripts 2>/dev/null | head -80 || true
else
  grep -RInE 'Transformer|Attention|MHSA|C2fAttn|SimAM|EMA' ultralytics subway_yolo config models scripts 2>/dev/null | head -80 || true
fi
echo PROCS
ps -eo pid,ppid,stat,etime,cmd | grep -E 'remote_phase5|remote_phase6|train_pipeline|evaluate_frozen|calibrate|collect_hard' | grep -v grep || true
echo MARKERS
find 'output/7.14训练结果' -maxdepth 5 -type f \( \
  -name 'COMPLETE' -o -name '*STOP*' -o -name '*FAILED*' -o \
  -name 'selection.json' -o -name 'seed_summary.json' -o \
  -name 'thresholds.json' -o -name 'summary.txt' \) | sort | tail -80
