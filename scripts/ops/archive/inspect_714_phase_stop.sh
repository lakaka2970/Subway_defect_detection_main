#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main

date '+%F %T'
echo PROCS
ps -eo pid,ppid,stat,etime,cmd | grep -E 'remote_phase5|remote_phase6|train_pipeline|evaluate_frozen|stage_gate|collect_hard|calibrate' | grep -v grep || true

echo PHASE5_LOG
tail -n 160 'output/7.14训练结果/phase5.log' 2>/dev/null || true

echo PHASE6_LOG
tail -n 120 'output/7.14训练结果/phase6.log' 2>/dev/null || true

echo S3D_AUDIT
for f in \
  output/7.14训练结果/audits/phase5_screen_s3_d/*.json \
  output/7.14训练结果/audits/phase5_screen_s3_d/*.log \
  output/7.14训练结果/phase5/screen/s3_d/calibration_eval.log \
  output/7.14训练结果/phase5/screen/s3_d/STOP_CONDITION.txt
do
  if [ -f "$f" ]; then
    echo "FILE:$f"
    tail -n 120 "$f"
  fi
done

echo MARKERS
find 'output/7.14训练结果' -maxdepth 6 -type f \( \
  -name 'COMPLETE' -o -name '*STOP*' -o -name '*FAILED*' -o \
  -name 'selection.json' -o -name 'seed_summary.json' -o \
  -name 'thresholds.json' -o -name 'summary.txt' \) | sort | tail -120

echo GPU
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader || true
