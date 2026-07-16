#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main
RUN='output/7.14训练结果'

for name in phase5 phase6; do
  if [ -s "$RUN/$name.pid" ]; then
    pid=$(cat "$RUN/$name.pid")
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
  fi
done

nohup bash scripts/remote_phase5_stage3_screen_714.sh > "$RUN/phase5.log" 2>&1 &
echo $! > "$RUN/phase5.pid"
nohup bash scripts/remote_phase6_stage45_714.sh > "$RUN/phase6.log" 2>&1 &
echo $! > "$RUN/phase6.pid"
echo GATED_PHASES_REARMED
