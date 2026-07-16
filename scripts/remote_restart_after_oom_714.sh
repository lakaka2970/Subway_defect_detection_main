#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main
RUN='output/7.14训练结果'
kill_tree() {
  local pid=$1 child
  for child in $(pgrep -P "$pid" || true); do kill_tree "$child"; done
  kill "$pid" 2>/dev/null || true
}
for name in after_crops phase4 phase5 phase6; do
  [ -f "$RUN/$name.pid" ] || continue
  pid=$(cat "$RUN/$name.pid")
  kill_tree "$pid"
done
sleep 1
rm -rf "$RUN/phase2"
nohup bash scripts/remote_after_crops_714.sh > "$RUN/after_crops.log" 2>&1 &
echo $! > "$RUN/after_crops.pid"
nohup bash scripts/remote_phase4_stage2_ablation_714.sh > "$RUN/phase4.log" 2>&1 &
echo $! > "$RUN/phase4.pid"
nohup bash scripts/remote_phase5_stage3_screen_714.sh > "$RUN/phase5.log" 2>&1 &
echo $! > "$RUN/phase5.pid"
nohup bash scripts/remote_phase6_stage45_714.sh > "$RUN/phase6.log" 2>&1 &
echo $! > "$RUN/phase6.pid"
echo RESTARTED
