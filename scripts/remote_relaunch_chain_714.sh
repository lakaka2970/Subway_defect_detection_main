#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main
RUN='output/7.14训练结果'
mkdir -p "$RUN"

master=$(pgrep -f '^bash scripts/remote_generate_crops_714.sh$' | head -1 || true)
if [ -z "$master" ]; then
  # The master may already have completed; locate its launcher log below.
  master=0
else
  echo "$master" > "$RUN/phase0_crops.pid"
  while kill -0 "$master" 2>/dev/null; do sleep 10; done
fi

launcher_log=$(find output -mindepth 2 -maxdepth 2 -type f -name phase0_crops_parallel.log \
  ! -path "$RUN/*" -print -quit)
test -n "$launcher_log"
cp "$launcher_log" "$RUN/phase0_crops_parallel.log"
grep -q '^COMPLETE$' "$RUN/phase0_crops_parallel.log"
grep -q 'VALIDATION PASSED' "$RUN/phase0_validation.log"

nohup bash scripts/remote_after_crops_714.sh > "$RUN/after_crops.log" 2>&1 &
echo $! > "$RUN/after_crops.pid"
nohup bash scripts/remote_generate_crops_1024_714.sh > "$RUN/crop1024.log" 2>&1 &
echo $! > "$RUN/crop1024.pid"
nohup bash scripts/remote_phase4_stage2_ablation_714.sh > "$RUN/phase4.log" 2>&1 &
echo $! > "$RUN/phase4.pid"
nohup bash scripts/remote_phase5_stage3_screen_714.sh > "$RUN/phase5.log" 2>&1 &
echo $! > "$RUN/phase5.pid"
nohup bash scripts/remote_phase6_stage45_714.sh > "$RUN/phase6.log" 2>&1 &
echo $! > "$RUN/phase6.pid"

echo "CHAIN_STARTED"
