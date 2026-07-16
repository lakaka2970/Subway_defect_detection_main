#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main
GOOD='output/7.14训练结果'
BAD='output/7.14????'
mkdir -p "$GOOD/generated_configs"
if [ -d "$BAD/generated_configs" ]; then cp -a "$BAD/generated_configs/." "$GOOD/generated_configs/"; fi
for name in run_manifest.yaml dataset_manifest.json eval_manifest.json pip_freeze.txt git_commit.txt git_status.txt code_changes.patch train_command.txt; do
  [ ! -f "$BAD/$name" ] || cp "$BAD/$name" "$GOOD/$name"
done

for pidname in phase4 phase5 phase6; do
  if [ -f "$GOOD/$pidname.pid" ]; then
    pid=$(cat "$GOOD/$pidname.pid")
    children=$(pgrep -P "$pid" || true)
    [ -z "$children" ] || kill $children 2>/dev/null || true
    kill "$pid" 2>/dev/null || true
  fi
done
sleep 1
nohup bash scripts/remote_phase4_stage2_ablation_714.sh > "$GOOD/phase4.log" 2>&1 &
echo $! > "$GOOD/phase4.pid"
nohup bash scripts/remote_phase5_stage3_screen_714.sh > "$GOOD/phase5.log" 2>&1 &
echo $! > "$GOOD/phase5.pid"
nohup bash scripts/remote_phase6_stage45_714.sh > "$GOOD/phase6.log" 2>&1 &
echo $! > "$GOOD/phase6.pid"
echo REPAIRED
