#!/usr/bin/env bash
set -u

ROOT="/root/autodl-tmp/projects/Subway_defect_detection_main"
RUN="$ROOT/output/7.14训练结果"
LOG="$RUN/logs/watchdog_714.log"
PY="/root/miniconda3/bin/python"

mkdir -p "$RUN/logs"

while true; do
  {
    echo "===== $(date '+%F %T') ====="
    cd "$ROOT" || exit 0
    ps -eo pid,ppid,stat,etime,cmd | grep -E 'remote_phase5|remote_phase6|train_pipeline|evaluate_frozen|stage_gate|collect_hard_negatives|calibrate_thresholds' | grep -v grep || true
    "$PY" - <<'PY'
from pathlib import Path
import json

root = Path("output/7.14训练结果")
patterns = [
    "phase5/screen/*/train/stage_3/results.csv",
    "phase5/full/*/train/stage_3/results.csv",
    "phase6/**/results.csv",
]
for pat in patterns:
    for p in sorted(root.glob(pat)):
        lines = p.read_text(errors="replace").splitlines()
        if not lines:
            continue
        print("RESULT", p, "epochs", max(0, len(lines) - 1), "last", lines[-1][:220])

for p in sorted(root.glob("**/*")):
    if p.is_file() and ("STOP" in p.name or "FAILED" in p.name):
        print("MARKER", p)

for p in sorted((root / "audits").glob("**/*.json"))[-40:]:
    try:
        data = json.loads(p.read_text(errors="replace"))
        status = data.get("status") or data.get("passed") or data.get("ok")
        print("AUDIT", p, status)
    except Exception:
        print("AUDIT", p)
PY
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader 2>/dev/null || true
  } >> "$LOG" 2>&1
  sleep 300
done
