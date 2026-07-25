#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main

echo "TIME"
date
echo "PROCESS"
pid_file="output/logs/20260709_stage3_to_5_after_nan_fix.pid"
if [[ -f "$pid_file" ]]; then
  pid="$(cat "$pid_file")"
  ps -o pid,stat,etime,pcpu,pmem,cmd -p "$pid" || true
else
  echo "no pid file"
fi

summarize_csv() {
  local stage="$1"
  local csv_path="output/20260709_full_multisource/stage_${stage}/results.csv"
  echo "STAGE_${stage}"
  if [[ ! -f "$csv_path" ]]; then
    echo "missing"
    return
  fi
  /root/miniconda3/bin/python - "$csv_path" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="") as f:
    rows = list(csv.DictReader(f))
print("epochs", len(rows))
if rows:
    total_time = float(rows[-1].get("time", 0) or 0)
    avg_time = total_time / len(rows)
    print("time_sec", f"{total_time:.1f}", "avg_sec", f"{avg_time:.1f}", "avg_min", f"{avg_time / 60:.2f}")
    for row in rows[-3:]:
        print(
            "last",
            row.get("epoch"),
            row.get("metrics/precision(B)"),
            row.get("metrics/recall(B)"),
            row.get("metrics/mAP50(B)"),
            row.get("metrics/mAP50-95(B)"),
        )
    best = max(rows, key=lambda r: float(r.get("metrics/mAP50(B)", 0) or 0))
    print(
        "best",
        best.get("epoch"),
        best.get("metrics/precision(B)"),
        best.get("metrics/recall(B)"),
        best.get("metrics/mAP50(B)"),
        best.get("metrics/mAP50-95(B)"),
    )
    nan_rows = [r.get("epoch") for r in rows if "nan" in ",".join(r.values()).lower()]
    print("nan_epochs", ",".join(nan_rows) if nan_rows else "none")
PY
}

summarize_csv 3
summarize_csv 4
summarize_csv 5

echo "STAGE_DIRS"
find output/20260709_full_multisource -maxdepth 1 -type d -name 'stage_*' -printf '%f\n' | sort
