#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main

run_dir="output/20260709_full_multisource"

echo "RUN_DIR $run_dir"
echo "TIME"
date

for stage in 1a 1b 2 3 4 5; do
  csv_path="$run_dir/stage_${stage}/results.csv"
  echo "STAGE $stage"
  if [[ ! -f "$csv_path" ]]; then
    echo "missing"
    continue
  fi
  /root/miniconda3/bin/python - "$csv_path" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="") as f:
    rows = list(csv.DictReader(f))
if not rows:
    print("empty")
    raise SystemExit

def get(row, key):
    return float(row.get(key, 0) or 0)

total_time = get(rows[-1], "time")
best = max(rows, key=lambda r: get(r, "metrics/mAP50(B)"))
last = rows[-1]
print("epochs", len(rows))
print("time_sec", f"{total_time:.1f}")
print("avg_min", f"{total_time / len(rows) / 60:.2f}")
print(
    "best",
    best.get("epoch"),
    f"{get(best, 'metrics/precision(B)'):.5f}",
    f"{get(best, 'metrics/recall(B)'):.5f}",
    f"{get(best, 'metrics/mAP50(B)'):.5f}",
    f"{get(best, 'metrics/mAP50-95(B)'):.5f}",
)
print(
    "last",
    last.get("epoch"),
    f"{get(last, 'metrics/precision(B)'):.5f}",
    f"{get(last, 'metrics/recall(B)'):.5f}",
    f"{get(last, 'metrics/mAP50(B)'):.5f}",
    f"{get(last, 'metrics/mAP50-95(B)'):.5f}",
)
nan_rows = [r.get("epoch") for r in rows if "nan" in ",".join(r.values()).lower()]
print("nan_epochs", ",".join(nan_rows) if nan_rows else "none")
PY
done

echo "WEIGHTS"
find "$run_dir" -path '*/weights/best.pt' -o -path '*/weights/last.pt' | sort

echo "GLOBAL_WEIGHTS"
find weights -maxdepth 1 -type f -name 'stage*.pt' -printf '%p\n' | sort

echo "THRESHOLDS"
find "$run_dir" -name 'thresholds.json' -o -name 'pr_curves.json' | sort

echo "DATASETS"
for d in data/multi_datasets/mixed_pretrain data/Defect_dataset data/subway_crops_1024 data/subway_crops; do
  echo "DATASET $d"
  find "$d" -type f \( -name '*.jpg' -o -name '*.png' -o -name '*.jpeg' \) | wc -l
  find "$d" -type f -name '*.txt' | wc -l
done
