#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main

RUN='output/7.14训练结果'
PY=/root/miniconda3/bin/python
DATA=data/subway_crops/test.yaml

test -s "$RUN/phase0_validation.log"
grep -q 'VALIDATION PASSED' "$RUN/phase0_validation.log"

declare -A MODELS=(
  [old_s4]='output/20260704_121435/stage_4/weights/best.pt'
  [new_s3]='output/20260709_full_multisource/stage_3/weights/best.pt'
  [new_s4]='output/20260709_full_multisource/stage_4/weights/best.pt'
  [new_s5]='output/20260709_full_multisource/stage_5/weights/best.pt'
)

for name in old_s4 new_s3 new_s4 new_s5; do
  model=${MODELS[$name]}
  test -s "$model"
  mkdir -p "$RUN/phase1/$name"
  "$PY" scripts/evaluate_frozen.py \
    --model "$model" --data "$DATA" --split val \
    --imgsz 1280 --iou 0.5 --min-conf 0.001 --batch 4 --device 0 \
    --output "$RUN/phase1/$name" \
    > "$RUN/phase1/$name/run.log" 2>&1
done

"$PY" - <<'PY'
import json
from pathlib import Path
root = Path('output/7.14训练结果/phase1')
summary = {}
for path in sorted(root.glob('*/evaluation.json')):
    data = json.loads(path.read_text(encoding='utf-8'))
    summary[path.parent.name] = {
        'metrics': data['ultralytics_metrics'],
        'f2_workpoint': data['f2_workpoint'],
        'source_precision_bootstrap_95ci': data['source_precision_bootstrap_95ci'],
    }
(root / 'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
PY
