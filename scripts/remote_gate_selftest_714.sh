#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main
RUN='output/7.14训练结果'
mkdir -p "$RUN/audits/gate_selftest"
/root/miniconda3/bin/python scripts/stage_gate_714.py \
  --stage gate_selftest --check-code \
  --weights "$RUN/phase4/s2_a/train/stage_2/weights/best.pt" \
  --results "$RUN/phase4/s2_a/train/stage_2/results.csv" \
  --json "$RUN/phase4/s2_a/test_eval/evaluation.json" \
  --log "$RUN/phase4/s2_a/train.log" \
  --allow-nms-warning \
  --output "$RUN/audits/gate_selftest/gate.json" \
  > "$RUN/audits/gate_selftest/gate.log" 2>&1
/root/miniconda3/bin/python - <<'PY'
import json
from pathlib import Path
p=Path('output/7.14训练结果/audits/gate_selftest/gate.json')
d=json.loads(p.read_text(encoding='utf-8'))
print(d['status'], d['results'][0]['epochs'], d['weights'][0]['floating_tensors'], d['code']['focal_numeric_test'])
PY
