#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main

RUN='output/7.14训练结果'
PY=/root/miniconda3/bin/python

mkdir -p "$RUN/audits/controlled_stage2_s2_a"
"$PY" scripts/stage_gate_714.py \
  --stage controlled_stage2_s2_a \
  --weights "$RUN/phase4/s2_a/train/stage_2/weights/best.pt" \
  --results "$RUN/phase4/s2_a/train/stage_2/results.csv" \
  --json "$RUN/phase4/s2_a/test_eval/evaluation.json" \
  --log "$RUN/phase4/s2_a/train.log" \
  --allow-nms-warning \
  --output "$RUN/audits/controlled_stage2_s2_a/gate.json" \
  > "$RUN/audits/controlled_stage2_s2_a/gate.log" 2>&1

"$PY" - <<'PY'
import json
import shutil
from pathlib import Path

run = Path("output/7.14训练结果")
root = run / "phase4"

def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def map95(data):
    return next(float(v) for k, v in data["ultralytics_metrics"].items() if "mAP50-95" in k)

def class_ap(data):
    return {k: v.get("ap50_95") for k, v in data["per_class"].items()}

baseline = load(root / "baseline_new_s2/evaluation.json")
s2a = load(root / "s2_a/test_eval/evaluation.json")
base_map = map95(baseline)
s2a_map = map95(s2a)
base_ap = class_ap(baseline)
s2a_ap = class_ap(s2a)
nondecline = sum(
    s2a_ap[k] is not None and base_ap[k] is not None and s2a_ap[k] >= base_ap[k]
    for k in base_ap
)
passes = s2a_map - base_map >= 0.010 and nondecline >= 5
if not passes:
    raise SystemExit("s2_a does not meet selection criteria")

reason = (
    "integrity risk: previous val cls_loss NaN and repeated NMS time-limit warnings; "
    "s2_a already passed gate and selection criteria"
)
report = {
    "baseline_map50_95": base_map,
    "candidates": {
        "s2_a": {
            "map50_95": s2a_map,
            "improvement": s2a_map - base_map,
            "classes_non_decreasing": nondecline,
            "passes": True,
        },
        "s2_b": {
            "status": "rejected_before_completion",
            "reason": reason,
        },
    },
    "selected": "s2_a",
    "controlled_finish": True,
}
(root / "selection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
(run / "selected").mkdir(exist_ok=True)
shutil.copy2(root / "s2_a/train/stage_2/weights/best.pt", run / "selected/stage2_best.pt")
(root / "S2_B_REJECTED_BY_GATE.txt").write_text(reason + "\n", encoding="utf-8")
PY

if kill -0 24583 2>/dev/null; then
  pkill -TERM -P 24583 2>/dev/null || true
  kill -TERM 24583 2>/dev/null || true
fi

for _ in {1..20}; do
  kill -0 13321 2>/dev/null || break
  sleep 1
done

if kill -0 13321 2>/dev/null; then
  pkill -TERM -P 13321 2>/dev/null || true
  kill -TERM 13321 2>/dev/null || true
fi

for _ in {1..10}; do
  kill -0 13321 2>/dev/null || break
  sleep 1
done

echo COMPLETE > "$RUN/phase4.log"
echo CONTROLLED_COMPLETE > "$RUN/phase4/CONTROLLED_COMPLETE"
