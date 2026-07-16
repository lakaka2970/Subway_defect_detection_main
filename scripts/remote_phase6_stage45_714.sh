#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main
RUN='output/7.14训练结果'
PY=/root/miniconda3/bin/python

preflight_gate() {
  local stage=$1 audit="$RUN/audits/$1"
  mkdir -p "$audit"
  "$PY" scripts/validate_dataset.py --dataset_root data/subway_crops \
    --splits train hn_mining calibration test --allow-unlabelled --workers 8 \
    > "$audit/dataset.log" 2>&1
  grep -q 'VALIDATION PASSED' "$audit/dataset.log"
  "$PY" scripts/stage_gate_714.py --stage "$stage-code-data" --check-code \
    --output "$audit/code.json" > "$audit/code.log" 2>&1
}

train_gate() {
  local stage=$1 root=$2 log=$3 audit="$RUN/audits/$1"
  mkdir -p "$audit"
  "$PY" scripts/stage_gate_714.py --stage "$stage-train" \
    --weights "$root/weights/best.pt" --results "$root/results.csv" --log "$log" \
    --output "$audit/train.json" > "$audit/train_gate.log" 2>&1
}

eval_gate() {
  local stage=$1 root=$2 log=$3 audit="$RUN/audits/$1"
  mkdir -p "$audit"
  "$PY" scripts/stage_gate_714.py --stage "$stage-eval" \
    --json "$root/evaluation.json" --log "$log" --output "$audit/eval.json" \
    > "$audit/eval_gate.log" 2>&1
}

phase5_pid=$(cat "$RUN/phase5.pid")
while kill -0 "$phase5_pid" 2>/dev/null; do sleep 30; done
grep -q '^COMPLETE$' "$RUN/phase5.log"
test -s "$RUN/selected/stage3_best.pt"
preflight_gate phase6_stage4_preflight

mkdir -p "$RUN/phase6/stage4"
cp config/train/pretrain/stage4_short_finetune.yaml "$RUN/phase6/stage4/config.yaml"
"$PY" scripts/train_pipeline.py --stages 4 --model yolo11m-EMA-SimAM \
  --output "$RUN/phase6/stage4/train" --device 0 --batch 4 --workers 4 \
  --pretrained "$RUN/selected/stage3_best.pt" > "$RUN/phase6/stage4/train.log" 2>&1
S4="$RUN/phase6/stage4/train/stage_4/weights/best.pt"
test -s "$S4"
train_gate phase6_stage4 "$RUN/phase6/stage4/train/stage_4" "$RUN/phase6/stage4/train.log"
cp "$S4" "$RUN/selected/stage4_best.pt"
"$PY" scripts/evaluate_frozen.py --model "$S4" --data data/subway_crops/test.yaml \
  --output "$RUN/phase6/stage4/test_eval" --imgsz 1280 --device 0 --batch 4 \
  --min-conf .001 --iou .5 > "$RUN/phase6/stage4/test_eval.log" 2>&1
eval_gate phase6_stage4 "$RUN/phase6/stage4/test_eval" "$RUN/phase6/stage4/test_eval.log"

# Strict paired control: identical Stage 4 initialisation, before model-mined HN are added.
mkdir -p "$RUN/phase6/s5_control"
preflight_gate phase6_s5_control_preflight
cp config/train/pretrain/stage5_hard_negative.yaml "$RUN/phase6/s5_control/config.yaml"
"$PY" scripts/train_pipeline.py --stages 5 --model yolo11m-EMA-SimAM \
  --output "$RUN/phase6/s5_control/train" --device 0 --batch 4 --workers 4 \
  --pretrained "$S4" > "$RUN/phase6/s5_control/train.log" 2>&1
CONTROL="$RUN/phase6/s5_control/train/stage_5/weights/best.pt"
test -s "$CONTROL"
train_gate phase6_s5_control "$RUN/phase6/s5_control/train/stage_5" "$RUN/phase6/s5_control/train.log"

# G1: temporarily expose only train-derived hn_mining as subway_crops val.
cp data/subway_crops/subway_crops.yaml "$RUN/phase6/subway_crops_before_hn.yaml"
preflight_gate phase6_g1_preflight
"$PY" - <<'PY'
from pathlib import Path
import yaml
p=Path('data/subway_crops/subway_crops.yaml'); d=yaml.safe_load(p.read_text())
d['val']='hn_mining/images'; p.write_text(yaml.safe_dump(d,sort_keys=False),encoding='utf-8')
PY
TARGET=$(readlink -m /root/autodl-tmp/projects/Subway_defect_detection_main/data/hard_negatives_714)
case "$TARGET" in /root/autodl-tmp/projects/Subway_defect_detection_main/data/*) ;; *) exit 40;; esac
rm -rf -- "$TARGET"
"$PY" scripts/collect_hard_negatives.py --model "$S4" \
  --data data/subway_crops/subway_crops.yaml --conf .30 --device 0 --imgsz 1280 \
  --output data/hard_negatives_714 > "$RUN/phase6/g1_collect_hard_negatives.log" 2>&1
cp "$RUN/phase6/subway_crops_before_hn.yaml" data/subway_crops/subway_crops.yaml
test -s data/hard_negatives_714/summary.json
mkdir -p "$RUN/audits/phase6_g1"
"$PY" scripts/stage_gate_714.py --stage phase6_g1_collect \
  --json data/hard_negatives_714/summary.json --log "$RUN/phase6/g1_collect_hard_negatives.log" \
  --output "$RUN/audits/phase6_g1/gate.json" > "$RUN/audits/phase6_g1/gate.log" 2>&1

# G2: audit and mix unlabelled FP crops. No calibration/test source is used.
hn_count=$(find data/hard_negatives_714 -mindepth 2 -maxdepth 2 -type f -name '*.jpg' | wc -l)
[ "$hn_count" -gt 0 ]
before=$(find data/subway_crops/train/images -maxdepth 1 -type f -name '*.jpg' | wc -l)
find data/hard_negatives_714 -mindepth 2 -maxdepth 2 -type f -name '*.jpg' \
  -exec cp -n {} data/subway_crops/train/images/ \;
after=$(find data/subway_crops/train/images -maxdepth 1 -type f -name '*.jpg' | wc -l)
mixed=$((after-before))
[ "$mixed" -gt 0 ]
HN_COUNT="$hn_count" MIXED="$mixed" "$PY" - <<'PY'
import json, os
from pathlib import Path
d={'collected':int(os.environ['HN_COUNT']),'mixed':int(os.environ['MIXED']),
   'source':'train-derived hn_mining','calibration_test_leakage':0}
Path('output/7.14训练结果/phase6/g2_mix_summary.json').write_text(json.dumps(d,indent=2),encoding='utf-8')
PY
preflight_gate phase6_g2_postmix
mkdir -p "$RUN/audits/phase6_g2"
"$PY" scripts/stage_gate_714.py --stage phase6_g2_mix \
  --json "$RUN/phase6/g2_mix_summary.json" \
  --output "$RUN/audits/phase6_g2/gate.json" > "$RUN/audits/phase6_g2/gate.log" 2>&1

# G3: actual HN training, paired against control.
mkdir -p "$RUN/phase6/s5_hn"
preflight_gate phase6_s5_hn_preflight
cp config/train/pretrain/stage5_hard_negative.yaml "$RUN/phase6/s5_hn/config.yaml"
"$PY" scripts/train_pipeline.py --stages 5 --model yolo11m-EMA-SimAM \
  --output "$RUN/phase6/s5_hn/train" --device 0 --batch 4 --workers 4 \
  --pretrained "$S4" > "$RUN/phase6/s5_hn/train.log" 2>&1
HN="$RUN/phase6/s5_hn/train/stage_5/weights/best.pt"
test -s "$HN"
train_gate phase6_s5_hn "$RUN/phase6/s5_hn/train/stage_5" "$RUN/phase6/s5_hn/train.log"
cp "$HN" "$RUN/selected/stage5_hn_best.pt"

for pair in "s5_control:$CONTROL" "s5_hn:$HN"; do
  name=${pair%%:*}; model=${pair#*:}
  "$PY" scripts/evaluate_frozen.py --model "$model" --data data/subway_crops/test.yaml \
    --output "$RUN/phase6/$name/test_eval" --imgsz 1280 --device 0 --batch 4 \
    --min-conf .001 --iou .5 > "$RUN/phase6/$name/test_eval.log" 2>&1
done
eval_gate phase6_s5_control "$RUN/phase6/s5_control/test_eval" "$RUN/phase6/s5_control/test_eval.log"
eval_gate phase6_s5_hn "$RUN/phase6/s5_hn/test_eval" "$RUN/phase6/s5_hn/test_eval.log"

# G4: calibrate on calibration only, then apply frozen thresholds to test.
"$PY" scripts/calibrate_thresholds.py --model "$HN" \
  --data data/subway_crops/calibration.yaml --split val --imgsz 1280 --device 0 \
  --min-conf .01 --iou .5 --target-precision .90 --target-recall .80 \
  --output "$RUN/phase6/s5_hn/calibrated_thresholds" \
  > "$RUN/phase6/s5_hn/calibration.log" 2>&1
"$PY" scripts/evaluate_calibrated_thresholds.py --model "$HN" \
  --data data/subway_crops/test.yaml --split val \
  --thresholds "$RUN/phase6/s5_hn/calibrated_thresholds/thresholds.json" \
  --output "$RUN/phase6/s5_hn/calibrated_test" --imgsz 1280 --device 0 \
  --min-conf .01 --iou .5 > "$RUN/phase6/s5_hn/calibrated_test.log" 2>&1
mkdir -p "$RUN/audits/phase6_g4"
"$PY" scripts/stage_gate_714.py --stage phase6_g4_calibration \
  --json "$RUN/phase6/s5_hn/calibrated_thresholds/thresholds.json" \
  --json "$RUN/phase6/s5_hn/calibrated_test/calibrated_test_report.json" \
  --log "$RUN/phase6/s5_hn/calibration.log" \
  --log "$RUN/phase6/s5_hn/calibrated_test.log" \
  --output "$RUN/audits/phase6_g4/gate.json" > "$RUN/audits/phase6_g4/gate.log" 2>&1

"$PY" - <<'PY'
import json
from pathlib import Path
r=Path('output/7.14训练结果/phase6')
def load(p): return json.loads(p.read_text(encoding='utf-8'))
def m(d): return next(float(v) for k,v in d['ultralytics_metrics'].items() if 'mAP50-95' in k)
def macro_p60(d): return sum(x['p_at_r60'] for x in d['per_class'].values())/len(d['per_class'])
s4=load(r/'stage4/test_eval/evaluation.json'); hn=load(r/'s5_hn/test_eval/evaluation.json')
rec4=s4['f2_workpoint']['tp']/max(s4['f2_workpoint']['tp']+s4['f2_workpoint']['fn'],1)
rec5=hn['f2_workpoint']['tp']/max(hn['f2_workpoint']['tp']+hn['f2_workpoint']['fn'],1)
nondecline=sum((hn['per_class'][c].get('ap50') or 0)>=(s4['per_class'][c].get('ap50') or 0) for c in s4['per_class'])
cb4=s4['per_class']['CBHPM']['f2_operating_point']['recall']; cb5=hn['per_class']['CBHPM']['f2_operating_point']['recall']
report={'map_delta':m(hn)-m(s4),'p_at_r60_delta':macro_p60(hn)-macro_p60(s4),
        'recall_delta':rec5-rec4,'cbhpm_recall_delta':cb5-cb4,
        'classes_ap50_non_decreasing':nondecline}
report['passes']=report['p_at_r60_delta']>=.05 and report['recall_delta']>=-.03 and report['map_delta']>=-.005 and report['cbhpm_recall_delta']>=-.03 and nondecline>=5
(r/'stage5_acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if not report['passes']:
 (r/'STOP_CONDITION.txt').write_text('Stage 5 HN failed acceptance; calibration retained for diagnosis.\n')
 raise SystemExit(41)
PY
echo COMPLETE
