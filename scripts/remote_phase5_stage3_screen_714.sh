#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main
RUN='output/7.14训练结果'
PY=/root/miniconda3/bin/python
CFG=config/train/pretrain/stage3_main_training.yaml
BACKUP="$RUN/generated_configs/stage3_main_training.original.yaml"

phase4_pid=$(cat "$RUN/phase4.pid")
while kill -0 "$phase4_pid" 2>/dev/null; do sleep 30; done
grep -q '^COMPLETE$' "$RUN/phase4.log"
test -s "$RUN/selected/stage2_best.pt"
cp "$CFG" "$BACKUP"
trap 'cp "$BACKUP" "$CFG"' EXIT

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
  local stage=$1 root=$2 audit="$RUN/audits/$1"
  mkdir -p "$audit"
  "$PY" scripts/stage_gate_714.py --stage "$stage-train" \
    --weights "$root/weights/best.pt" --results "$root/results.csv" \
    --log "${root%/stage_3}/../train.log" --output "$audit/train.json" \
    > "$audit/train_gate.log" 2>&1
}

eval_gate() {
  local stage=$1 root=$2 log=$3 audit="$RUN/audits/$1"
  mkdir -p "$audit"
  "$PY" scripts/stage_gate_714.py --stage "$stage-eval" \
    --json "$root/evaluation.json" --log "$log" --output "$audit/eval.json" \
    > "$audit/eval_gate.log" 2>&1
}

preflight_gate phase5_preflight

# Validate the selected Stage-2 artefact before it is allowed to initialise
# Stage 3. If S2-B is rejected by the integrity gate, use the already-passing
# S2-A candidate and record the fail-closed fallback explicitly.
stage2_name=$("$PY" - <<'PY'
import json
from pathlib import Path
print(json.loads(Path('output/7.14训练结果/phase4/selection.json').read_text())['selected'])
PY
)
mkdir -p "$RUN/audits/selected_stage2"
stage2_nms_args=()
if [ "$stage2_name" = s2_a ]; then stage2_nms_args=(--allow-nms-warning); fi
if ! "$PY" scripts/stage_gate_714.py --stage selected_stage2 \
  --weights "$RUN/selected/stage2_best.pt" \
  --results "$RUN/phase4/$stage2_name/train/stage_2/results.csv" \
  --json "$RUN/phase4/$stage2_name/test_eval/evaluation.json" \
  --log "$RUN/phase4/$stage2_name/train.log" \
  "${stage2_nms_args[@]}" \
  --output "$RUN/audits/selected_stage2/gate.json" \
  > "$RUN/audits/selected_stage2/gate.log" 2>&1; then
  if [ "$stage2_name" != s2_b ]; then exit 29; fi
  "$PY" scripts/stage_gate_714.py --stage fallback_stage2_s2_a \
    --weights "$RUN/phase4/s2_a/train/stage_2/weights/best.pt" \
    --results "$RUN/phase4/s2_a/train/stage_2/results.csv" \
    --json "$RUN/phase4/s2_a/test_eval/evaluation.json" \
    --log "$RUN/phase4/s2_a/train.log" \
    --allow-nms-warning \
    --output "$RUN/audits/selected_stage2/fallback_s2_a.json" \
    > "$RUN/audits/selected_stage2/fallback_s2_a.log" 2>&1
  cp "$RUN/phase4/s2_a/train/stage_2/weights/best.pt" "$RUN/selected/stage2_best.pt"
  printf 's2_b failed integrity gate; s2_a substituted after passing gate.\n' \
    > "$RUN/phase4/S2_B_REJECTED_BY_GATE.txt"
fi

write_config() {
  local variant=$1 epochs=$2 seed=$3
  VARIANT="$variant" EPOCHS="$epochs" SEED="$seed" "$PY" - <<'PY'
import os, yaml
from pathlib import Path
p=Path('config/train/pretrain/stage3_main_training.yaml')
d=yaml.safe_load(Path('output/7.14训练结果/generated_configs/stage3_main_training.original.yaml').read_text(encoding='utf-8'))
d.update(dict(path='data/subway_crops',train='train/images',val='calibration/images',
 imgsz=1280,batch=4,epochs=int(os.environ['EPOCHS']),seed=int(os.environ['SEED']),amp=False,conf=.01,
 lr0=2e-4,warmup_bias_lr=2e-4,mosaic=.05,copy_paste=0.0,auto_augment='none',
 hsv_s=.3,hsv_v=.3,degrees=2.0,translate=.05,scale=.2,multi_scale=0.0,
 fl_gamma=2.0,class_weights=[]))
v=os.environ['VARIANT']
if v=='s3_b': d['lr0']=3e-4; d['warmup_bias_lr']=3e-4
elif v=='s3_d': d['multi_scale']=.5
elif v=='s3_h': d['fl_gamma']=2.5
elif v=='s3_f': d['class_weights']=[1.0,1.0,2.0,1.3,1.0,1.5,1.5]
p.write_text(yaml.safe_dump(d,sort_keys=False),encoding='utf-8')
PY
}

for variant in s3_a s3_b s3_d s3_h s3_f; do
  out="$RUN/phase5/screen/$variant"
  mkdir -p "$out"
  preflight_gate "phase5_screen_${variant}_preflight"
  write_config "$variant" 30 0
  cp "$CFG" "$out/config.yaml"
  "$PY" scripts/train_pipeline.py --stages 3 --model yolo11m-EMA-SimAM \
    --output "$out/train" --device 0 --batch 4 --workers 4 \
    --pretrained "$RUN/selected/stage2_best.pt" > "$out/train.log" 2>&1
  test -s "$out/train/stage_3/weights/best.pt"
  if grep -Eiq '(^|[,[:space:]])(nan|inf)([,[:space:]]|$)' "$out/train/stage_3/results.csv"; then
    echo "NaN/Inf detected in $variant" > "$out/STOP_CONDITION.txt"
    exit 30
  fi
  train_gate "phase5_screen_${variant}" "$out/train/stage_3"
  "$PY" scripts/evaluate_frozen.py --model "$out/train/stage_3/weights/best.pt" \
    --data data/subway_crops/calibration.yaml --output "$out/calibration_eval" \
    --imgsz 1280 --device 0 --batch 4 --min-conf .001 --iou .5 \
    > "$out/calibration_eval.log" 2>&1
  eval_gate "phase5_screen_${variant}" "$out/calibration_eval" "$out/calibration_eval.log"
done

"$PY" - <<'PY'
import json, shutil
from pathlib import Path
r=Path('output/7.14训练结果/phase5/screen')
def load(n): return json.loads((r/n/'calibration_eval/evaluation.json').read_text(encoding='utf-8'))
def m(d):
 return next(float(v) for k,v in d['ultralytics_metrics'].items() if 'mAP50-95' in k)
base=load('s3_a'); bm=m(base)
base_r={c:base['per_class'][c]['f2_operating_point']['recall'] for c in ('SVHBNM','CBHPM')}
rows={}
for n in ('s3_a','s3_b','s3_d','s3_h','s3_f'):
 d=load(n); recalls={c:d['per_class'][c]['f2_operating_point']['recall'] for c in base_r}
 rows[n]={'map50_95':m(d),'delta_vs_a':m(d)-bm,'p0_recall':recalls,
          'passes':m(d)>=bm-.005 and all(recalls[c]>=base_r[c]-.05 for c in base_r)}
passing=[n for n,x in rows.items() if x['passes']]
selected=max(passing,key=lambda n:rows[n]['map50_95'])
report={'baseline':bm,'variants':rows,'selected':selected}
(r/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(Path('output/7.14训练结果/selected/stage3_variant.txt')).write_text(selected+'\n')
print(json.dumps(report,indent=2))
PY

selected=$(tr -d '\r\n' < "$RUN/selected/stage3_variant.txt")
for seed in 0 1 2; do
  out="$RUN/phase5/full/${selected}_seed${seed}"
  mkdir -p "$out"
  preflight_gate "phase5_full_${selected}_seed${seed}_preflight"
  write_config "$selected" 80 "$seed"
  cp "$CFG" "$out/config.yaml"
  "$PY" scripts/train_pipeline.py --stages 3 --model yolo11m-EMA-SimAM \
    --output "$out/train" --device 0 --batch 4 --workers 4 \
    --pretrained "$RUN/selected/stage2_best.pt" > "$out/train.log" 2>&1
  test -s "$out/train/stage_3/weights/best.pt"
  train_gate "phase5_full_${selected}_seed${seed}" "$out/train/stage_3"
  "$PY" scripts/evaluate_frozen.py --model "$out/train/stage_3/weights/best.pt" \
    --data data/subway_crops/calibration.yaml --output "$out/calibration_eval" \
    --imgsz 1280 --device 0 --batch 4 --min-conf .001 --iou .5 \
    > "$out/calibration_eval.log" 2>&1
  eval_gate "phase5_full_${selected}_seed${seed}" "$out/calibration_eval" "$out/calibration_eval.log"
done

"$PY" - <<'PY'
import json, shutil, statistics
from pathlib import Path
run=Path('output/7.14训练结果'); variant=(run/'selected/stage3_variant.txt').read_text().strip()
rows={}
for seed in (0,1,2):
 p=run/f'phase5/full/{variant}_seed{seed}'
 d=json.loads((p/'calibration_eval/evaluation.json').read_text(encoding='utf-8'))
 score=next(float(v) for k,v in d['ultralytics_metrics'].items() if 'mAP50-95' in k)
 rows[seed]={'map50_95':score,'path':str(p/'train/stage_3/weights/best.pt')}
values=[x['map50_95'] for x in rows.values()]
mean=statistics.mean(values); rel_std=statistics.pstdev(values)/max(abs(mean),1e-12)
report={'variant':variant,'seeds':rows,'mean':mean,'relative_std':rel_std,'passes_stability':rel_std<.05}
(run/'phase5/full/seed_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if rel_std>=.05:
 (run/'phase5/full/STOP_CONDITION.txt').write_text('Three-seed relative std is >=5%.\n')
 raise SystemExit(31)
best=max(rows,key=lambda s:rows[s]['map50_95'])
report['selected_seed']=best
shutil.copy2(rows[best]['path'],run/'selected/stage3_best.pt')
print(json.dumps(report,indent=2))
PY

mkdir -p "$RUN/phase5/test"
preflight_gate phase5_frozen_test_preflight
"$PY" scripts/evaluate_frozen.py --model "$RUN/selected/stage3_best.pt" \
  --data data/subway_crops/test.yaml --output "$RUN/phase5/test" \
  --imgsz 1280 --device 0 --batch 4 --min-conf .001 --iou .5 \
  > "$RUN/phase5/test.log" 2>&1
eval_gate phase5_frozen_test "$RUN/phase5/test" "$RUN/phase5/test.log"

"$PY" - <<'PY'
import json
from pathlib import Path
run=Path('output/7.14训练结果')
old=json.loads((run/'phase1/new_s3/evaluation.json').read_text(encoding='utf-8'))
new=json.loads((run/'phase5/test/evaluation.json').read_text(encoding='utf-8'))
def m(d): return next(float(v) for k,v in d['ultralytics_metrics'].items() if 'mAP50-95' in k)
def p60(d): return sum(v['p_at_r60'] for v in d['per_class'].values())/len(d['per_class'])
report={'old_map50_95':m(old),'new_map50_95':m(new),'map_delta':m(new)-m(old),
        'old_p_at_r60':p60(old),'new_p_at_r60':p60(new),'p_at_r60_delta':p60(new)-p60(old),
        'svhbnm_ap_old':old['per_class']['SVHBNM']['ap50_95'],
        'svhbnm_ap_new':new['per_class']['SVHBNM']['ap50_95']}
report['passes']=report['map_delta']>=.010 and report['p_at_r60_delta']>=.05 and report['svhbnm_ap_new']>=report['svhbnm_ap_old']
(run/'phase5/acceptance.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
if not report['passes']:
 (run/'phase5/STOP_CONDITION.txt').write_text('Stage 3 full run failed frozen-test acceptance.\n')
 raise SystemExit(32)
PY
echo COMPLETE
