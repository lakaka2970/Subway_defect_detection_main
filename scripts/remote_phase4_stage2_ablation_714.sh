#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main

RUN='output/7.14训练结果'
PY=/root/miniconda3/bin/python
CFG=config/train/pretrain/stage2_domain_adapt.yaml
BACKUP="$RUN/generated_configs/stage2_domain_adapt.original.yaml"
cp "$CFG" "$BACKUP"
trap 'cp "$BACKUP" "$CFG"' EXIT

after_pid=$(cat "$RUN/after_crops.pid")
crop1024_pid=$(cat "$RUN/crop1024.pid")
while kill -0 "$after_pid" 2>/dev/null || kill -0 "$crop1024_pid" 2>/dev/null; do sleep 30; done
grep -q '^COMPLETE$' "$RUN/after_crops.log"
grep -q '^COMPLETE$' "$RUN/crop1024.log"

mkdir -p "$RUN/phase4/baseline_new_s2"
"$PY" scripts/evaluate_frozen.py \
  --model output/20260709_full_multisource/stage_2/weights/best.pt \
  --data data/subway_crops/test.yaml --output "$RUN/phase4/baseline_new_s2" \
  --imgsz 1280 --device 0 --batch 4 --min-conf .001 --iou .5 \
  > "$RUN/phase4/baseline_new_s2/run.log" 2>&1

run_variant() {
  local name=$1 dataset=$2
  local out="$RUN/phase4/$name"
  mkdir -p "$out"
  DATASET="$dataset" "$PY" - <<'PY'
import os, yaml
from pathlib import Path
p=Path('config/train/pretrain/stage2_domain_adapt.yaml')
d=yaml.safe_load(p.read_text(encoding='utf-8'))
d['path']=os.environ['DATASET']
d['train']='train/images'; d['val']='calibration/images'
p.write_text(yaml.safe_dump(d, sort_keys=False), encoding='utf-8')
PY
  cp "$CFG" "$out/stage2_config.yaml"
  "$PY" scripts/train_pipeline.py --stages 2 --model yolo11m-EMA-SimAM \
    --output "$out/train" --device 0 --batch 12 --workers 4 \
    --pretrained output/20260709_full_multisource/stage_1b/weights/best.pt \
    > "$out/train.log" 2>&1
  test -s "$out/train/stage_2/weights/best.pt"
  "$PY" scripts/evaluate_frozen.py \
    --model "$out/train/stage_2/weights/best.pt" --data data/subway_crops/test.yaml \
    --output "$out/test_eval" --imgsz 1280 --device 0 --batch 4 --min-conf .001 --iou .5 \
    > "$out/test_eval.log" 2>&1
}

run_variant s2_a data/subway_crops_1024_714
run_variant s2_b data/subway_crops

"$PY" - <<'PY'
import json, shutil
from pathlib import Path
root=Path('output/7.14训练结果/phase4')
def load(name): return json.loads((root/name/'test_eval/evaluation.json').read_text(encoding='utf-8'))
baseline=json.loads((root/'baseline_new_s2/evaluation.json').read_text(encoding='utf-8'))
candidates={n:load(n) for n in ('s2_a','s2_b')}
def map95(d):
 for k,v in d['ultralytics_metrics'].items():
  if 'mAP50-95' in k: return float(v)
 raise KeyError('mAP50-95 metric missing')
def class_ap(d): return {k:v.get('ap50_95') for k,v in d['per_class'].items()}
base_map=map95(baseline); base_ap=class_ap(baseline)
rows={}
for name,data in candidates.items():
 aps=class_ap(data)
 nondecline=sum(aps[k] is not None and base_ap[k] is not None and aps[k]>=base_ap[k] for k in base_ap)
 rows[name]={'map50_95':map95(data),'improvement':map95(data)-base_map,
             'classes_non_decreasing':nondecline,
             'passes':map95(data)-base_map>=.010 and nondecline>=5}
passing=[n for n,r in rows.items() if r['passes']]
report={'baseline_map50_95':base_map,'candidates':rows,'selected':None}
if passing:
 selected=max(passing,key=lambda n:rows[n]['map50_95'])
 report['selected']=selected
 dst=Path('output/7.14训练结果/selected'); dst.mkdir(parents=True,exist_ok=True)
 shutil.copy2(root/selected/'train/stage_2/weights/best.pt',dst/'stage2_best.pt')
(root/'selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
if not passing:
 (root/'STOP_CONDITION.txt').write_text('No Stage 2 candidate passed +0.010 mAP50-95 and 5/7 class non-decline.\n')
 raise SystemExit(20)
PY
echo COMPLETE
