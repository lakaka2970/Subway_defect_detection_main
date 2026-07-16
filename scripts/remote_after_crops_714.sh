#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main

RUN='output/7.14训练结果'
PY=/root/miniconda3/bin/python
crop_pid=$(cat "$RUN/phase0_crops.pid")
while kill -0 "$crop_pid" 2>/dev/null; do sleep 20; done
grep -q '^COMPLETE$' "$RUN/phase0_crops_parallel.log"
grep -q 'VALIDATION PASSED' "$RUN/phase0_validation.log"

if [ "$(find data/subway_crops/hn_mining/images -maxdepth 1 -type f 2>/dev/null | wc -l)" -eq 0 ]; then
  "$PY" scripts/create_hn_mining_split.py \
    --dataset-root data/subway_crops \
    --manifest data/eval_v20260714/manifest.json \
    --ratio 0.10 \
    > "$RUN/phase0_hn_split.log" 2>&1
fi

"$PY" - <<'PY'
from pathlib import Path
import yaml
root = Path('data/subway_crops').resolve()
base = {'path': str(root), 'nc': 7,
        'names': ['VHBNM','VHBNL','SVHBNM','SVHBNL','SVHTNL','CBHPM','CBVPM']}
for name, train, val in (
    ('subway_crops.yaml', 'train/images', 'calibration/images'),
    ('calibration.yaml', 'calibration/images', 'calibration/images'),
    ('test.yaml', 'test/images', 'test/images'),
    ('hn_mining.yaml', 'hn_mining/images', 'hn_mining/images'),
):
    (root / name).write_text(yaml.safe_dump(dict(base, train=train, val=val), sort_keys=False),
                             encoding='utf-8')
PY

"$PY" scripts/validate_dataset.py --dataset_root data/subway_crops \
  --splits train hn_mining calibration test --allow-unlabelled --workers 8 \
  > "$RUN/phase0_validation_after_hn_split.log" 2>&1
cp data/eval_v20260714/manifest.json "$RUN/eval_manifest.json"
cp data/eval_v20260714/manifest.json "$RUN/dataset_manifest.json"

hn_count=$(find data/subway_crops/train/images/hard_normals -maxdepth 1 -type f | wc -l)
if [ "$hn_count" -lt 2000 ]; then
  "$PY" scripts/generate_scene_augmentations.py \
    --train_images data/subway_crops/train/images \
    --train_labels data/subway_crops/train/labels \
    --hard-normal --hn-aug-intensity high --hn-per-region 3 \
    --max-samples 800 --workers 4 \
    --hn-output data/subway_crops/train/images/hard_normals \
    > "$RUN/phase35_hard_normal.log" 2>&1
  hn_count=$(find data/subway_crops/train/images/hard_normals -maxdepth 1 -type f | wc -l)
fi
[ "$hn_count" -ge 2000 ]
find data/subway_crops/train/labels -name 'hn_*.txt' -print -quit | grep -q . && {
  echo 'Hard Normal labels unexpectedly exist' >&2; exit 9;
} || true

if [ ! -s "$RUN/phase1/comparison.json" ]; then
  bash scripts/remote_phase1_eval_714.sh > "$RUN/phase1.log" 2>&1
fi
bash scripts/remote_phase2_calibrate_714.sh > "$RUN/phase2.log" 2>&1
echo COMPLETE
