#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/projects/Subway_defect_detection_main
RUN='output/7.14训练结果'
PY=/root/miniconda3/bin/python
MANIFEST=data/eval_v20260714/manifest.json

mkdir -p "$RUN/crop_shards"
split -n l/4 -d --additional-suffix=.txt \
  data/eval_v20260714/train_sources.txt "$RUN/crop_shards/train_"

common=(
  --src data/Defect_dataset/images
  --labels data/Defect_dataset/labels
  --output data/subway_crops
  --crop-size 1280
  --negatives-per-image 40
  --eval-negatives-per-image 10
  --unlabelled-negatives-per-image 3
  --balance --debiasing
  --split-manifest "$MANIFEST"
)

pids=()
for shard in "$RUN"/crop_shards/train_*.txt; do
  name=$(basename "$shard" .txt)
  "$PY" scripts/generate_native_crops.py "${common[@]}" \
    --source-list "$shard" --only-split train \
    > "$RUN/crop_shards/${name}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

"$PY" scripts/generate_native_crops.py "${common[@]}" \
  --source-list data/eval_v20260714/calibration_sources.txt --only-split calibration \
  > "$RUN/crop_shards/calibration.log" 2>&1 &
cal_pid=$!
"$PY" scripts/generate_native_crops.py "${common[@]}" \
  --source-list data/eval_v20260714/test_sources.txt --only-split test \
  > "$RUN/crop_shards/test.log" 2>&1 &
test_pid=$!
wait "$cal_pid"
wait "$test_pid"

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
):
    cfg = dict(base, train=train, val=val)
    (root / name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
PY

"$PY" scripts/validate_dataset.py --dataset_root data/subway_crops \
  --splits train calibration test --allow-unlabelled --workers 8 \
  > "$RUN/phase0_validation.log" 2>&1

find data/subway_crops/train/images -maxdepth 1 -type f | sort > "$RUN/train_crop_images.txt"
find data/subway_crops/train/labels -maxdepth 1 -type f | sort > "$RUN/train_crop_labels.txt"
echo COMPLETE
