#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/projects/Subway_defect_detection_main
RUN='output/7.14训练结果'
PY=/root/miniconda3/bin/python
MANIFEST=data/eval_v20260714/manifest.json

crop_pid=$(cat "$RUN/phase0_crops.pid")
while kill -0 "$crop_pid" 2>/dev/null; do sleep 20; done
grep -q '^COMPLETE$' "$RUN/phase0_crops_parallel.log"

OUT=data/subway_crops_1024_714
mkdir -p "$OUT" "$RUN/crop1024_shards"
common=(--src data/Defect_dataset/images --labels data/Defect_dataset/labels
        --output "$OUT" --crop-size 1024 --negatives-per-image 40
        --eval-negatives-per-image 10 --unlabelled-negatives-per-image 3
        --balance --debiasing --split-manifest "$MANIFEST")
pids=()
for shard in "$RUN"/crop_shards/train_*.txt; do
  name=$(basename "$shard" .txt)
  "$PY" scripts/generate_native_crops.py "${common[@]}" --source-list "$shard" --only-split train \
    > "$RUN/crop1024_shards/${name}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
"$PY" scripts/generate_native_crops.py "${common[@]}" \
  --source-list data/eval_v20260714/calibration_sources.txt --only-split calibration \
  > "$RUN/crop1024_shards/calibration.log" 2>&1 & p1=$!
"$PY" scripts/generate_native_crops.py "${common[@]}" \
  --source-list data/eval_v20260714/test_sources.txt --only-split test \
  > "$RUN/crop1024_shards/test.log" 2>&1 & p2=$!
wait "$p1"; wait "$p2"

"$PY" scripts/create_hn_mining_split.py --dataset-root "$OUT" --manifest "$MANIFEST" --ratio .10 \
  --output-sources data/eval_v20260714/hn_mining_sources.txt > "$RUN/crop1024_hn_split.log" 2>&1
"$PY" - <<'PY'
from pathlib import Path
import yaml
root=Path('data/subway_crops_1024_714').resolve()
base={'path':str(root),'nc':7,'names':['VHBNM','VHBNL','SVHBNM','SVHBNL','SVHTNL','CBHPM','CBVPM']}
for name,train,val in (
 ('subway_crops.yaml','train/images','calibration/images'),
 ('calibration.yaml','calibration/images','calibration/images'),
 ('test.yaml','test/images','test/images'),
 ('hn_mining.yaml','hn_mining/images','hn_mining/images')):
 (root/name).write_text(yaml.safe_dump(dict(base,train=train,val=val),sort_keys=False),encoding='utf-8')
PY
"$PY" scripts/validate_dataset.py --dataset_root "$OUT" \
  --splits train hn_mining calibration test --allow-unlabelled --workers 8 \
  > "$RUN/crop1024_validation.log" 2>&1
"$PY" scripts/generate_scene_augmentations.py \
  --train_images "$OUT/train/images" --train_labels "$OUT/train/labels" \
  --hard-normal --hn-aug-intensity high --hn-per-region 3 --max-samples 800 --workers 4 \
  --hn-output "$OUT/train/images/hard_normals" > "$RUN/crop1024_hard_normal.log" 2>&1
[ "$(find "$OUT/train/images/hard_normals" -maxdepth 1 -type f | wc -l)" -ge 2000 ]
echo COMPLETE
