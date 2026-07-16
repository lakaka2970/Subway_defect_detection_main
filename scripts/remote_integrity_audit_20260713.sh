#!/usr/bin/env bash

set -u

ROOT="/root/autodl-tmp/projects/Subway_defect_detection_main"
cd "$ROOT" || exit 2

echo "AUDIT_ROOT=$PWD"
date '+AUDIT_TIME=%F %T %Z'

echo "[environment]"
/root/miniconda3/bin/python - <<'PY'
import platform

import torch
import ultralytics

print(f"python={platform.python_version()}")
print(f"torch={torch.__version__}")
print(f"ultralytics={ultralytics.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo "[critical_files]"
for file in \
    scripts/train_pipeline.py \
    scripts/multi_source_pretrain_yaml.py \
    scripts/multi_source_dataset_builder.py \
    scripts/calibrate_thresholds.py \
    scripts/validate_dataset.py \
    subway_defect/augmentations/scene.py \
    subway_defect/models/yolo11m-EMA-SimAM.yaml \
    config/train/pretrain/stage1a_public_head.yaml \
    config/train/pretrain/stage1b_public_backbone.yaml \
    config/train/pretrain/stage2_domain_adapt.yaml \
    config/train/pretrain/stage3_main_training.yaml \
    config/train/pretrain/stage4_short_finetune.yaml \
    config/train/pretrain/stage5_hard_negative.yaml; do
    if [[ -f "$file" ]]; then
        sha256sum "$file"
    else
        echo "MISSING $file"
    fi
done

echo "[dataset_yaml_paths]"
for file in config/train/pretrain/stage{1a_public_head,1b_public_backbone,2_domain_adapt,3_main_training,4_short_finetune,5_hard_negative}.yaml; do
    echo "--- $file"
    if [[ -f "$file" ]]; then
        grep -E '^(path|train|val|nc|names):' "$file" || true
    else
        echo "MISSING"
    fi
done

echo "[dataset_integrity]"
/root/miniconda3/bin/python - <<'PY'
from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

DATASETS = {
    "Defect_dataset": {
        "root": Path("data/Defect_dataset"),
        "nc": 7,
        "layouts": [("images/train", "labels/train"), ("images/val", "labels/val")],
    },
    "mixed_pretrain": {
        "root": Path("data/multi_datasets/mixed_pretrain"),
        "nc": 1,
        "layouts": [("images/train", "labels/train"), ("images/val", "labels/val")],
    },
    "subway_crops_1024": {
        "root": Path("data/subway_crops_1024"),
        "nc": 7,
        "layouts": [("train/images", "train/labels"), ("val/images", "val/labels")],
    },
    "subway_crops": {
        "root": Path("data/subway_crops"),
        "nc": 7,
        "layouts": [("train/images", "train/labels"), ("val/images", "val/labels")],
    },
}


def inspect_split(root: Path, image_rel: str, label_rel: str, nc: int) -> dict:
    image_dir = root / image_rel
    label_dir = root / label_rel
    images = sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    labels = sorted(label_dir.rglob("*.txt")) if label_dir.is_dir() else []
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}
    result = {
        "root": str(root),
        "split": image_rel,
        "image_dir_exists": image_dir.is_dir(),
        "label_dir_exists": label_dir.is_dir(),
        "images": len(images),
        "labels": len(labels),
        "missing_labels": len(image_stems - label_stems),
        "orphan_labels": len(label_stems - image_stems),
        "zero_byte_images": sum(p.stat().st_size == 0 for p in images),
        "empty_labels": 0,
        "boxes": 0,
        "invalid_label_rows": 0,
        "decode_errors": 0,
        "class_counts": [0] * nc,
    }

    for label in labels:
        text = label.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            result["empty_labels"] += 1
            continue
        for line in text.splitlines():
            parts = line.split()
            valid = len(parts) == 5
            try:
                values = [float(v) for v in parts]
            except ValueError:
                values = []
                valid = False
            if valid:
                cls = int(values[0])
                valid = values[0] == cls and 0 <= cls < nc
                valid = valid and all(math.isfinite(v) for v in values)
                valid = valid and all(0.0 <= v <= 1.0 for v in values[1:])
                valid = valid and values[3] > 0.0 and values[4] > 0.0
            if not valid:
                result["invalid_label_rows"] += 1
                continue
            result["boxes"] += 1
            result["class_counts"][cls] += 1

    for image in images:
        try:
            with Image.open(image) as handle:
                handle.verify()
        except Exception:
            result["decode_errors"] += 1
    return result


for dataset_name, config in DATASETS.items():
    root = config["root"]
    if not root.is_dir():
        print(json.dumps({"dataset": dataset_name, "missing_root": str(root)}, ensure_ascii=False))
        continue
    for image_rel, label_rel in config["layouts"]:
        report = inspect_split(root, image_rel, label_rel, config["nc"])
        report["dataset"] = dataset_name
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
PY

echo "[public_source_dirs]"
for dir in \
    data/multi_datasets/public/{gc10_det,neu_det,kolektor_sdd2,rsdds} \
    data/multi_datasets/mixed_pretrain; do
    if [[ -d "$dir" ]]; then
        printf '%s files=' "$dir"
        find "$dir" -type f | wc -l
    else
        echo "MISSING $dir"
    fi
done

echo "[mixed_pretrain_link_targets]"
/root/miniconda3/bin/python - <<'PY'
from collections import Counter
from pathlib import Path

root = Path("data/multi_datasets/mixed_pretrain")
links = [p for p in root.rglob("*") if p.is_symlink()]
broken = [p for p in links if not p.exists()]
print(f"symlinks={len(links)} broken={len(broken)}")
targets = Counter()
for link in links:
    try:
        target = link.resolve(strict=True)
    except FileNotFoundError:
        continue
    parts = target.parts
    marker = "multi_datasets"
    if marker in parts:
        index = parts.index(marker)
        key = "/".join(parts[index:index + 4])
    else:
        key = str(target.parent)
    targets[key] += 1
for target, count in targets.most_common():
    print(f"{count} {target}")
for link in links[:12]:
    try:
        print(f"LINK {link} -> {link.resolve(strict=True)}")
    except FileNotFoundError:
        print(f"BROKEN {link}")
PY

echo "[multi_dataset_tree]"
find data/multi_datasets -maxdepth 3 -type d -print | sort
find data/multi_datasets -maxdepth 2 -type f -print | sort

echo "[dataset_timestamps]"
stat -c '%y %n' \
    data/multi_datasets/mixed_pretrain \
    config/train/pretrain/stage1a_public_head.yaml \
    output/20260709_full_multisource/stage_1a/results.csv \
    output/logs/20260709_full_stage1a_to_5.log

echo "[stage5_post_training_artifacts]"
for artifact in \
    output/20260709_full_multisource/stage_5/calibrated_thresholds/thresholds.json \
    output/20260709_full_multisource/stage_5/calibrated_thresholds/pr_curves.json; do
    if [[ -s "$artifact" ]]; then
        stat -c "OK %s %n" "$artifact"
    else
        echo "MISSING_OR_EMPTY $artifact"
    fi
done
find data output/20260709_full_multisource -maxdepth 5 \
    -iname '*hard*negative*' -print 2>/dev/null | sort

echo "[log_dataset_scans]"
grep -aE 'Scanning|images, [0-9]+ backgrounds|train: .*images|val: .*images' \
    output/logs/20260709_full_stage1a_to_5.log | head -80 || true

echo "[training_outputs]"
for run in output/20260704_121435 output/20260709_full_multisource; do
    if [[ -d "$run" ]]; then
        printf '%s files=' "$run"
        find "$run" -type f | wc -l
        du -sh "$run"
    else
        echo "MISSING $run"
    fi
done

for stage in stage_1a stage_1b stage_2 stage_3 stage_4 stage_5; do
    dir="output/20260709_full_multisource/$stage"
    for file in args.yaml results.csv weights/best.pt weights/last.pt; do
        if [[ -s "$dir/$file" ]]; then
            stat -c "OK %s %n" "$dir/$file"
        else
            echo "MISSING_OR_EMPTY $dir/$file"
        fi
    done
done

echo "[published_weight_matches]"
declare -A PUBLISHED=(
    [stage_1a]="weights/stage1a_public_head.pt"
    [stage_1b]="weights/stage1b_public_backbone.pt"
    [stage_2]="weights/stage2_domain_adapt.pt"
    [stage_3]="weights/stage3_main.pt"
    [stage_4]="weights/stage4_best_finetune.pt"
    [stage_5]="weights/stage5_calibrated.pt"
)
for stage in stage_1a stage_1b stage_2 stage_3 stage_4 stage_5; do
    source="output/20260709_full_multisource/$stage/weights/best.pt"
    target="${PUBLISHED[$stage]}"
    if [[ -f "$source" && -f "$target" ]]; then
        source_hash=$(sha256sum "$source" | cut -d' ' -f1)
        target_hash=$(sha256sum "$target" | cut -d' ' -f1)
        if [[ "$source_hash" == "$target_hash" ]]; then
            echo "MATCH $stage $source_hash"
        else
            echo "MISMATCH $stage source=$source_hash published=$target_hash"
        fi
    else
        echo "MISSING_WEIGHT_PAIR $stage source=$source target=$target"
    fi
done

echo "[logs_and_calibration]"
for file in \
    output/logs/20260709_full_stage1a_to_5.log \
    output/logs/20260709_stage3_to_5_after_nan_fix.log \
    output/20260709_full_multisource/stage_5/calibrated_thresholds/thresholds.json \
    output/20260709_full_multisource/stage_5/calibrated_thresholds/pr_curves.json; do
    if [[ -s "$file" ]]; then
        stat -c "OK %s %n" "$file"
    else
        echo "MISSING_OR_EMPTY $file"
    fi
done

echo "[git_state]"
git rev-parse HEAD 2>/dev/null || echo "NO_GIT_HEAD"
git status --short 2>/dev/null | head -100 || true
