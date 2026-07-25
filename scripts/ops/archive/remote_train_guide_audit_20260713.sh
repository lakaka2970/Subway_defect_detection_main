#!/usr/bin/env bash

set -u

ROOT="/root/autodl-tmp/projects/Subway_defect_detection_main"
PYTHON="/root/miniconda3/bin/python"
RUN="$ROOT/output/20260709_full_multisource"

cd "$ROOT" || exit 2

pass() { printf 'PASS %s\n' "$*"; }
missing() { printf 'MISSING %s\n' "$*"; }
info() { printf 'INFO %s\n' "$*"; }

echo "TRAIN_GUIDE_AUDIT_ROOT=$PWD"
date '+TRAIN_GUIDE_AUDIT_TIME=%F %T %Z'

echo "[runtime]"
if nvidia-smi -L 2>/dev/null | grep -q 'GPU'; then
    nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
        --format=csv,noheader
    pass "CUDA device is visible to nvidia-smi"
else
    missing "CUDA device is not visible to nvidia-smi"
fi
"$PYTHON" - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"torch_cuda_available={torch.cuda.is_available()}")
print(f"torch_cuda_count={torch.cuda.device_count()}")
PY
df -h "$ROOT" | tail -n 1
ps -eo pid,etime,cmd | grep -E 'train_pipeline|collect_hard|calibrate_threshold|audit_labels' \
    | grep -v grep || true

echo "[A_environment]"
if "$PYTHON" -c 'from subway_yolo import YOLO; from subway_defect.modules.EMA import EMA; from subway_defect.modules.SimAM import SimAM' 2>/dev/null; then
    pass "editable package imports"
else
    missing "project imports"
fi
info "branch=$(git branch --show-current) commit=$(git rev-parse HEAD)"
git remote -v | head -n 4

echo "[B_data]"
for path in \
    data/Defect_dataset \
    data/multi_datasets/public/gc10_det \
    data/multi_datasets/public/neu_det \
    data/multi_datasets/public/kolektor_sdd2 \
    data/multi_datasets/public/rsdds \
    data/multi_datasets/mixed_pretrain \
    data/subway_crops \
    data/subway_crops_1024; do
    if [[ -d "$path" ]]; then
        pass "$path"
    else
        missing "$path"
    fi
done
for dataset in data/Defect_dataset data/multi_datasets/mixed_pretrain data/subway_crops data/subway_crops_1024; do
    echo "--- validate $dataset"
    "$PYTHON" scripts/validate_dataset.py --dataset "$dataset" || true
done

echo "[C_optional_augmentation]"
for path in \
    data/subway_crops_cp \
    data/Defect_dataset/augmentation_summary.json \
    data/subway_crops/augmentation_summary.json; do
    if [[ -e "$path" ]]; then
        pass "$path"
    else
        info "not found (optional): $path"
    fi
done
find data/Defect_dataset data/subway_crops -maxdepth 4 -type f \
    \( -iname '*scene*summary*' -o -iname '*copy*paste*summary*' -o -iname '*synthetic*summary*' \) \
    -print 2>/dev/null | sort

echo "[D_configs]"
for stage in \
    stage1a_public_head \
    stage1b_public_backbone \
    stage2_domain_adapt \
    stage3_main_training \
    stage4_short_finetune \
    stage5_hard_negative; do
    path="config/train/pretrain/${stage}.yaml"
    if [[ -s "$path" ]]; then
        pass "$path"
    else
        missing "$path"
    fi
done

echo "[E_full_training]"
for stage in stage_1a stage_1b stage_2 stage_3 stage_4 stage_5; do
    failures=0
    for file in args.yaml results.csv weights/best.pt weights/last.pt; do
        [[ -s "$RUN/$stage/$file" ]] || failures=$((failures + 1))
    done
    if [[ "$failures" -eq 0 ]]; then
        rows=$(wc -l < "$RUN/$stage/results.csv")
        pass "$stage artifacts results_rows=$rows"
    else
        missing "$stage missing_artifacts=$failures"
    fi
done

echo "[G_stage5_complete_flow]"
for path in \
    data/hard_negatives/summary.json \
    output/stage5_hnm/stage_5/weights/best.pt \
    output/stage5_hnm/calibrated_thresholds/thresholds.json \
    output/stage5_hnm/calibrated_thresholds/per_class_report.csv \
    output/stage5_hnm/calibrated_thresholds/summary.txt; do
    if [[ -s "$path" ]]; then
        pass "$path"
    else
        missing "$path"
    fi
done
find data/hard_negatives -maxdepth 2 -type f -print 2>/dev/null | head -n 30

echo "[H_label_audit]"
for path in \
    data/label_audit/audit_summary.json \
    data/label_audit/audit_report.txt; do
    if [[ -s "$path" ]]; then
        pass "$path"
    else
        missing "$path"
    fi
done

echo "[cli_signatures]"
for script in \
    validate_dataset.py \
    collect_hard_negatives.py \
    calibrate_thresholds.py \
    audit_labels.py \
    generate_scene_augmentations.py \
    generate_defect_copy_paste.py; do
    echo "--- scripts/$script --help"
    "$PYTHON" "scripts/$script" --help 2>&1 | sed -n '1,80p'
done
