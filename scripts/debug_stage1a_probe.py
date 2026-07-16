#!/usr/bin/env python3
"""Minimal Stage 1A training probe."""

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subway_defect.train.configs import _safe_load_yaml
from subway_yolo import YOLO


cfg_path = Path("config/train/pretrain/stage1a_public_head.yaml")
cfg = _safe_load_yaml(cfg_path)
for key in ("path", "train", "val", "nc", "names", "test"):
    cfg.pop(key, None)
cfg.update(
    data=str(cfg_path.resolve()),
    epochs=1,
    batch=2,
    workers=0,
    device="0",
    project="output/debug_stage1a",
    name="probe",
    pretrained=False,
    imgsz=320,
    cache=False,
    plots=False,
    verbose=True,
)

print("building model", flush=True)
t0 = time.time()
model = YOLO("subway_defect/models/yolo11m-EMA-SimAM.yaml")
print(f"built {time.time() - t0:.1f}s", flush=True)

print("training", flush=True)
t0 = time.time()
model.train(**cfg)
print(f"trained {time.time() - t0:.1f}s", flush=True)
