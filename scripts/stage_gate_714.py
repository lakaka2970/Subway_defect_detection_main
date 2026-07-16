#!/usr/bin/env python3
"""Fail-closed integrity gate between the 2026-07-14 training stages."""

from __future__ import annotations

import argparse
import csv
import json
import math
import py_compile
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FATAL_LOG_PATTERNS = (
    re.compile(r"traceback \(most recent call last\)", re.I),
    re.compile(r"\b(?:runtime|assertion|value|key|index)error\b", re.I),
    re.compile(r"segmentation fault", re.I),
)

CRITICAL_CODE = (
    "scripts/train_pipeline.py",
    "scripts/evaluate_frozen.py",
    "scripts/evaluate_calibrated_thresholds.py",
    "scripts/calibrate_thresholds.py",
    "scripts/collect_hard_negatives.py",
    "scripts/validate_dataset.py",
    "subway_yolo/utils/loss.py",
)


def require_file(path: Path, minimum_size: int = 1) -> None:
    if not path.is_file() or path.stat().st_size < minimum_size:
        raise RuntimeError(f"missing or undersized file: {path}")


def check_code() -> dict:
    checked = []
    for name in CRITICAL_CODE:
        path = Path(name)
        require_file(path)
        py_compile.compile(str(path), doraise=True)
        checked.append(name)

    # Exercise the custom focal loss with saturated logits. This catches the
    # AMP NaN failure that appeared during the 1280 Stage-2 validation.
    from torch import nn
    from subway_yolo.utils.loss import FocalLoss

    focal = FocalLoss(
        nn.BCEWithLogitsLoss(reduction="none"),
        gamma=2.0,
        alpha=0.25,
        class_weights=[1.0, 2.0, 3.0],
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for dtype in ((torch.float16, torch.float32) if device == "cuda" else (torch.float32,)):
        logits = torch.tensor([[[1000.0, -1000.0, 0.0]]], device=device, dtype=dtype, requires_grad=True)
        target = torch.tensor([[[1.0, 0.0, 1.0]]], device=device, dtype=dtype)
        with torch.autocast(device_type=device, enabled=device == "cuda"):
            loss = focal(logits, target).sum()
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(logits.grad).all():
            raise RuntimeError(f"non-finite focal loss/gradient for {dtype}")
    return {"compiled": checked, "focal_numeric_test": "passed"}


def check_weights(path: Path) -> dict:
    require_file(path, 1_000_000)
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise RuntimeError(f"unexpected checkpoint object in {path}: {type(obj)!r}")
    model = obj.get("ema") or obj.get("model")
    if model is None:
        raise RuntimeError(f"checkpoint has no model/ema payload: {path}")
    state = model.float().state_dict() if hasattr(model, "state_dict") else model
    tensor_count = 0
    for key, value in state.items():
        if torch.is_tensor(value) and value.is_floating_point():
            tensor_count += 1
            if not torch.isfinite(value).all():
                raise RuntimeError(f"non-finite checkpoint tensor: {path}:{key}")
    if tensor_count == 0:
        raise RuntimeError(f"checkpoint has no floating tensors: {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "floating_tensors": tensor_count}


def check_results(path: Path) -> dict:
    require_file(path)
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"results CSV has no epochs: {path}")
    for row_number, row in enumerate(rows, start=1):
        for key, raw in row.items():
            if raw is None or not raw.strip():
                raise RuntimeError(f"blank result at {path}:{row_number}:{key}")
            try:
                value = float(raw)
            except ValueError as exc:
                raise RuntimeError(f"non-numeric result at {path}:{row_number}:{key}={raw!r}") from exc
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite result at {path}:{row_number}:{key}={raw!r}")
    return {"path": str(path), "epochs": len(rows), "last_epoch": int(float(rows[-1]["epoch"]))}


def walk_finite(value, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            walk_finite(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            walk_finite(child, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"non-finite JSON value at {location}: {value}")


def check_json(path: Path) -> dict:
    require_file(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    walk_finite(data)
    return {"path": str(path), "top_level_keys": sorted(data) if isinstance(data, dict) else []}


def check_log(path: Path, allow_nms_warning: bool = False) -> dict:
    require_file(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in FATAL_LOG_PATTERNS:
        match = pattern.search(text)
        if match:
            raise RuntimeError(f"fatal log pattern {match.group(0)!r} in {path}")
    nms_warnings = len(re.findall(r"NMS time limit .* exceeded", text, flags=re.I))
    if nms_warnings and not allow_nms_warning:
        raise RuntimeError(f"{nms_warnings} NMS time-limit warnings in {path}")
    oom_warnings = len(re.findall(r"cuda out of memory", text, flags=re.I))
    recoverable_oom = len(
        re.findall(r"CUDA out of memory with batch=\d+\. Reducing to batch=\d+ and retrying", text, flags=re.I)
    )
    if oom_warnings and recoverable_oom < oom_warnings:
        raise RuntimeError(f"{oom_warnings} CUDA OOM messages but only {recoverable_oom} recoverable retries in {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "nms_time_limit_warnings": nms_warnings,
        "nms_warning_override": bool(nms_warnings and allow_nms_warning),
        "cuda_oom_warnings": oom_warnings,
        "cuda_oom_recoverable_retries": recoverable_oom,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--check-code", action="store_true")
    parser.add_argument("--weights", action="append", default=[], type=Path)
    parser.add_argument("--results", action="append", default=[], type=Path)
    parser.add_argument("--json", action="append", default=[], type=Path)
    parser.add_argument("--log", action="append", default=[], type=Path)
    parser.add_argument("--allow-nms-warning", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {"stage": args.stage, "status": "passed"}
    try:
        if args.check_code:
            report["code"] = check_code()
        report["weights"] = [check_weights(path) for path in args.weights]
        report["results"] = [check_results(path) for path in args.results]
        report["json"] = [check_json(path) for path in args.json]
        report["logs"] = [check_log(path, args.allow_nms_warning) for path in args.log]
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        raise
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
