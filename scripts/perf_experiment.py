#!/usr/bin/env python3
"""Performance experiment runner for training optimization.

Runs short training experiments (default 5 epochs) with the PerfMonitor
callback to measure throughput, GPU utilization, and memory usage under
different configurations (batch size, workers, cache strategy).

Usage:
    # Baseline experiment (default config)
    python scripts/perf_experiment.py

    # Custom batch/workers
    python scripts/perf_experiment.py --batch 24 --workers 16

    # Use mixed_pretrain dataset (Stage 1A)
    python scripts/perf_experiment.py --data data/multi_datasets/mixed_pretrain/data.yaml --imgsz 1024

    # Use subway_crops (Stage 3+)
    python scripts/perf_experiment.py --data data/subway_crops/subway_crops.yaml --imgsz 1280

    # Quick 3-epoch test
    python scripts/perf_experiment.py --epochs 3 --batch 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Training performance experiment")
    p.add_argument("--data", type=str, default="data/multi_datasets/mixed_pretrain/data.yaml",
                   help="Dataset YAML path")
    p.add_argument("--model", type=str, default="yolo_weights/yolo11s.pt",
                   help="Model weights or YAML")
    p.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    p.add_argument("--batch", type=int, default=16, help="Batch size")
    p.add_argument("--workers", type=int, default=16, help="DataLoader workers")
    p.add_argument("--imgsz", type=int, default=1024, help="Image size")
    p.add_argument("--device", type=str, default="0", help="Device (0, cpu)")
    p.add_argument("--cache", type=str, default=None, choices=["ram", "disk", None],
                   help="Cache strategy (default: auto)")
    p.add_argument("--name", type=str, default="perf_exp", help="Run name")
    p.add_argument("--project", type=str, default="output/perf_experiments",
                   help="Output project dir")
    return p.parse_args()


def main():
    args = parse_args()

    from subway_yolo import YOLO
    from subway_yolo.utils.callbacks.perf_monitor import register_perf_monitor

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"ERROR: Dataset not found: {data_path}")
        print(f"  Run data preparation first (see docs/Train_guide.md § B)")
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        alt = _PROJECT_ROOT / "weights" / model_path.name
        if alt.exists():
            model_path = alt
        else:
            print(f"WARNING: Model not found at {model_path}, will attempt download")

    print("=" * 60)
    print("  Performance Experiment")
    print("=" * 60)
    print(f"  Data:    {args.data}")
    print(f"  Model:   {args.model}")
    print(f"  Epochs:  {args.epochs}")
    print(f"  Batch:   {args.batch}")
    print(f"  Workers: {args.workers}")
    print(f"  ImgSz:   {args.imgsz}")
    print(f"  Device:  {args.device}")
    print(f"  Cache:   {args.cache or 'auto'}")
    print("=" * 60)

    model = YOLO(str(model_path))

    train_args = {
        "data": str(data_path),
        "epochs": args.epochs,
        "batch": args.batch,
        "workers": args.workers,
        "imgsz": args.imgsz,
        "device": args.device,
        "project": str(_PROJECT_ROOT / args.project),
        "name": args.name,
        "exist_ok": True,
        "verbose": True,
        "plots": False,
        "val": True,
        "save": True,
        "save_period": args.epochs,
        "amp": True,
        "patience": args.epochs + 5,
        "optimizer": "SGD",
        "lr0": 0.01,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "warmup_epochs": 1,
        "mosaic": 0.5,
        "mixup": 0.0,
        "cos_lr": True,
    }

    if args.cache:
        train_args["cache"] = args.cache

    from subway_yolo.engine.trainer import BaseTrainer
    original_do_train = BaseTrainer._do_train

    def _patched_do_train(self):
        register_perf_monitor(self)
        original_do_train(self)

    BaseTrainer._do_train = _patched_do_train

    t0 = time.time()
    results = model.train(**train_args)
    elapsed = time.time() - t0

    BaseTrainer._do_train = original_do_train

    print("\n" + "=" * 60)
    print("  Experiment Complete")
    print("=" * 60)
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    perf_dir = _PROJECT_ROOT / args.project / args.name / "perf_monitor"
    metrics_file = perf_dir / "metrics.jsonl"
    if metrics_file.exists():
        lines = metrics_file.read_text(encoding="utf-8").strip().split("\n")
        if lines:
            epochs_data = [json.loads(l) for l in lines]
            avg_batch = sum(e.get("avg_batch_ms", 0) for e in epochs_data) / len(epochs_data)
            avg_gpu = sum(e.get("gpu_util_pct", 0) for e in epochs_data) / len(epochs_data)
            max_ram = max(e.get("ram_pct", 0) for e in epochs_data)
            avg_epoch = sum(e.get("epoch_time_s", 0) for e in epochs_data) / len(epochs_data)

            n_train = epochs_data[0].get("n_batches", 0) * epochs_data[0].get("batch_size", args.batch)
            samples_per_sec = n_train / avg_epoch if avg_epoch > 0 else 0

            print(f"  Avg batch time:  {avg_batch:.1f} ms")
            print(f"  Avg epoch time:  {avg_epoch:.1f} s")
            print(f"  Throughput:      {samples_per_sec:.1f} samples/s")
            print(f"  Avg GPU util:    {avg_gpu:.1f}%")
            print(f"  Max RAM usage:   {max_ram:.1f}%")
            print(f"  Metrics log:     {metrics_file}")

            if max_ram > 50:
                print(f"\n  ⚠️  RAM exceeds 50% target! Consider reducing workers or cache=disk")
            if 0 < avg_gpu < 70:
                print(f"\n  ⚠️  GPU underutilized! Consider increasing workers or enabling cache=ram")
            elif avg_gpu >= 85:
                print(f"\n  ✅ GPU well utilized ({avg_gpu:.0f}%)")

    print("=" * 60)


if __name__ == "__main__":
    main()
