#!/usr/bin/env python3
"""
Stage 4 Single-Variable Experiments for fine-tuning strategy search.

Runs a matrix of 6 experiments isolating individual Stage 4 hyper-parameters
to identify what causes the degradation (mAP50 0.389→0.379) observed in 7.24.

Experiments (from docs/plans/2026-07-24下一步计划.md, Section 6.2):

    S4-C  Control:     Current Stage 4 config, 5 epochs — confirm degradation
    S4-F7 Freeze 0-7:  Instead of freeze 0-9 (allow deeper adaptation)
    S4-F5 Freeze 0-5:  Even more adaptation freedom
    S4-L  LR 5e-5:     5× higher learning rate for faster convergence
    S4-A  Light Aug:    Restore translate=0.02 + fliplr=0.5 (test if zero aug overfit)
    S4-M  Combined:     freeze 0-5 + lr 5e-5 + light augmentation

Usage::

    python scripts/stage4_experiments.py
    python scripts/stage4_experiments.py --base-weights weights/stage3_main.pt --epochs 5
    python scripts/stage4_experiments.py --experiments S4-C S4-L  # subset only
    python scripts/stage4_experiments.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ── Base Stage 4 config (current) ──────────────────────────────────────────
BASE_STAGE4_CONFIG: dict = {
    "data": "data/subway_crops/subway_crops.yaml",
    "epochs": 5,
    "imgsz": 1280,
    "batch": 16,
    "amp": True,
    "optimizer": "AdamW",
    "lr0": 0.00001,          # 1e-5
    "lrf": 1.0,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 0,
    "warmup_bias_lr": 0.00001,
    "cos_lr": False,
    "patience": 5,
    "save_period": 1,
    "device": "0",
    "workers": 4,
    "project": "runs/stage4_experiments",
    "exist_ok": True,
    "verbose": False,
    # Augmentations — zero/light (per experiment)
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "erasing": 0.0,
    "degrees": 0.0,
    "translate": 0.02,
    "scale": 0.1,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "hsv_h": 0.003,
    "hsv_s": 0.1,
    "hsv_v": 0.1,
    "close_mosaic": 0,
    "freeze": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "seed": 0,
    "conf": 0.01,
}


@dataclass
class Experiment:
    """A single Stage 4 experiment variant."""
    name: str
    description: str
    overrides: dict  # Key-value overrides on top of BASE_STAGE4_CONFIG


# ── Experiment definitions ─────────────────────────────────────────────────

EXPERIMENTS: List[Experiment] = [
    Experiment(
        name="S4-C",
        description="Control — current Stage 4 config, 5 epochs (confirm degradation)",
        overrides={
            "freeze": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "lr0": 1e-5,
            "translate": 0.02,   # Keep base translate
            "fliplr": 0.0,        # No flip
        },
    ),
    Experiment(
        name="S4-F7",
        description="Freeze 0-7 only (allow deeper backbone adaptation)",
        overrides={
            "freeze": [0, 1, 2, 3, 4, 5, 6, 7],
            "lr0": 1e-5,
            "translate": 0.02,
            "fliplr": 0.0,
        },
    ),
    Experiment(
        name="S4-F5",
        description="Freeze 0-5 only (even more adaptation freedom)",
        overrides={
            "freeze": [0, 1, 2, 3, 4, 5],
            "lr0": 1e-5,
            "translate": 0.02,
            "fliplr": 0.0,
        },
    ),
    Experiment(
        name="S4-L",
        description="LR 5e-5 (5× higher, faster convergence)",
        overrides={
            "freeze": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "lr0": 5e-5,
            "translate": 0.02,
            "fliplr": 0.0,
        },
    ),
    Experiment(
        name="S4-A",
        description="Light augmentation: translate=0.02 + fliplr=0.5 (test zero-aug hypothesis)",
        overrides={
            "freeze": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            "lr0": 1e-5,
            "translate": 0.02,
            "fliplr": 0.5,
            "hsv_h": 0.005,
            "hsv_s": 0.15,
            "hsv_v": 0.15,
        },
    ),
    Experiment(
        name="S4-M",
        description="Combined: freeze 0-5 + lr 5e-5 + light augmentation (best guess)",
        overrides={
            "freeze": [0, 1, 2, 3, 4, 5],
            "lr0": 5e-5,
            "translate": 0.02,
            "fliplr": 0.5,
            "hsv_h": 0.005,
            "hsv_s": 0.15,
            "hsv_v": 0.15,
        },
    ),
]


def build_config(base: dict, overrides: dict) -> dict:
    """Merge overrides into base config."""
    cfg = base.copy()
    cfg.update(overrides)
    cfg["name"] = f"stage4_{overrides.get('name', 'variant')}"
    return cfg


def run_experiment(
    exp: Experiment,
    base_weights: str | Path,
    device: str = "0",
    dry_run: bool = False,
) -> Optional[Dict]:
    """Run a single Stage 4 experiment.

    Args:
        exp: Experiment definition.
        base_weights: Path to Stage 3 best.pt to initialize from.
        device: CUDA device string.
        dry_run: Print config only, don't train.

    Returns:
        Dict with results or None on failure.
    """
    cfg = build_config(BASE_STAGE4_CONFIG, {**exp.overrides, "name": exp.name})
    cfg["device"] = device

    print()
    print("=" * 70)
    print(f"  {exp.name}: {exp.description}")
    print("=" * 70)
    print(f"  Init:       {base_weights}")
    print(f"  Freeze:     {cfg['freeze']}")
    print(f"  LR:         {cfg['lr0']:.6f}")
    print(f"  Aug:        translate={cfg['translate']}, fliplr={cfg['fliplr']}")
    print(f"  Epochs:     {cfg['epochs']}")
    print(f"  Device:     cuda:{device}")
    print()

    if dry_run:
        print("  [DRY-RUN] Would train here")
        return None

    # Save config for reproducibility
    project_dir = Path(cfg["project"])
    exp_dir = project_dir / cfg["name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Actually, use a dedicated experiment directory
    run_root = _PROJECT_ROOT / "runs" / "stage4_experiments" / exp.name
    if run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    cfg["project"] = str(run_root.parent)
    cfg["name"] = exp.name
    cfg.pop("data_dict", None)  # Ensure clean config

    try:
        from subway_yolo import YOLO

        model = YOLO(str(base_weights))
        t0 = time.time()

        # Train
        model.train(**cfg)

        elapsed = time.time() - t0
        print(f"\n  {exp.name} complete in {elapsed:.0f}s ({elapsed/60:.1f} min)")

        # Find the best.pt
        best_path = run_root / "weights" / "best.pt"
        if best_path.exists():
            print(f"  Best weights: {best_path}")

            # Run evaluation on frozen_eval
            print(f"\n  Evaluating on frozen_eval ...")
            eval_metrics = _evaluate(best_path, device)
            if eval_metrics:
                print(f"  mAP50: {eval_metrics.get('mAP50', 'N/A')}")
                print(f"  Precision: {eval_metrics.get('precision', 'N/A')}")
                print(f"  Recall: {eval_metrics.get('recall', 'N/A')}")
                return {
                    "experiment": exp.name,
                    "best_path": str(best_path),
                    **eval_metrics,
                }
            return {"experiment": exp.name, "best_path": str(best_path)}
        else:
            print(f"  ERROR: best.pt not found at {best_path}")
            return None

    except Exception as exc:
        print(f"  FAILED: {exc}")
        import traceback
        traceback.print_exc()
        return None


def _evaluate(weights_path: str | Path, device: str = "0") -> Optional[Dict]:
    """Run evaluation on frozen_eval split using val mode.

    We use the subway_crops dataset but override val to frozen_eval for
    independent evaluation.
    """
    try:
        from subway_yolo import YOLO

        model = YOLO(str(weights_path))

        # Use the data YAML but evaluate on frozen_eval
        results = model.val(
            data="data/subway_crops/subway_crops.yaml",
            split="val",  # Use the val split from data yaml — we can't easily override
            imgsz=1280,
            batch=16,
            device=device,
            verbose=False,
        )

        # Extract key metrics (Ultralytics Results object)
        metrics = {}
        if hasattr(results, 'box'):
            box = results.box
            metrics["mAP50"] = float(box.map50) if box.map50 is not None else 0.0
            metrics["mAP50-95"] = float(box.map) if box.map is not None else 0.0
            # Per-class P/R from confusion matrix
            if hasattr(box, 'ap_class_index'):
                metrics["ap_per_class"] = box.ap.tolist() if box.ap is not None else []
        elif hasattr(results, 'results_dict'):
            rd = results.results_dict
            metrics["mAP50"] = rd.get("metrics/mAP50(B)", 0.0)
            metrics["mAP50-95"] = rd.get("metrics/mAP50-95(B)", 0.0)
            metrics["precision"] = rd.get("metrics/precision(B)", 0.0)
            metrics["recall"] = rd.get("metrics/recall(B)", 0.0)

        return metrics

    except Exception as exc:
        print(f"  Eval error: {exc}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Stage 4 single-variable experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/stage4_experiments.py
  python scripts/stage4_experiments.py --base-weights weights/stage3_main.pt
  python scripts/stage4_experiments.py --experiments S4-C S4-L
  python scripts/stage4_experiments.py --dry-run
  python scripts/stage4_experiments.py --epochs 5 --device 0
""",
    )
    parser.add_argument(
        "--base-weights", type=str, default="weights/stage3_main.pt",
        help="Path to Stage 3 best weights (default: weights/stage3_main.pt)",
    )
    parser.add_argument(
        "--experiments", type=str, nargs="*",
        choices=[e.name for e in EXPERIMENTS],
        default=None,
        help="Specific experiments to run (default: all 6)",
    )
    parser.add_argument(
        "--epochs", type=int, default=5,
        help="Epochs per experiment (default: 5)",
    )
    parser.add_argument(
        "--device", type=str, default="0",
        help="CUDA device (default: 0)",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print experiment configs without training",
    )
    args = parser.parse_args()

    # Validate base weights exist
    base_weights = Path(args.base_weights)
    if not args.dry_run and not base_weights.exists():
        print(f"ERROR: Base weights not found: {base_weights}")
        sys.exit(1)

    # Select experiments
    selected = EXPERIMENTS
    if args.experiments:
        selected = [e for e in EXPERIMENTS if e.name in args.experiments]

    print("=" * 70)
    print("  Stage 4 Single-Variable Experiments")
    print("=" * 70)
    print(f"  Base weights:     {base_weights}")
    print(f"  Experiments:      {len(selected)}")
    print(f"  Epochs each:      {args.epochs}")
    print(f"  Device:           cuda:{args.device}")
    print(f"  Seed:             {args.seed}")
    print(f"  Estimated time:   ~{len(selected) * 18} minutes ({len(selected)} × ~18 min)")
    print()

    if args.dry_run:
        for exp in selected:
            run_experiment(exp, base_weights, args.device, dry_run=True)
        print("\n  [DRY-RUN] All experiments listed.")
        return

    # Run experiments sequentially
    results = []
    t_start = time.time()

    for i, exp in enumerate(selected):
        print(f"\n{'─'*70}")
        print(f"  Experiment {i+1}/{len(selected)}: {exp.name}")
        print(f"{'─'*70}")

        result = run_experiment(exp, base_weights, args.device, dry_run=False)
        if result:
            results.append(result)
        else:
            print(f"  ⚠ {exp.name} returned no results — may have failed")

    # ── Final Report ──
    total_elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print("  Stage 4 Experiments — Final Report")
    print("=" * 70)
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print()

    if not results:
        print("  NO RESULTS — all experiments may have failed.")
        print("  Check individual run logs under runs/stage4_experiments/")
        return

    # Print summary table
    print(f"  {'Experiment':<8} {'mAP50':>8} {'mAP50-95':>10} {'P':>8} {'R':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*10} {'─'*8} {'─'*8}")

    # Get baseline Stage 3 metrics for comparison
    baseline_map50 = None
    for r in results:
        if r["experiment"] == "S4-C":
            baseline_map50 = r.get("mAP50")
            break

    best_exp = None
    best_map50 = 0.0

    for r in results:
        name = r["experiment"]
        m50 = r.get("mAP50", 0)
        m5095 = r.get("mAP50-95", 0)
        p = r.get("precision", 0)
        rec = r.get("recall", 0)
        flag = ""
        if m50 > best_map50:
            best_map50 = m50
            best_exp = name
            flag = " ← BEST"
        print(f"  {name:<8} {m50:>8.4f} {m5095:>10.4f} {p:>8.4f} {rec:>8.4f}{flag}")

    print()
    if baseline_map50 and best_map50:
        delta = best_map50 - baseline_map50
        print(f"  Best vs Control: {best_exp} ΔmAP50={delta:+.4f} ({delta/baseline_map50*100:+.1f}%)")
    elif best_exp:
        print(f"  Best experiment: {best_exp} (mAP50={best_map50:.4f})")

    print()
    print("  Recommended Stage 4 config:")
    best_exp_def = next((e for e in EXPERIMENTS if e.name == best_exp), None)
    if best_exp_def:
        print(f"    freeze: {best_exp_def.overrides.get('freeze')}")
        print(f"    lr0:    {best_exp_def.overrides.get('lr0')}")
        print(f"    aug:    translate={best_exp_def.overrides.get('translate')}, "
              f"fliplr={best_exp_def.overrides.get('fliplr')}")
    print()
    print("  Individual run logs: runs/stage4_experiments/*/")


if __name__ == "__main__":
    main()
