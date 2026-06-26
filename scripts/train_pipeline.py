#!/usr/bin/env python3
"""Multi-stage training pipeline with safety checks and auto-tuning.

Pre-flight checks (blocking — will abort if any fail):
  - GPU available with sufficient VRAM
  - Dataset YAML / image files exist for each requested stage
  - Model YAML loads and is compatible with COCO pretrained weights
  - ``weights/`` directory exists (created if missing)

Hardware auto-tuning:
  - Batch size derived from detected VRAM, model family, and image size
  - DataLoader workers capped at 8 (safe default for cloud instances)
  - ``cache=disk`` on first stage (speeds up subsequent stages)

Unified training stages::

    Stage 1 → Stage 2 → Stage 3 → Stage 4      (recommended core)
    Stage P2 (optional, for P2 models only)
    Stage 5  (optional, hard negative mining)

Stage chaining::

    python scripts/train_pipeline.py --stages 1 2 3 4

Usage::

    # Recommended: full core pipeline
    python scripts/train_pipeline.py --model yolo11m-EMA-SimAM --device 0

    # Specific stages
    python scripts/train_pipeline.py --stages 1 2 --model yolo11s-EMA-SimAM --device 0

    # Custom batch / workers
    python scripts/train_pipeline.py --stages 3 --model yolo11m-EMA-SimAM --device 0 --batch 24 --workers 4

    # Dry-run (print plan, don't train)
    python scripts/train_pipeline.py --stages 1 2 3 4 --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_pipeline")

# ── Project paths ───────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / "config" / "train"
_PRETRAIN_CONFIG_DIR = _CONFIG_DIR / "pretrain"
_WEIGHTS_DIR = _PROJECT_ROOT / "weights"
_MODELS_DIR = _PROJECT_ROOT / "subway_defect" / "models"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"

# ── Unified stage definitions ───────────────────────────────────────────────
STAGE_DEFS: Dict[str, dict] = {
    "p2": {
        "name": "Stage P2 (optional): TT100K P2 Tiny-Object Head Warmup",
        "yaml": "stage_p2_tiny_pretrain.yaml",
        "output": "weights/stage_p2_tiny_pretrain.pt",
        "epochs": 80,
        "desc": "P2 head warmup on tiny objects (only for P2 models)",
        "required": False,
    },
    "1": {
        "name": "Stage 1: Public Defect Pretraining",
        "yaml": "stage1_public_pretrain.yaml",
        "output": "weights/stage1_public_pretrain.pt",
        "epochs": 120,
        "desc": "DeepPCB + NEU-DET + GC10-DET → generic_defect (1 class)",
        "required": True,
    },
    "2": {
        "name": "Stage 2: Custom Domain Adaptation",
        "yaml": "stage2_domain_adapt.yaml",
        "output": "weights/stage2_domain_adapt.pt",
        "epochs": 50,
        "desc": "Frozen backbone, adapt neck+head to contact-net 7 classes",
        "required": True,
    },
    "3": {
        "name": "Stage 3: Main Training",
        "yaml": "stage3_main_training.yaml",
        "output": "weights/stage3_main.pt",
        "epochs": 120,
        "desc": "Full unfreeze, 1280px native crops, heavy augmentations",
        "required": True,
    },
    "4": {
        "name": "Stage 4: Short Fine-Tune",
        "yaml": "stage4_short_finetune.yaml",
        "output": "weights/stage4_best_finetune.pt",
        "epochs": 30,
        "desc": "Low LR, minimal augmentation, convergence to real distribution",
        "required": True,
    },
    "5": {
        "name": "Stage 5 (optional): Hard Negative Mining + Calibration",
        "yaml": "stage5_hard_negative.yaml",
        "output": "weights/stage5_calibrated.pt",
        "epochs": 30,
        "desc": "Reduce false positives, per-class threshold calibration",
        "required": False,
    },
}

# ── Model definitions ───────────────────────────────────────────────────────
_MODEL_FILES = {
    "yolo11s-EMA-SimAM": "yolo11s-EMA-SimAM.yaml",
    "yolo11s-P2-EMA-SimAM": "yolo11s-P2-EMA-SimAM.yaml",
    "yolo11m-EMA-SimAM": "yolo11m-EMA-SimAM.yaml",
    "yolo11m-P2-SimAM": "yolo11m-P2-SimAM.yaml",
}


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                        PRE-FLIGHT SAFETY CHECKS                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

@dataclass
class PreflightResult:
    """Result of pre-flight hardware / environment checks."""
    gpu_name: str = ""
    vram_gb: float = 0.0
    cpu_cores: int = 8
    ram_gb: float = 32.0
    recommended_batch: int = 8
    recommended_workers: int = 4
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0


def run_preflight(
    model_key: str,
    stage_ids: List[str],
    batch_override: Optional[int] = None,
) -> PreflightResult:
    """Run all pre-flight safety checks.

    Returns a :class:`PreflightResult`; callers should inspect ``.ok`` and
    ``.errors`` / ``.warnings`` before proceeding.
    """
    result = PreflightResult()

    # ── 1. Python version ──
    if sys.version_info < (3, 10):
        result.errors.append(f"Python ≥ 3.10 required (found {sys.version})")

    # ── 2. GPU + HardwareProfile ──
    from subway_defect.train.configs import HardwareProfile
    profile = HardwareProfile.detect()
    result.gpu_name = profile.gpu_name
    result.vram_gb = profile.vram_gb
    result.cpu_cores = profile.cpu_cores
    result.ram_gb = profile.ram_gb
    result.recommended_workers = profile.recommended_workers

    if result.vram_gb <= 0:
        result.errors.append(
            "No GPU detected. Training requires an NVIDIA GPU with CUDA."
        )
    elif result.vram_gb < 6.0:
        result.errors.append(
            f"GPU VRAM too low: {result.vram_gb:.1f} GiB. Minimum 8 GiB required."
        )

    # ── 3. Model YAML ──
    model_file = _MODEL_FILES.get(model_key)
    if model_file is None:
        result.errors.append(
            f"Unknown model key '{model_key}'. Choose from: {list(_MODEL_FILES)}"
        )
    else:
        model_path = _MODELS_DIR / model_file
        if not model_path.exists():
            result.errors.append(f"Model YAML not found: {model_path}")

    # ── 4. Stage configs / data ──
    for sid in stage_ids:
        stage = STAGE_DEFS.get(sid)
        if stage is None:
            result.errors.append(f"Unknown stage '{sid}'. Valid: {sorted(STAGE_DEFS)}")
            continue

        cfg_path = _PRETRAIN_CONFIG_DIR / stage["yaml"]
        if not cfg_path.exists():
            result.errors.append(
                f"Stage {sid} config not found: {cfg_path}\n"
                f"  Run: python scripts/multi_source_pretrain_yaml.py --stages {sid}"
            )
            continue

        # Quick check: does the data path referenced by the config exist?
        from subway_defect.train.configs import _safe_load_yaml
        try:
            cfg = _safe_load_yaml(cfg_path)
        except Exception:
            continue

        data_path = cfg.get("path", "")
        if data_path and not Path(data_path).is_dir():
            result.errors.append(
                f"Stage {sid}: dataset root not found: {data_path}"
            )

        train_img = Path(data_path) / "images" / "train" if data_path else None
        if train_img and train_img.is_dir():
            n = len(list(train_img.glob("*")))
            if n == 0:
                result.errors.append(f"Stage {sid}: no images in {train_img}")
        elif train_img:
            result.warnings.append(
                f"Stage {sid}: images/train not found at {train_img} — training may fail"
            )

    # ── 5. Weights directory ──
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 6. COCO pretrained weight ──
    coco_weight = _WEIGHTS_DIR / "yolo11s.pt"
    if "yolo11s" in model_key and not coco_weight.exists():
        result.warnings.append(
            f"{coco_weight.name} not found in weights/ — will be auto-downloaded"
        )
    coco_weight_m = _WEIGHTS_DIR / "yolo11m.pt"
    if "yolo11m" in model_key and not coco_weight_m.exists():
        result.warnings.append(
            f"{coco_weight_m.name} not found in weights/ — will be auto-downloaded"
        )

    # ── 7. Batch size recommendation ──
    if result.vram_gb > 0 and not batch_override:
        result.recommended_batch = profile.recommend_batch_size(
            model_path=str(model_path) if model_file else "",
        )
    elif batch_override:
        result.recommended_batch = batch_override

    return result


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                          STAGE TRAINING LOGIC                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def _resolve_pretrained(stage_id: str, stages_done: List[str]) -> Optional[Path]:
    """Find the best pretrained weight for *stage_id*.

    Priority: previous stage output > COCO pretrain > None.
    """
    # Find the most recent completed stage output
    stage_order = ["p2", "1", "2", "3", "4", "5"]
    try:
        current_idx = stage_order.index(stage_id)
    except ValueError:
        current_idx = len(stage_order)

    for prev_id in reversed(stage_order[:current_idx]):
        if prev_id in stages_done:
            out = STAGE_DEFS.get(prev_id, {}).get("output", "")
            if out:
                p = Path(out)
                if p.exists():
                    return p

    # Fallback: COCO pretrained weights
    return None


def _run_stage(
    stage_id: str,
    model_key: str,
    device: str,
    batch: int,
    workers: int,
    pretrained: Optional[Path],
    dry_run: bool = False,
) -> bool:
    """Execute a single training stage.  Returns True on success."""
    stage = STAGE_DEFS[stage_id]
    cfg_path = _PRETRAIN_CONFIG_DIR / stage["yaml"]
    model_file = _MODEL_FILES[model_key]
    model_path = _MODELS_DIR / model_file
    output_weight = Path(stage["output"])

    print()
    print("=" * 70)
    print(f"  {stage['name']}")
    print(f"  {stage['desc']}")
    print("=" * 70)
    print(f"  Config:   {cfg_path}")
    print(f"  Model:    {model_path}")
    print(f"  Device:   cuda:{device}")
    print(f"  Batch:    {batch}")
    print(f"  Workers:  {workers}")
    print(f"  Epochs:   {stage['epochs']}")
    print(f"  Output:   {output_weight}")
    if pretrained:
        print(f"  Init:     {pretrained}")
    else:
        print(f"  Init:     COCO pretrained (auto-download)")
    print()

    if dry_run:
        print("  [DRY-RUN] Would train here")
        return True

    # ── Load config via safe loader ──
    from subway_defect.train.configs import _safe_load_yaml

    cfg = _safe_load_yaml(cfg_path)
    cfg["batch"] = batch
    cfg["workers"] = workers
    cfg["device"] = device
    cfg["project"] = str(_PROJECT_ROOT / "output")
    cfg["name"] = f"stage{stage_id}"

    # Resolve pretrained weights
    if pretrained and pretrained.exists():
        cfg["pretrained"] = str(pretrained)
    elif not pretrained:
        coco_map = {"yolo11s": "yolo11s.pt", "yolo11m": "yolo11m.pt"}
        for k, w in coco_map.items():
            if k in model_key:
                w_path = _WEIGHTS_DIR / w
                cfg["pretrained"] = str(w_path) if w_path.exists() else w
                break

    logger.info("Starting %s (%d epochs, batch=%d) ...",
                stage["name"], stage["epochs"], batch)
    t0 = time.time()

    try:
        from subway_yolo import YOLO
        model = YOLO(str(model_path))
        model.train(**cfg)

        elapsed = time.time() - t0
        logger.info("%s complete in %s", stage["name"],
                     time.strftime("%H:%M:%S", time.gmtime(elapsed)))

        # Copy best.pt to designated output location
        best_src = (
            _PROJECT_ROOT / "output" / f"stage{stage_id}" / "weights" / "best.pt"
        )
        if best_src.exists():
            output_weight.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_src, output_weight)
            logger.info("Saved: %s", output_weight)
            return True
        else:
            logger.error("best.pt not found at %s — training may have failed", best_src)
            return False

    except Exception as exc:
        logger.error("Stage %s failed: %s", stage_id, exc)
        return False


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                MAIN                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified multi-stage training pipeline with safety checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recommended: full core pipeline
  python scripts/train_pipeline.py --model yolo11m-EMA-SimAM --device 0

  # Specific stages
  python scripts/train_pipeline.py --stages 1 2 --model yolo11s-EMA-SimAM --device 0

  # Include optional P2 stage
  python scripts/train_pipeline.py --stages p2 1 2 3 4

  # Dry-run
  python scripts/train_pipeline.py --stages 1 2 3 4 --dry-run
        """,
    )
    parser.add_argument(
        "--stages", type=str, nargs="+", default=["1", "2", "3", "4"],
        help="Stages to run (default: 1 2 3 4). Valid: p2, 1, 2, 3, 4, 5",
    )
    parser.add_argument(
        "--phases", type=str, nargs="*", dest="stages_deprecated",
        help="Deprecated: use --stages instead.",
    )
    parser.add_argument(
        "--model", type=str, default="yolo11m-EMA-SimAM",
        choices=list(_MODEL_FILES),
        help="Model variant (default: yolo11m-EMA-SimAM)",
    )
    parser.add_argument("--device", type=str, default="0", help="CUDA device (default: 0)")
    parser.add_argument(
        "--batch", type=int, default=None,
        help="Override auto-detected batch size",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override DataLoader worker count (default: auto)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan and pre-flight results without training",
    )
    args = parser.parse_args()

    # Resolve stages (handle --phases deprecation)
    stage_ids = args.stages if args.stages else (args.stages_deprecated or ["1", "2", "3", "4"])
    # Normalize to strings
    stage_ids = [str(s) for s in stage_ids]

    # ── Validate stages ──
    for sid in stage_ids:
        if sid not in STAGE_DEFS:
            print(f"ERROR: Unknown stage '{sid}'. Valid: {sorted(STAGE_DEFS)}")
            sys.exit(1)

    # ═════════════════════════════════════════════════════════════════════
    #  PRE-FLIGHT CHECKS
    # ═════════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  PRE-FLIGHT SAFETY CHECKS")
    print("=" * 70)

    preflight = run_preflight(args.model, stage_ids, args.batch)

    print(f"  GPU:      {preflight.gpu_name or 'N/A'}  ({preflight.vram_gb:.1f} GiB VRAM)")
    print(f"  CPU:      {preflight.cpu_cores} cores")
    print(f"  RAM:      {preflight.ram_gb:.1f} GiB")
    print(f"  Model:    {args.model}")
    print(f"  Stages:   {stage_ids}")
    print(f"  Batch:    {preflight.recommended_batch} (auto)" if not args.batch
          else f"  Batch:    {args.batch} (manual)")
    print(f"  Workers:  {args.workers or preflight.recommended_workers}")

    # ── Warnings ──
    if preflight.warnings:
        print()
        for w in preflight.warnings:
            print(f"  [WARN] {w}")

    # ── Errors ──
    if preflight.errors:
        print()
        print("  " + "=" * 60)
        print("  PRE-FLIGHT FAILED — fix these before training:")
        for e in preflight.errors:
            print(f"    [ERROR] {e}")
        print("  " + "=" * 60)
        sys.exit(1)

    print()
    print("  [OK] All pre-flight checks passed")

    batch = args.batch or preflight.recommended_batch
    workers = args.workers or preflight.recommended_workers

    # ═════════════════════════════════════════════════════════════════════
    #  GENERATE STAGE CONFIGS (if missing)
    # ═════════════════════════════════════════════════════════════════════
    missing_stages = []
    for sid in stage_ids:
        cfg_path = _PRETRAIN_CONFIG_DIR / STAGE_DEFS[sid]["yaml"]
        if not cfg_path.exists():
            missing_stages.append(sid)

    if missing_stages:
        print(f"\n  Generating configs for stages {missing_stages} ...")
        if not args.dry_run:
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / "multi_source_pretrain_yaml.py"),
                 "--stages"] + missing_stages,
                capture_output=False,
                cwd=str(_PROJECT_ROOT),
            )
            if result.returncode != 0:
                print("  [ERROR] Config generation failed — see output above")
                sys.exit(1)

    # ═════════════════════════════════════════════════════════════════════
    #  TRAINING LOOP
    # ═════════════════════════════════════════════════════════════════════
    stages_done: List[str] = []
    failed: List[str] = []

    for sid in stage_ids:
        pretrained = _resolve_pretrained(sid, stages_done)
        ok = _run_stage(
            stage_id=sid,
            model_key=args.model,
            device=args.device,
            batch=batch,
            workers=workers,
            pretrained=pretrained,
            dry_run=args.dry_run,
        )
        if ok:
            stages_done.append(sid)
        else:
            failed.append(sid)
            logger.error(
                "Stage %s failed — stopping. Fix the issue and re-run with "
                "--stages %s to resume.",
                sid, " ".join(str(s) for s in stage_ids if (
                    stage_ids.index(s) >= stage_ids.index(sid)
                )),
            )
            break

    # ═════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═════════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  TRAINING PIPELINE SUMMARY")
    print("=" * 70)
    if args.dry_run:
        print("  Mode:      DRY-RUN (no training executed)")
    print(f"  Completed: {stages_done}")
    if failed:
        print(f"  Failed:    {failed}")
    print(f"  Output weights:")
    for sid in stage_ids:
        out = STAGE_DEFS[sid]["output"]
        exists = Path(out).exists()
        marker = " ← EXISTS" if exists else " (not generated)"
        label = "Required" if STAGE_DEFS[sid].get("required") else "Optional"
        print(f"    Stage {sid} [{label}]: {out}{marker}")
    print("=" * 70)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
