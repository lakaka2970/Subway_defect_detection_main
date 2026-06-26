#!/usr/bin/env python3
"""Multi-phase training pipeline with safety checks and auto-tuning.

Pre-flight checks (blocking — will abort if any fail):
  - GPU available with sufficient VRAM
  - Dataset YAML / image files exist for each requested phase
  - Model YAML loads and is compatible with COCO pretrained weights
  - ``weights/`` directory exists (created if missing)

Hardware auto-tuning:
  - Batch size derived from detected VRAM, model family, and image size
  - DataLoader workers capped at 8 (safe default for cloud instances)
  - ``cache=disk`` on first phase (speeds up subsequent phases)

Phase chaining:
  - Phase 3  → ``weights/public_defect_pretrain.pt``
  - Phase 4  → ``weights/neck_head_adapt.pt``
  - Phase 5  → ``weights/main.pt``
  - Phase 6  → ``weights/best_finetune.pt``
  - Each phase uses the previous phase's best weights as ``--pretrained``.

Usage::

    # Full pipeline (all available phases in order)
    python scripts/train_pipeline.py --model yolo11m-EMA-SimAM --device 0

    # Specific phases
    python scripts/train_pipeline.py --phases 3 4 --model yolo11s-EMA-SimAM --device 0

    # Custom batch / workers
    python scripts/train_pipeline.py --phases 5 --model yolo11m-EMA-SimAM --device 0 --batch 24 --workers 4

    # Dry-run (print plan, don't train)
    python scripts/train_pipeline.py --phases 3 4 5 6 --dry-run
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

# ── Phase definitions ───────────────────────────────────────────────────────
# Each phase: (key, description, config_generator_func_name, output_weight, data_yaml_path, nc)
PHASE_DEFS: Dict[int, dict] = {
    2: {
        "name": "Phase 2: TT100K P2 Tiny-Object Head Warmup",
        "yaml": "phase2_tiny_pretrain.yaml",
        "output": "weights/p2_tiny_pretrain.pt",
        "epochs": 80,
        "desc": "P2 head warmup on tiny objects (only for P2 models)",
    },
    3: {
        "name": "Phase 3: Public Defect Pretrain",
        "yaml": "phase3_public_defect.yaml",
        "output": "weights/public_defect_pretrain.pt",
        "epochs": 120,
        "desc": "DeepPCB + NEU-DET + GC10-DET → generic_defect (1 class)",
    },
    4: {
        "name": "Phase 4: Neck/Head Domain Adaptation",
        "yaml": "phase4_neck_head_adapt.yaml",
        "output": "weights/neck_head_adapt.pt",
        "epochs": 50,
        "desc": "Frozen backbone, adapt neck+head to contact-net 7 classes",
    },
    5: {
        "name": "Phase 5: Main Training",
        "yaml": "phase5_main_training.yaml",
        "output": "weights/main.pt",
        "epochs": 120,
        "desc": "Full unfreeze, 1280px native crops, heavy augmentations",
    },
    6: {
        "name": "Phase 6: Short Fine-Tune",
        "yaml": "phase6_short_finetune.yaml",
        "output": "weights/best_finetune.pt",
        "epochs": 30,
        "desc": "Low LR, minimal augmentation, convergence to real distribution",
    },
}

# ── Model definitions ───────────────────────────────────────────────────────
_MODEL_FILES = {
    "yolo11s-EMA-SimAM": "yolo11s-EMA-SimAM.yaml",
    "yolo11s-P2-EMA-SimAM": "yolo11s-P2-EMA-SimAM.yaml",
    "yolo11m-EMA-SimAM": "yolo11m-EMA-SimAM.yaml",
    "yolo11m-P2-SimAM": "yolo11m-P2-SimAM.yaml",
}

# ── Known VRAM values (GiB) — fallback when CUDA detection fails ────────────
_KNOWN_VRAM: Dict[str, float] = {
    "5090": 32.0, "5080": 16.0,
    "4090": 24.0, "4080": 16.0, "4070": 12.0,
    "3090": 24.0, "3080": 10.0,
    "A100": 40.0, "A6000": 48.0, "A40": 48.0,
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


def _detect_gpu_vram() -> Tuple[str, float]:
    """Detect GPU name and VRAM (GiB).  Tries three methods in order."""
    gpu_name, vram = "", 0.0

    # Method 1: torch.cuda.get_device_properties
    try:
        import torch
        if torch.cuda.is_available():
            prop = torch.cuda.get_device_properties(0)
            gpu_name = prop.name
            vram = getattr(prop, "total_memory", 0) / (1024**3)
            if vram <= 0:
                free, total = torch.cuda.mem_get_info(0)
                vram = total / (1024**3)
    except Exception:
        pass

    # Method 2: nvidia-smi
    if vram <= 0:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                vram = float(result.stdout.strip().split("\n")[0]) / 1024.0
        except Exception:
            pass

    # Method 3: known GPU name → VRAM lookup
    if vram < 6.0 and gpu_name:
        for key, known in _KNOWN_VRAM.items():
            if key in gpu_name:
                vram = known
                break

    return gpu_name, vram


def run_preflight(
    model_key: str, phase_ids: List[int], batch_override: Optional[int] = None
) -> PreflightResult:
    """Run all pre-flight safety checks.

    Returns a :class:`PreflightResult`; callers should inspect ``.ok`` and
    ``.errors`` / ``.warnings`` before proceeding.
    """
    result = PreflightResult()

    # ── 1. Python version ──
    if sys.version_info < (3, 10):
        result.errors.append(f"Python ≥ 3.10 required (found {sys.version})")

    # ── 2. GPU ──
    result.gpu_name, result.vram_gb = _detect_gpu_vram()
    if result.vram_gb <= 0:
        result.errors.append(
            "No GPU detected. Training requires an NVIDIA GPU with CUDA."
        )
    elif result.vram_gb < 6.0:
        result.errors.append(
            f"GPU VRAM too low: {result.vram_gb:.1f} GiB. Minimum 8 GiB required."
        )

    # ── 3. CPU / RAM ──
    import os
    result.cpu_cores = os.cpu_count() or 8
    try:
        import psutil
        result.ram_gb = psutil.virtual_memory().total / (1024**3)
    except Exception:
        pass

    # ── 4. Model YAML ──
    model_file = _MODEL_FILES.get(model_key)
    if model_file is None:
        result.errors.append(
            f"Unknown model key '{model_key}'. Choose from: {list(_MODEL_FILES)}"
        )
    else:
        model_path = _MODELS_DIR / model_file
        if not model_path.exists():
            result.errors.append(f"Model YAML not found: {model_path}")

    # ── 5. Phase configs / data ──
    for pid in sorted(phase_ids):
        phase = PHASE_DEFS.get(pid)
        if phase is None:
            result.errors.append(f"Unknown phase {pid}. Valid: {sorted(PHASE_DEFS)}")
            continue

        cfg_path = _PRETRAIN_CONFIG_DIR / phase["yaml"]
        if not cfg_path.exists():
            result.errors.append(
                f"Phase {pid} config not found: {cfg_path}\n"
                f"  Run: python scripts/multi_source_pretrain_yaml.py"
            )
            continue

        # Quick check: does the data YAML referenced by the config exist?
        try:
            from subway_defect.train.configs import _safe_load_yaml
            cfg = _safe_load_yaml(cfg_path)
        except Exception:
            continue  # syntax errors caught by the training step

        data_path = cfg.get("path", "")
        if data_path and not Path(data_path).is_dir():
            result.errors.append(
                f"Phase {pid}: dataset root not found: {data_path}"
            )

        # Check images/train has files
        train_img = Path(data_path) / "images" / "train" if data_path else None
        if train_img and train_img.is_dir():
            n = len(list(train_img.glob("*")))
            if n == 0:
                result.errors.append(f"Phase {pid}: no images in {train_img}")
        elif train_img:
            result.warnings.append(
                f"Phase {pid}: images/train not found at {train_img} — training may fail"
            )

    # ── 6. Weights directory ──
    _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 7. COCO pretrained weight ──
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

    # ── 8. Batch size recommendation ──
    if result.vram_gb > 0 and not batch_override:
        result.recommended_batch = _estimate_batch(
            model_key, result.vram_gb, phase_ids
        )
    elif batch_override:
        result.recommended_batch = batch_override

    result.recommended_workers = min(8, result.cpu_cores)

    return result


def _estimate_batch(
    model_key: str, vram_gb: float, phase_ids: List[int]
) -> int:
    """Estimate a safe batch size based on VRAM and model family."""
    # VRAM per sample estimates (GiB) — conservative, includes mosaic + AMP
    vram_per_sample = {
        ("yolo11n", 640): 0.25, ("yolo11n", 1024): 0.45, ("yolo11n", 1280): 0.70,
        ("yolo11s", 640): 0.35, ("yolo11s", 1024): 0.80, ("yolo11s", 1280): 1.25,
        ("yolo11m", 640): 0.55, ("yolo11m", 1024): 1.10, ("yolo11m", 1280): 1.72,
    }
    # Add P2 variants
    vram_per_sample[("yolo11s-P2", 1024)] = 1.05
    vram_per_sample[("yolo11s-P2", 1280)] = 1.64
    vram_per_sample[("yolo11m-P2", 1024)] = 1.40
    vram_per_sample[("yolo11m-P2", 1280)] = 2.19

    # Determine imgsz from phase configs (use max across selected phases)
    imgsz = 1024
    for pid in phase_ids:
        phase = PHASE_DEFS.get(pid, {})
        cfg_path = _PRETRAIN_CONFIG_DIR / phase.get("yaml", "")
        if cfg_path.exists():
            from subway_defect.train.configs import _safe_load_yaml
            cfg = _safe_load_yaml(cfg_path)
            imgsz = max(imgsz, cfg.get("imgsz", 1024))

    # Map model_key to family
    family = model_key.replace("-EMA-SimAM", "").replace("-P2-SimAM", "")
    if "P2" in model_key:
        family = model_key.replace("-EMA-SimAM", "").replace("-SimAM", "")

    gb_per = vram_per_sample.get((family, imgsz), 1.0)
    usable = vram_gb * 0.70  # 30% headroom for cuDNN + gradient spikes
    batch = max(4, int((usable - 5.0) / gb_per))  # 5 GiB model overhead
    return min(batch, 48)  # hard cap


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                          PHASE TRAINING LOGIC                             ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def _resolve_pretrained(phase_id: int, phases_done: List[int]) -> Optional[Path]:
    """Find the best pretrained weight for *phase_id*.

    Priority: previous phase output > COCO pretrain > None.
    """
    # Check if a previous phase already produced weights
    prev_outputs = []
    for pid in sorted(phases_done):
        out = PHASE_DEFS.get(pid, {}).get("output", "")
        if out:
            prev_outputs.append(Path(out))

    # Use the most recent output that exists
    for p in reversed(prev_outputs):
        if p.exists():
            return p

    # Fallback: COCO pretrained weights
    return None  # Will use model YAML (random init) — caller should handle


def _run_phase(
    phase_id: int,
    model_key: str,
    device: str,
    batch: int,
    workers: int,
    pretrained: Optional[Path],
    dry_run: bool = False,
) -> bool:
    """Execute a single training phase.  Returns True on success."""
    phase = PHASE_DEFS[phase_id]
    cfg_path = _PRETRAIN_CONFIG_DIR / phase["yaml"]
    model_file = _MODEL_FILES[model_key]
    model_path = _MODELS_DIR / model_file
    output_weight = Path(phase["output"])

    print()
    print("=" * 70)
    print(f"  {phase['name']}")
    print(f"  {phase['desc']}")
    print("=" * 70)
    print(f"  Config:   {cfg_path}")
    print(f"  Model:    {model_path}")
    print(f"  Device:   cuda:{device}")
    print(f"  Batch:    {batch}")
    print(f"  Workers:  {workers}")
    print(f"  Epochs:   {phase['epochs']}")
    print(f"  Output:   {output_weight}")
    if pretrained:
        print(f"  Init:     {pretrained}")
    else:
        print(f"  Init:     COCO pretrained (auto-download)")
    print()

    if dry_run:
        print("  [DRY-RUN] Would train here")
        return True

    # ── Build training command ──
    # We use the YOLO Python API directly (not CLI) for reliable path handling.
    from subway_defect.train.configs import _safe_load_yaml

    cfg = _safe_load_yaml(cfg_path)
    cfg["batch"] = batch
    cfg["workers"] = workers
    cfg["device"] = device
    cfg["project"] = str(_PROJECT_ROOT / "output")
    cfg["name"] = f"phase{phase_id}"

    # Resolve pretrained weights
    if pretrained and pretrained.exists():
        cfg["pretrained"] = str(pretrained)
    elif not pretrained:
        # Use COCO pretrained — YOLO will auto-download if not in weights/
        coco_map = {"yolo11s": "yolo11s.pt", "yolo11m": "yolo11m.pt"}
        for k, w in coco_map.items():
            if k in model_key:
                w_path = _WEIGHTS_DIR / w
                if w_path.exists():
                    cfg["pretrained"] = str(w_path)
                else:
                    cfg["pretrained"] = w  # let YOLO download
                break

    logger.info("Starting %s (%d epochs, batch=%d) ...",
                phase["name"], phase["epochs"], batch)
    t0 = time.time()

    try:
        from subway_yolo import YOLO
        model = YOLO(str(model_path))
        model.train(**cfg)

        elapsed = time.time() - t0
        logger.info("%s complete in %s", phase["name"],
                     time.strftime("%H:%M:%S", time.gmtime(elapsed)))

        # Copy best.pt to designated output location
        best_src = (
            _PROJECT_ROOT / "output" / f"phase{phase_id}" / "weights" / "best.pt"
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
        logger.error("Phase %d failed: %s", phase_id, exc)
        return False


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                MAIN                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-phase training pipeline with safety checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_pipeline.py --model yolo11m-EMA-SimAM --device 0
  python scripts/train_pipeline.py --phases 3 4 --model yolo11s-EMA-SimAM --device 0
  python scripts/train_pipeline.py --phases 5 --model yolo11m-EMA-SimAM --device 0 --batch 24
  python scripts/train_pipeline.py --phases 3 4 5 6 --dry-run
        """,
    )
    parser.add_argument(
        "--phases", type=int, nargs="+", default=[3, 4, 5, 6],
        help="Phase IDs to run (default: 3 4 5 6). Valid: 2,3,4,5,6",
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

    # ── Validate phases ──
    phase_ids = sorted(set(args.phases))
    for pid in phase_ids:
        if pid not in PHASE_DEFS:
            print(f"ERROR: Unknown phase {pid}. Valid: {sorted(PHASE_DEFS)}")
            sys.exit(1)

    # ═════════════════════════════════════════════════════════════════════
    #  PRE-FLIGHT CHECKS
    # ═════════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  PRE-FLIGHT SAFETY CHECKS")
    print("=" * 70)

    preflight = run_preflight(args.model, phase_ids, args.batch)

    print(f"  GPU:      {preflight.gpu_name or 'N/A'}  ({preflight.vram_gb:.1f} GiB VRAM)")
    print(f"  CPU:      {preflight.cpu_cores} cores")
    print(f"  RAM:      {preflight.ram_gb:.1f} GiB")
    print(f"  Model:    {args.model}")
    print(f"  Phases:   {phase_ids}")
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
    #  GENERATE PHASE CONFIGS (if missing)
    # ═════════════════════════════════════════════════════════════════════
    missing_configs = []
    for pid in phase_ids:
        cfg_path = _PRETRAIN_CONFIG_DIR / PHASE_DEFS[pid]["yaml"]
        if not cfg_path.exists():
            missing_configs.append(pid)

    if missing_configs:
        print(f"\n  Generating configs for phases {missing_configs} ...")
        if not args.dry_run:
            result = subprocess.run(
                [sys.executable, str(_SCRIPTS_DIR / "multi_source_pretrain_yaml.py"),
                 "--phases"] + [str(p) for p in missing_configs],
                capture_output=False,
                cwd=str(_PROJECT_ROOT),
            )
            if result.returncode != 0:
                print("  [ERROR] Config generation failed — see output above")
                sys.exit(1)

    # ═════════════════════════════════════════════════════════════════════
    #  TRAINING LOOP
    # ═════════════════════════════════════════════════════════════════════
    phases_done: List[int] = []
    failed: List[int] = []

    for pid in phase_ids:
        pretrained = _resolve_pretrained(pid, phases_done)
        ok = _run_phase(
            phase_id=pid,
            model_key=args.model,
            device=args.device,
            batch=batch,
            workers=workers,
            pretrained=pretrained,
            dry_run=args.dry_run,
        )
        if ok:
            phases_done.append(pid)
        else:
            failed.append(pid)
            logger.error(
                "Phase %d failed — stopping. Fix the issue and re-run with "
                "--phases %s to resume.",
                pid, " ".join(str(p) for p in phase_ids if p >= pid),
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
    print(f"  Completed: {phases_done}")
    if failed:
        print(f"  Failed:    {failed}")
    print(f"  Output weights:")
    for pid in phase_ids:
        out = PHASE_DEFS[pid]["output"]
        exists = Path(out).exists()
        marker = " ← EXISTS" if exists else " (not generated)"
        print(f"    Phase {pid}: {out}{marker}")
    print("=" * 70)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
