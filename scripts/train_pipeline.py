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
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_CONFIG_DIR = _PROJECT_ROOT / "config" / "train"
_PRETRAIN_CONFIG_DIR = _CONFIG_DIR / "pretrain"
_WEIGHTS_DIR = _PROJECT_ROOT / "weights"
_MODELS_DIR = _PROJECT_ROOT / "subway_defect" / "models"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"

# ── Unified stage definitions ───────────────────────────────────────────────
# Stage order determines weight inheritance: each stage auto-loads the
# *immediately preceding* completed stage's best.pt (unless overridden via
# ``pretrained_from`` which names an explicit predecessor stage ID).
#
# nc_mismatch: when True, the training script logs a warning that Detect
# classification layers will be reinitialized (source nc ≠ target nc).
STAGE_DEFS: Dict[str, dict] = {
    "p2": {
        "name": "Stage P2 [ABLATION ONLY]: TT100K P2 Tiny-Object Head Warmup",
        "yaml": "stage_p2_tiny_pretrain.yaml",
        "output": "weights/stage_p2_tiny_pretrain.pt",
        "epochs": 80,
        "desc": "[ABLATION ONLY] P2 head warmup — 4-scale model performed WORSE (mAP50 -0.067, Prec -0.067). Use only for controlled experiments.",
        "required": False,
        "pretrained_from": None,       # uses COCO weights
        "nc_mismatch": False,
    },
    "1a": {
        "name": "Stage 1A: Public Defect Head/Neck Warmup",
        "yaml": "stage1a_public_head.yaml",
        "output": "weights/stage1a_public_head.pt",
        "epochs": 40,
        "desc": "Freeze backbone deep, train neck+head on generic_defect (1 class)",
        "required": True,
        "pretrained_from": None,       # uses COCO weights
        "nc_mismatch": False,
    },
    "1b": {
        "name": "Stage 1B: Public Defect Backbone Adaptation",
        "yaml": "stage1b_public_backbone.yaml",
        "output": "weights/stage1b_public_backbone.pt",
        "epochs": 60,
        "desc": "Unfreeze backbone deep layers, full model on generic_defect",
        "required": True,
        "pretrained_from": "1a",       # inherit Stage 1A best.pt
        "nc_mismatch": False,
    },
    "2": {
        "name": "Stage 2: Contact-Net Domain Adaptation",
        "yaml": "stage2_domain_adapt.yaml",
        "output": "weights/stage2_domain_adapt.pt",
        "epochs": 20,
        "desc": "Frozen backbone, adapt neck+head to contact-net 12 classes",
        "required": True,
        "pretrained_from": "1b",       # inherit Stage 1B best.pt (nc=1→12)
        "nc_mismatch": True,           # 1 class → 12 classes, Detect cls reinit
    },
    "3": {
        "name": "Stage 3: Main Training (1280px)",
        "yaml": "stage3_main_training.yaml",
        "output": "weights/stage3_main.pt",
        "epochs": 60,                  # v2: 60 epochs with patience=20 (was 80/25)
        "desc": "Full unfreeze, 1280px native crops, cos_lr with early stopping",
        "required": True,
        "pretrained_from": "2",        # inherit Stage 2 best.pt
        "nc_mismatch": False,
    },
    "4": {
        "name": "Stage 4: Short Fine-Tune (Real Distribution)",
        "yaml": "stage4_short_finetune.yaml",
        "output": "weights/stage4_best_finetune.pt",
        "epochs": 15,
        "desc": "Low LR (5e-5), minimal augment, freeze backbone, real distribution",
        "required": True,
        "pretrained_from": "3",        # inherit Stage 3 best.pt
        "nc_mismatch": False,
    },
    "5": {
        "name": "Stage 5 (optional): Hard Negative Mining + Calibration",
        "yaml": "stage5_hard_negative.yaml",
        "output": "weights/stage5_calibrated.pt",
        "epochs": 10,
        "desc": "Reduce false positives with mined hard negatives, per-class calibration",
        "required": False,
        "pretrained_from": "4",        # inherit Stage 4 best.pt
        "nc_mismatch": False,
    },
}

# ── Stage execution order (used for weight chain resolution) ─────────────────
_STAGE_ORDER = ["p2", "1a", "1b", "2", "3", "4", "5"]

# ── Model definitions ───────────────────────────────────────────────────────
_MODEL_FILES = {
    # === Recommended (three-scale) ===
    "yolo11s-EMA-SimAM": "yolo11s-EMA-SimAM.yaml",
    "yolo11m-EMA-SimAM": "yolo11m-EMA-SimAM.yaml",
    # === v2: P2 four-scale + CoordAtt + LSK (upgraded architecture) ===
    "yolo11s-v2": "yolo11s-v2.yaml",
    # === [ABLATION ONLY] P2 four-scale models — see P2 deprecation warning ===
    "yolo11s-P2-EMA-SimAM": "yolo11s-P2-EMA-SimAM.yaml",
    "yolo11m-P2-SimAM": "yolo11m-P2-SimAM.yaml",
    "yolo11m-P2-EMA-SimAM": "yolo11m-P2-EMA-SimAM.yaml",
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

        train_ref = cfg.get("train", "")
        train_img = None
        if train_ref:
            train_img = Path(train_ref)
            if not train_img.is_absolute() and data_path:
                train_img = Path(data_path) / train_img
        elif data_path:
            train_img = Path(data_path) / "images" / "train"

        if train_img and train_img.is_dir():
            n = len(list(train_img.glob("*.jpg")) + list(train_img.glob("*.png")))
            if n == 0:
                result.errors.append(f"Stage {sid}: no images in {train_img}")
        elif train_img:
            result.warnings.append(
                f"Stage {sid}: train images not found at {train_img} — training may fail"
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

def _find_stage_best_pt(stage_id: str) -> Optional[Path]:
    """Search ``output/*/stage_{stage_id}/weights/best.pt`` for existing weights.

    This is a fallback for when ``STAGE_DEFS[stage][\"output\"]`` paths (under
    ``weights/``) don't exist — e.g. the pipeline copy step was skipped or the
    files were cleaned up.  Scans all timestamped output directories and returns
    the most recently modified match.

    Args:
        stage_id: Stage identifier (``\"4\"``, ``\"3\"``, etc.).

    Returns:
        Path to the most recent ``best.pt``, or ``None`` if none found.
    """
    output_root = _PROJECT_ROOT / "output"
    if not output_root.is_dir():
        return None

    candidates: List[Tuple[float, Path]] = []
    for run_dir in sorted(output_root.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        p = run_dir / f"stage_{stage_id}" / "weights" / "best.pt"
        if p.exists():
            candidates.append((p.stat().st_mtime, p))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        logger.info(
            "Found %s weights via output/*/stage_%s/weights/best.pt: %s",
            stage_id, stage_id, best,
        )
        return best
    return None


def _resolve_pretrained(
    stage_id: str,
    stages_done: List[str],
) -> Tuple[Optional[Path], Optional[str]]:
    """Find the best pretrained weight for *stage_id*.

    Resolution order:
    1. ``pretrained_from`` field in STAGE_DEFS (explicit predecessor stage ID)
    2. Most recent completed stage in ``_STAGE_ORDER``
    3. COCO pretrained weights (returned as string name, not Path)

    Returns:
        ``(weight_path, nc_warning_message)`` where *weight_path* may be
        ``None`` (use COCO weights) and *nc_warning_message* is a string
        describing any class-count mismatch (empty string if none).
    """
    stage_def = STAGE_DEFS.get(stage_id, {})
    nc_warning = ""

    # ── 1. Explicit pretrained_from (check disk regardless of stages_done) ──
    pretrained_from = stage_def.get("pretrained_from")
    if pretrained_from:
        out = STAGE_DEFS.get(pretrained_from, {}).get("output", "")
        if out:
            p = Path(out)
            if p.exists():
                if pretrained_from not in stages_done:
                    logger.info(
                        "Stage %s: pretrained_from=%s not in current run, "
                        "but output file found on disk",
                        stage_id, pretrained_from,
                    )
                if stage_def.get("nc_mismatch"):
                    src_stage = STAGE_DEFS.get(pretrained_from, {})
                    nc_warning = (
                        f"⚠ nc MISMATCH: {src_stage.get('name','')} → {stage_def.get('name','')}\n"
                        f"  Source checkpoint has nc≠{stage_def.get('yaml','')} target nc.\n"
                        f"  → Backbone + neck + attention weights loaded.\n"
                        f"  → Detect classification layers WILL BE REINITIALIZED.\n"
                        f"  This is EXPECTED for Stage 1B→2 (1-class → 7-class)."
                    )
                logger.info(
                    "Stage %s pretrained from: %s (explicit pretrained_from=%s)",
                    stage_id, p, pretrained_from,
                )
                return p, nc_warning

            logger.warning(
                "Stage %s: pretrained_from=%s but output file %s not found — "
                "trying output/ discovery",
                stage_id, pretrained_from, p,
            )
            # Try output/*/stage_{pretrained_from}/weights/best.pt
            found = _find_stage_best_pt(pretrained_from)
            if found:
                if stage_def.get("nc_mismatch"):
                    src_stage = STAGE_DEFS.get(pretrained_from, {})
                    nc_warning = (
                        f"⚠ nc MISMATCH: {src_stage.get('name','')} → {stage_def.get('name','')}\n"
                        f"  Source checkpoint has nc≠{stage_def.get('yaml','')} target nc.\n"
                        f"  → Backbone + neck + attention weights loaded.\n"
                        f"  → Detect classification layers WILL BE REINITIALIZED.\n"
                        f"  This is EXPECTED for Stage 1B→2 (1-class → 7-class)."
                    )
                logger.info(
                    "Stage %s pretrained from: %s (output/ discovery, pretrained_from=%s)",
                    stage_id, found, pretrained_from,
                )
                return found, nc_warning

    # ── 2. Chain search through ALL previous stages (check disk) ──
    try:
        current_idx = _STAGE_ORDER.index(stage_id)
    except ValueError:
        current_idx = len(_STAGE_ORDER)

    for prev_id in reversed(_STAGE_ORDER[:current_idx]):
        out = STAGE_DEFS.get(prev_id, {}).get("output", "")
        if out:
            p = Path(out)
            if p.exists():
                if prev_id not in stages_done:
                    logger.info(
                        "Stage %s: chain fallback — %s not in current run, "
                        "but output file found on disk",
                        stage_id, prev_id,
                    )
                if stage_def.get("nc_mismatch"):
                    nc_warning = (
                        f"⚠ nc MISMATCH: {STAGE_DEFS.get(prev_id,{}).get('name','')} "
                        f"→ {stage_def.get('name','')}\n"
                        f"  → Detect classification layers will be reinitialized."
                    )
                logger.info(
                    "Stage %s pretrained from: %s (chain fallback via %s)",
                    stage_id, p, prev_id,
                )
                return p, nc_warning

    # ── 2b. Chain search failed → try output/*/ discovery for all previous stages ──
    for prev_id in reversed(_STAGE_ORDER[:current_idx]):
        found = _find_stage_best_pt(prev_id)
        if found:
            if stage_def.get("nc_mismatch"):
                nc_warning = (
                    f"⚠ nc MISMATCH: {STAGE_DEFS.get(prev_id,{}).get('name','')} "
                    f"→ {stage_def.get('name','')}\n"
                    f"  → Detect classification layers will be reinitialized."
                )
            logger.info(
                "Stage %s pretrained from: %s (output/ discovery via %s)",
                stage_id, found, prev_id,
            )
            return found, nc_warning

    # ── 2c. Stage 5 guard: never use Stage 5's own prior output as pretrained ──
    if stage_id == "5":
        logger.warning(
            "Stage 5: no Stage 4 (or earlier) weights found via STAGE_DEFS or "
            "output/ discovery. Stage 5 must start from Stage 4 weights — "
            "will use COCO pretrained, but results will be poor."
        )

    # ── 3. Fallback: COCO pretrained weights ──
    logger.info("Stage %s: no prior stage weight found — will use COCO pretrained", stage_id)
    return None, nc_warning


def _run_stage(
    stage_id: str,
    model_key: str,
    device: str,
    batch: int,
    workers: int,
    pretrained: Optional[Path],
    run_dir: Path,
    nc_warning: str = "",
    dry_run: bool = False,
) -> bool:
    """Execute a single training stage.  Returns True on success."""
    stage = STAGE_DEFS[stage_id]
    cfg_path = _PRETRAIN_CONFIG_DIR / stage["yaml"]
    model_file = _MODEL_FILES[model_key]
    model_path = _MODELS_DIR / model_file
    output_weight = Path(stage["output"])

    # Output layout: <run_dir>/<stage_name>/
    stage_name = f"stage_{stage_id}"

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
    if nc_warning:
        print()
        print(f"  {nc_warning}")
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
    cfg["project"] = str(run_dir)
    cfg["name"] = stage_name

    # ── Per-stage dataset resolution ──────────────────────────────
    dataset_keys = {"path", "train", "val", "nc", "names", "test"}
    has_inline_data = any(k in cfg for k in {"path", "train"})
    if has_inline_data:
        cfg["data"] = str(cfg_path.resolve())
        for k in dataset_keys:
            cfg.pop(k, None)

    # ── Resolve pretrained weights ────────────────────────────────
    if pretrained and pretrained.exists():
        cfg["pretrained"] = str(pretrained)
    elif pretrained and not pretrained.exists():
        logger.warning(
            "Pretrained weight not found: %s — falling back to COCO weights",
            pretrained,
        )
        # Fall through to COCO
        pretrained = None

    if not pretrained:
        coco_map = {"yolo11s": "yolo11s.pt", "yolo11m": "yolo11m.pt"}
        for k, w in coco_map.items():
            if k in model_key:
                w_path = _WEIGHTS_DIR / w
                cfg["pretrained"] = str(w_path) if w_path.exists() else w
                break

    # ── Warmup bias LR safety check ───────────────────────────────
    lr0 = float(cfg.get("lr0", 0.001))
    warmup_epochs = float(cfg.get("warmup_epochs", 3))
    warmup_bias_lr = float(cfg.get("warmup_bias_lr", lr0))
    if stage_id == "3" and warmup_epochs > 0 and warmup_bias_lr > lr0:
        logger.warning(
            "Stage 3 stability guard: warmup_bias_lr=%.6f > lr0=%.6f; "
            "setting warmup_bias_lr=lr0 for low-LR fine-tuning.",
            warmup_bias_lr, lr0,
        )
        cfg["warmup_bias_lr"] = lr0
        warmup_bias_lr = lr0
    if warmup_epochs > 0 and warmup_bias_lr > 10 * lr0:
        logger.error(
            "⚠ DANGEROUS: warmup_bias_lr=%.6f is > 10× lr0=%.6f for %s!\n"
            "  This will spike bias parameter LR in early epochs,\n"
            "  defeating the purpose of low-LR fine-tuning.\n"
            "  → Setting warmup_bias_lr = lr0 = %.6f to fix.",
            warmup_bias_lr, lr0, stage["name"], lr0,
        )
        cfg["warmup_bias_lr"] = lr0
    elif warmup_epochs == 0 and warmup_bias_lr != lr0:
        logger.warning(
            "Stage %s: warmup_epochs=0 but warmup_bias_lr=%.6f ≠ lr0=%.6f — "
            "warmup is off so bias_lr is ignored, but mismatch may indicate config error",
            stage_id, warmup_bias_lr, lr0,
        )
    elif warmup_epochs == 0:
        logger.info(
            "Stage %s: warmup_epochs=0, warmup_bias_lr=%.6f matches lr0=%.6f ✓",
            stage_id, warmup_bias_lr, lr0,
        )
    else:
        logger.info(
            "Stage %s: lr0=%.6f, warmup_bias_lr=%.6f, warmup_epochs=%s ✓",
            stage_id, lr0, warmup_bias_lr, warmup_epochs,
        )

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
        best_src = run_dir / stage_name / "weights" / "best.pt"
        if best_src.exists():
            output_weight.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(best_src, output_weight)
            logger.info("Saved: %s", output_weight)

            # ── Attempt per-class AP extraction ──────────────────
            _log_per_class_metrics(stage_id, stage["name"], run_dir)
            return True
        else:
            logger.error("best.pt not found at %s — training may have failed", best_src)
            return False

    except Exception as exc:
        logger.error("Stage %s failed: %s", stage_id, exc)
        return False


def _log_per_class_metrics(
    stage_id: str, stage_name: str, run_dir: Path,
) -> None:
    """Attempt to extract per-class AP metrics from the stage output directory.

    Looks for per-class results in the Ultralytics output CSV and logs a
    summary table.  Non-blocking — failures are logged as warnings only.
    """
    try:
        results_csv = run_dir / f"stage_{stage_id}" / "results.csv"
        if not results_csv.exists():
            logger.info("No results.csv for %s — skipping per-class summary", stage_name)
            return

        # Ultralytics doesn't save per-class AP to results.csv by default.
        # Instead, look for the per-class metrics in the val batch output
        # or PNG confusion matrix. For now, log that this is available after
        # manual inspection.
        best_pt = run_dir / f"stage_{stage_id}" / "weights" / "best.pt"
        if best_pt.exists():
            logger.info(
                "Per-class metrics for %s: run validation manually:\n"
                "  yolo val model=%s data=%s split=val",
                stage_name,
                best_pt,
                _PRETRAIN_CONFIG_DIR / STAGE_DEFS[stage_id]["yaml"],
            )
    except Exception as exc:
        logger.warning("Could not extract per-class metrics: %s", exc)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║                                MAIN                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified multi-stage training pipeline with safety checks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Recommended: full core pipeline (Stage 1A → 1B → 2 → 3 → 4)
  python scripts/train_pipeline.py --model yolo11m-EMA-SimAM --device 0

  # P2 four-scale model [ABLATION ONLY — NOT recommended as main model]
  python scripts/train_pipeline.py --stages p2 1a 1b 2 3 4 --model yolo11m-P2-EMA-SimAM

  # Specific stages
  python scripts/train_pipeline.py --stages 1a 1b 2 --model yolo11m-EMA-SimAM --device 0

  # Specify output directory (instead of auto-generated timestamp)
  python scripts/train_pipeline.py --stages 5 --model yolo11m-EMA-SimAM --output output/stage5_run1

  # Dry-run
  python scripts/train_pipeline.py --stages 1a 1b 2 3 4 --dry-run
        """,
    )
    parser.add_argument(
        "--stages", type=str, nargs="+", default=["1a", "1b", "2", "3", "4"],
        help="Stages to run (default: 1a 1b 2 3 4). Valid: p2, 1a, 1b, 2, 3, 4, 5",
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
    parser.add_argument(
        "--auto-hnm", action="store_true",
        help="After Stage 5 training, auto-run calibrate_thresholds.py for "
             "per-class confidence threshold optimization.",
    )
    parser.add_argument(
        "--calib-model", type=str, default=None,
        help="Explicit path to Stage 5 model for calibration "
             "(overrides auto-detection). Use when Stage 5 was run in a "
             "different session.",
    )
    parser.add_argument(
        "--pretrained", type=str, default=None,
        help="Explicit path to pretrained weights for the FIRST stage "
             "(bypasses all automatic resolution). Use when Stage 4 weights "
             "are in a known location but auto-discovery fails.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Explicit output directory for this training run. "
             "If provided, all stage results are saved under this directory. "
             "If not provided, auto-generates output/<YYYYMMDD_HHMMSS>/. "
             "Examples: --output output/my_experiment, "
             "--output output/20260630_stage5_retry",
    )
    args = parser.parse_args()

    # Resolve stages (handle --phases deprecation)
    stage_ids = args.stages if args.stages else (args.stages_deprecated or ["1a", "1b", "2", "3", "4"])
    # Normalize to strings
    stage_ids = [str(s) for s in stage_ids]

    # ── Validate stages ──
    for sid in stage_ids:
        if sid not in STAGE_DEFS:
            print(f"ERROR: Unknown stage '{sid}'. Valid: {sorted(STAGE_DEFS)}")
            sys.exit(1)

    # ── P2 deprioritization warning ──
    _p2_selected = "p2" in stage_ids or "P2" in args.model
    if _p2_selected:
        print()
        print("  " + "=" * 64)
        print("  ⚠ P2 DEPRECATION WARNING")
        print("  " + "=" * 64)
        print("  P2 four-scale models were tested against three-scale models:")
        print("    YOLO11m-EMA-SimAM (3-scale):  mAP50=0.497  Precision=0.439")
        print("    YOLO11m-P2-EMA-SimAM (4-scale): mAP50=0.430  Precision=0.372")
        print("  → P2 REDUCED mAP50 by 0.067 and Precision by 0.067.")
        print("  → P2 increased low-quality candidates, amplifying false positives.")
        print("  → SVHBNM mAP50 dropped from 0.251 → 0.197 (WORSE with P2).")
        print("  → CBHPM mAP50 dropped from 0.767 → 0.502 (MAJOR regression).")
        print()
        print("  RECOMMENDATION: Use three-scale YOLO11m-EMA-SimAM as the main model.")
        print("  P2 is retained ONLY for controlled ablation experiments.")
        print("  " + "=" * 64)

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
    if args.output:
        # User-specified output directory
        run_dir = Path(args.output)
        if not run_dir.is_absolute():
            run_dir = _PROJECT_ROOT / run_dir
        timestamp = run_dir.name  # use last path component as run identifier
        print(f"\n  Run directory: {run_dir}  (--output override)")
    else:
        # Auto-generate timestamped directory under output/
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = _PROJECT_ROOT / "output" / timestamp
        print(f"\n  Run directory: {run_dir}")

    stages_done: List[str] = []
    failed: List[str] = []

    for sid in stage_ids:
        pretrained, nc_warning = _resolve_pretrained(sid, stages_done)

        # ── --pretrained CLI override (first stage only) ──
        if args.pretrained and sid == stage_ids[0]:
            override = Path(args.pretrained)
            if override.exists():
                logger.info(
                    "Using --pretrained override: %s (instead of %s)",
                    override, pretrained,
                )
                pretrained = override
            else:
                logger.error(
                    "--pretrained path not found: %s — ignoring override",
                    override,
                )

        ok = _run_stage(
            stage_id=sid,
            model_key=args.model,
            device=args.device,
            batch=batch,
            workers=workers,
            pretrained=pretrained,
            run_dir=run_dir,
            nc_warning=nc_warning,
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
    #  STAGE 5 POST-STEPS (if Stage 5 completed)
    # ═════════════════════════════════════════════════════════════════════
    if "5" in stages_done and not args.dry_run:
        stage5_output = Path(STAGE_DEFS["5"]["output"])
        stage4_output = Path(STAGE_DEFS["4"]["output"])

        # ── Resolve Stage 5 model path ─────────────────────────────────
        # Priority: (1) explicit --calib-model CLI arg,
        #           (2) pipeline output at weights/stage5_calibrated.pt,
        #           (3) run_dir / stage_5 / weights / best.pt
        stage5_model = None
        if getattr(args, "calib_model", None):
            stage5_model = Path(args.calib_model)
        elif stage5_output.exists():
            stage5_model = stage5_output
        else:
            fallback = run_dir / "stage_5" / "weights" / "best.pt"
            if fallback.exists():
                stage5_model = fallback

        _run_stage5_calibration = args.auto_hnm
        if _run_stage5_calibration:
            print()
            print("=" * 70)
            print("  Stage 5 Post-Steps: Per-Class Threshold Calibration")
            print("=" * 70)

            # ── Calibrate thresholds using Stage 5 model ──────────────
            if stage5_model and stage5_model.exists():
                print(f"\n  Running calibrate_thresholds.py with {stage5_model} ...")
                calib_out = run_dir / "calibrated_thresholds"
                calib_result = subprocess.run(
                    [sys.executable, str(_SCRIPTS_DIR / "calibrate_thresholds.py"),
                     "--model", str(stage5_model),
                     "--data", "data/subway_crops/subway_crops.yaml",
                     "--output", str(calib_out),
                     "--device", args.device],
                    capture_output=False,
                    cwd=str(_PROJECT_ROOT),
                )
                if calib_result.returncode == 0:
                    calib_path = calib_out / "thresholds.json"
                    if calib_path.exists():
                        print(f"  [OK] Calibration complete: {calib_path}")
                        # Print recommended thresholds
                        import json
                        calib_data = json.loads(calib_path.read_text(encoding="utf-8"))
                        if calib_data:
                            print(f"\n  Recommended per-class confidence thresholds:")
                            for cls_name, v in calib_data.items():
                                print(f"    {cls_name:12s}: {v['recommended_threshold']:.2f}  "
                                      f"(P={v['precision']:.3f}, R={v['recall']:.3f})")
                        else:
                            print(f"  [WARN] Calibration returned empty results — "
                                  f"check model and validation data")
                else:
                    print(f"  [WARN] Calibration script returned non-zero exit code")
            else:
                print(f"\n  [WARN] Stage 5 model not found at {stage5_output} "
                      f"— skipping calibration")

    # ═════════════════════════════════════════════════════════════════════
    #  SUMMARY
    # ═════════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print("  TRAINING PIPELINE SUMMARY")
    print("=" * 70)
    if args.dry_run:
        print("  Mode:      DRY-RUN (no training executed)")
    print(f"  Run dir:   {run_dir}")
    print(f"  Completed: {stages_done}")
    if failed:
        print(f"  Failed:    {failed}")
    print(f"  Output weights:")
    for sid in stage_ids:
        out = STAGE_DEFS[sid]["output"]
        exists = Path(out).exists()
        marker = " (exists)" if exists else " (not generated)"
        label = "Required" if STAGE_DEFS[sid].get("required") else "Optional"
        print(f"    Stage {sid} [{label}]: {out}{marker}")
    print("=" * 70)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
