"""
Training hyperparameter presets with hardware-aware auto-tuning.

Each dict unpacks into ``YOLO.train(**preset)``.

The :class:`HardwareProfile` class auto-detects GPU VRAM / CPU cores / RAM
and provides per-stage recommendations that balance throughput against OOM risk.
"""

from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

import yaml

# ── Stage A: COCO Pretraining (base weights) ────────────────────
# Not a training stage itself — maps to Ultralytics official weights.
COCO_PRETRAINED = {
    "yolo11n": "yolo11n.pt",   # 2.6M params — ROI proposer
    "yolo11s": "yolo11s.pt",   # 9.5M params — vehicle-side
    "yolo11m": "yolo11m.pt",   # 20.1M params — ground-side
}


# ============================================================================
# Hardware auto-detection
# ============================================================================

@dataclass
class HardwareProfile:
    """Auto-detected hardware capabilities used to tune training configs.

    Attributes:
        gpu_name: GPU model string (empty if no CUDA).
        vram_gb: Total GPU VRAM in GiB (0 if no CUDA).
        cpu_cores: ``os.cpu_count()`` (physical cores on Linux, logical on Windows).
        ram_gb: Total system RAM in GiB.
        recommended_workers: DataLoader worker count safe for this machine.
        recommended_cache: ``"disk"`` or ``"ram"`` depending on available RAM.
    """

    gpu_name: str = ""
    vram_gb: float = 0.0
    cpu_cores: int = 8
    ram_gb: float = 32.0
    recommended_workers: int = 4
    recommended_cache: str = "disk"

    # Class-level constant — VRAM per sample estimates (GB), conservative
    # Key: (model_family, imgsz)  →  GB VRAM per sample with mosaic + mixup + AMP
    _VRAM_PER_SAMPLE: ClassVar[Dict[Tuple[str, int], float]] = {
        ("yolo11n", 640):   0.25,
        ("yolo11s", 640):   0.35,
        ("yolo11s", 1024):  0.80,
        ("yolo11s", 1280):  1.25,   # ~0.80 × (1280/1024)²
        ("yolo11m", 640):   0.55,
        ("yolo11m", 1024):  1.10,
        ("yolo11m", 1280):  1.72,   # ~1.10 × (1280/1024)²
        ("yolo11m-P2", 640):   0.70,
        ("yolo11m-P2", 1024):  1.40,
        ("yolo11m-P2", 1280):  2.19,  # ~1.40 × (1280/1024)²
    }

    @classmethod
    def detect(cls) -> "HardwareProfile":
        """Detect hardware and return a :class:`HardwareProfile`."""
        profile = cls()

        # -- CPU --
        profile.cpu_cores = os.cpu_count() or 8

        # -- GPU --
        try:
            import torch
            if torch.cuda.is_available():
                prop = torch.cuda.get_device_properties(0)
                profile.gpu_name = prop.name
                profile.vram_gb = prop.total_mem / (1024 ** 3)
                # Fallback: some drivers / container runtimes report total_mem=0
                # even when CUDA is available. Use mem_get_info as backup.
                if profile.vram_gb <= 0:
                    free, total = torch.cuda.mem_get_info(0)
                    profile.vram_gb = total / (1024 ** 3)

                # Third fallback: nvidia-smi CLI query (some container runtimes
                # report 0 from both torch APIs above)
                if profile.vram_gb <= 0:
                    try:
                        import subprocess
                        result = subprocess.run(
                            ["nvidia-smi", "--query-gpu=memory.total",
                             "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, timeout=5,
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            vram_mb = float(result.stdout.strip().split("\n")[0])
                            profile.vram_gb = vram_mb / 1024.0
                    except Exception:
                        pass

                # Sanity check: if GPU name contains a known high-VRAM card
                # but detection reports unrealistically low VRAM, use known value.
                _KNOWN_VRAM: dict = {
                    "5090": 32.0,
                    "5080": 16.0,
                    "4090": 24.0,
                    "4080": 16.0,
                    "3090": 24.0,
                    "3080": 10.0,
                    "A100": 40.0,
                    "A6000": 48.0,
                }
                if profile.vram_gb < 6.0 and profile.gpu_name:
                    for gpu_key, known_vram in _KNOWN_VRAM.items():
                        if gpu_key in profile.gpu_name:
                            profile.vram_gb = known_vram
                            break
        except Exception:
            pass

        # -- RAM --
        try:
            import psutil
            profile.ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            pass

        # -- Recommendations --
        # Workers: min(8, cpu_cores) to avoid excessive memory from prefetch buffers.
        # 8 workers is usually enough to saturate GPU I/O on a single-GPU node.
        profile.recommended_workers = min(8, profile.cpu_cores)
        # Prefer disk cache for persistence; only use ram cache if >48 GB RAM
        profile.recommended_cache = "ram" if profile.ram_gb > 48 else "disk"

        return profile

    def print_info(self) -> None:
        """Log detected hardware."""
        logger.info("=== Hardware Profile ===")
        logger.info("  GPU       : %s  (%.1f GB VRAM)", self.gpu_name or "N/A", self.vram_gb)
        logger.info("  CPU cores : %s", self.cpu_cores)
        logger.info("  RAM       : %.1f GB", self.ram_gb)
        logger.info("  Workers   : %s", self.recommended_workers)
        logger.info("  Cache     : %s", self.recommended_cache)
        logger.info("=========================")

    def estimate_batch_size(
        self,
        model_family: str = "yolo11s",
        imgsz: int = 1024,
        safety_margin: float = 0.75,
        min_batch: int = 4,
        max_batch: int = 64,
    ) -> int:
        """Estimate a safe batch size that leaves headroom on the GPU.

        Args:
            model_family: One of ``"yolo11n"``, ``"yolo11s"``, ``"yolo11m"``,
                ``"yolo11m-P2"``.
            imgsz: Training image size.
            safety_margin: Fraction of VRAM allowed for training (0-1).
                Default 0.75 leaves 25% headroom for cuDNN workspaces,
                gradient spikes, and mixed-precision buffers.
            min_batch: Floor value.
            max_batch: Ceiling value.

        Returns:
            Recommended batch size (int).
        """
        if self.vram_gb <= 0:
            return min_batch  # CPU-only fallback

        key = (model_family, imgsz)
        # Fall back to nearest imgsz key (closest match, not just 640)
        if key not in self._VRAM_PER_SAMPLE:
            same_family = [(f, s) for f, s in self._VRAM_PER_SAMPLE if f == model_family]
            if same_family:
                closest = min(same_family, key=lambda k: abs(k[1] - imgsz))
                key = closest
            else:
                key = ("yolo11s", 1024)  # ultimate fallback

        gb_per_sample = self._VRAM_PER_SAMPLE[key]
        usable_vram = self.vram_gb * safety_margin

        # Reserve 6 GB for model weights + optimizer states + cuDNN workspace
        # (AdamW uses ~2× SGD memory for moment buffers; multi_scale peaks use more)
        model_overhead = 6.0
        batch = int((usable_vram - model_overhead) / gb_per_sample)

        return max(min_batch, min(batch, max_batch))

    @staticmethod
    def _infer_family(model_path: str) -> str:
        """Infer model family string from a model path."""
        m = model_path.lower()
        if "yolo11n" in m:
            return "yolo11n"
        if "yolo11m" in m and "p2" in m:
            return "yolo11m-P2"
        if "yolo11m" in m:
            return "yolo11m"
        return "yolo11s"

    def recommend_batch_size(
        self, model_path: str, imgsz: int = 1024
    ) -> int:
        """Convenience wrapper that infers model family from path."""
        return self.estimate_batch_size(
            model_family=self._infer_family(model_path), imgsz=imgsz
        )


# ============================================================================
# Stage-specific training presets
# ============================================================================

# ── Common settings shared across stages ──────────────────────────
_COMMON_BASE: dict = {
    "device": "0",
    "amp": True,               # Automatic Mixed Precision — ~2× speed, 30% less VRAM
    "patience": 50,            # Early stopping after 50 epochs without improvement
    "save": True,
    "save_period": 10,         # Checkpoint every 10 epochs (reduces disk pressure)
    "exist_ok": True,          # Allow overwriting existing run directories
    "verbose": True,
}

# DataLoader — filled dynamically by HardwareProfile
_DATALOADER: dict = {
    "workers": 8,              # overridden at runtime
    "cache": "disk",           # overridden at runtime — "disk" persists across runs
}


# ── ROI proposer (Stage B) — YOLO11n, imgsz=640 ──────────────────

ROI_TRAIN_CONFIG: dict = {
    **_COMMON_BASE,
    **_DATALOADER,
    "data": "datasets/roi/roi_data.yaml",
    "epochs": 200,
    "imgsz": 640,
    "batch": 64,               # Safe default; overridden by HardwareProfile if smaller
    "optimizer": "SGD",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": True,
    "mosaic": 0.8,
    "mixup": 0.1,
    "copy_paste": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0005,
    "flipud": 0.0,
    "fliplr": 0.5,
    "close_mosaic": 0,
}


# ── Defect detector — C1 Head Warmup ─────────────────────────────

DEFECT_WARMUP_CONFIG: dict = {
    **_COMMON_BASE,
    **_DATALOADER,
    "epochs": 50,
    "imgsz": 1024,
    "batch": 16,
    "optimizer": "SGD",
    "lr0": 0.001,
    "lrf": 1.0,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 0.1,
    "cos_lr": False,
    "mosaic": 0.5,
    "mixup": 0.0,
    "copy_paste": 0.0,
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.6,
    "degrees": 5.0,
    "translate": 0.15,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0005,
    "flipud": 0.0,
    "fliplr": 0.5,
    "close_mosaic": 0,
    "freeze": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}


# ── Defect detector — C2 Full Training ───────────────────────────

DEFECT_FULL_TRAIN_CONFIG: dict = {
    **_COMMON_BASE,
    **_DATALOADER,
    "epochs": 200,
    "imgsz": 1024,
    "batch": 16,
    "optimizer": "SGD",        # Same optimizer family as C1 — avoids AdamW reset
    "lr0": 0.001,              # Match C1 LR; cosine decay to 1e-4
    "lrf": 0.1,                # Gentler decay — keeps LR higher for longer
    "momentum": 0.937,
    "weight_decay": 0.0005,    # Match C1 regularization
    "warmup_epochs": 5,        # Longer warmup when resuming from C1 checkpoint
    "cos_lr": True,
    "mosaic": 0.5,             # Reduced from 0.8 — less distortion for small dataset
    "mixup": 0.0,              # Disabled — let model learn real distribution first
    "copy_paste": 0.3,         # Reduced from 0.6 — milder class balancing
    "copy_paste_mode": "flip",
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.6,
    "degrees": 5.0,
    "translate": 0.15,
    "scale": 0.5,
    "shear": 2.0,
    "perspective": 0.0005,
    "flipud": 0.0,
    "fliplr": 0.5,
    "close_mosaic": 15,        # Turn off mosaic for final 15 epochs — adapt to real images
}


# ── Defect detector — C3 Fine-Tune ───────────────────────────────

DEFECT_FINETUNE_CONFIG: dict = {
    **_COMMON_BASE,
    **_DATALOADER,
    "epochs": 50,
    "imgsz": 1024,
    "batch": 8,
    "optimizer": "SGD",        # Same family as C1/C2 — avoids optimizer reset
    "lr0": 0.0001,             # Low constant LR for fine-tuning
    "lrf": 0.1,
    "momentum": 0.937,
    "weight_decay": 0.0005,    # Match C1/C2 regularization
    "cos_lr": False,
    "mosaic": 0.0,
    "mixup": 0.0,
    "copy_paste": 0.0,         # Disabled — fine-tuning on real data only
    "copy_paste_mode": "flip",
    "hsv_h": 0.01,
    "hsv_s": 0.4,
    "hsv_v": 0.3,
    "degrees": 2.0,
    "translate": 0.1,
    "scale": 0.3,
    "shear": 1.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.3,
    "erasing": 0.0,            # Disabled — erasing damages fine-grained defect features
    "close_mosaic": 0,
}


# ============================================================================
# Apply hardware profile to a config dict (mutates in-place)
# ============================================================================

def apply_hardware_profile(
    config: dict,
    profile: Optional[HardwareProfile] = None,
    model_path: str = "",
) -> dict:
    """Update *config* with hardware-aware batch / workers / cache.

    Args:
        config: A training preset dict (e.g. ``DEFECT_FULL_TRAIN_CONFIG``).
        profile: Pre-detected :class:`HardwareProfile`. Auto-detected if ``None``.
        model_path: Path to model YAML or .pt, used to infer model family
            for batch-size estimation.

    Returns:
        The modified dict (same object).
    """
    if profile is None:
        profile = HardwareProfile.detect()

    profile.print_info()

    # ── Workers ──
    config["workers"] = profile.recommended_workers

    # ── Cache ──
    config["cache"] = profile.recommended_cache

    # ── Batch size (VRAM-aware) ──
    imgsz = config.get("imgsz", 640)
    default_batch = config.get("batch", 16)

    # Account for multi_scale: reduce safety margin when imgsz varies per batch.
    # Peak imgsz = base * (1 + multi_scale); peak VRAM ≈ base_VRAM * (1 + ms)².
    # We size for the midpoint between base and peak to balance throughput vs OOM risk.
    multi_scale = float(config.get("multi_scale", 0.0))
    if multi_scale > 0:
        midpoint_factor = (1.0 + multi_scale / 2) ** 2  # VRAM scaling ~ pixel count
        safe_margin = max(0.35, 0.75 / midpoint_factor)  # floor 35% to avoid batch=0
    else:
        safe_margin = 0.75

    recommended_batch = profile.recommend_batch_size(
        model_path=model_path, imgsz=imgsz
    )
    # Apply multi_scale safety: reduce batch proportionally
    if multi_scale > 0:
        ms_batch = profile.estimate_batch_size(
            model_family=profile._infer_family(model_path),
            imgsz=imgsz,
            safety_margin=safe_margin,
        )
        recommended_batch = min(recommended_batch, ms_batch)
        logger.info("  → multi_scale=%.2f, safety_margin=%.0f%%, ms_aware_batch=%s",
                     multi_scale, safe_margin * 100, ms_batch)

    # Never go above what the preset specified as ceiling
    final_batch = min(recommended_batch, default_batch * 2)
    # But don't go below 4
    final_batch = max(final_batch, 4)
    config["batch"] = final_batch

    logger.info("  → batch=%s  workers=%s  cache=%s", final_batch, config["workers"], config["cache"])

    # ── OOM safeguard: warn if estimated VRAM usage is high ──
    if profile.vram_gb > 0:
        # Use peak imgsz for VRAM warning
        peak_imgsz = int(imgsz * (1.0 + multi_scale)) if multi_scale else imgsz
        gb_per_sample = HardwareProfile._VRAM_PER_SAMPLE.get(
            (profile._infer_family(model_path), peak_imgsz),
            HardwareProfile._VRAM_PER_SAMPLE.get(("yolo11s", 1024), 1.0),
        )
        est_vram = 6.0 + final_batch * gb_per_sample  # 6GB model overhead (EMA + AdamW + cuDNN)
        usage_pct = est_vram / profile.vram_gb * 100
        if usage_pct > 85:
            warnings.warn(
                f"Estimated VRAM usage: {est_vram:.1f} / {profile.vram_gb:.1f} GB "
                f"({usage_pct:.0f}%). "
                f"Consider reducing --batch or --imgsz if you see CUDA OOM errors.",
                stacklevel=2,
            )
        logger.info("  → Estimated VRAM (peak imgsz=%s): %.1f / %.1f GB (%.0f%%)",
                     peak_imgsz, est_vram, profile.vram_gb, usage_pct)

    return config


# ============================================================================
# YAML-based config loading
# ============================================================================

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def _load_yaml(filepath: Path) -> dict:
    """Load a YAML config file, returning an empty dict if missing."""
    if filepath.exists():
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_train_config(stage: str) -> dict:
    """Load training hyperparameters from ``config/train/<stage>.yaml``.

    Args:
        stage: One of ``"warmup"``, ``"full"``, ``"finetune"``.

    Returns:
        Config dict ready to unpack into ``YOLO.train(**config)``.

    Raises:
        FileNotFoundError: If the YAML file doesn't exist.
    """
    path = _CONFIG_DIR / "train" / f"{stage}.yaml"
    config = _load_yaml(path)
    if not config:
        raise FileNotFoundError(
            f"Training config not found: {path}\n"
            f"Expected YAML files in config/train/ "
            f"(warmup.yaml, full.yaml, finetune.yaml)"
        )
    return config


def load_inference_config() -> dict:
    """Load model inference/validation defaults from ``config/model/inference.yaml``.

    Returns:
        Inference config dict (empty if file missing).
    """
    return _load_yaml(_CONFIG_DIR / "model" / "inference.yaml")
