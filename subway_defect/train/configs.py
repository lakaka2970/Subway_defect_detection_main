"""
Training hyperparameter presets with hardware-aware auto-tuning.

Each dict unpacks into ``YOLO.train(**preset)``.

The :class:`HardwareProfile` class auto-detects GPU VRAM / CPU cores / RAM
and provides per-stage recommendations that balance throughput against OOM risk.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Dict, Optional, Tuple

import yaml


def _safe_load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return an empty dict for empty documents."""
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

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
        ("yolo11n", 640):  0.25,
        ("yolo11s", 640):  0.35,
        ("yolo11s", 1024): 0.80,
        ("yolo11s", 1280): 0.85,
        ("yolo11m", 640):  0.55,
        ("yolo11m", 1024): 1.10,
        ("yolo11m", 1280): 1.50,
        ("yolo11m-P2", 640):  0.70,
        ("yolo11m-P2", 1024): 1.40,
        # v2 (P2 + CoordAtt + LSK + DCNv4) — heavier than standard P2 variants
        ("yolo11m-P2-v2", 1024): 1.70,
        ("yolo11m-P2-v2", 1280): 2.00,
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
                profile.vram_gb = prop.total_memory / (1024 ** 3)
                # Fallback: some drivers / container runtimes report total_mem=0
                # even when CUDA is available. Use mem_get_info as backup.
                if profile.vram_gb <= 0:
                    free, total = torch.cuda.mem_get_info(0)
                    profile.vram_gb = total / (1024 ** 3)
        except Exception:
            pass

        # -- RAM --
        try:
            import psutil
            profile.ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            pass

        # -- Recommendations --
        # Workers: empirical testing on RTX 4090 + 64GB RAM shows 4 workers
        # saturates GPU (93.5% util) while keeping RAM < 50%. Higher worker
        # counts (8/16) inflate RAM to 61-71% with zero throughput gain.
        profile.recommended_workers = min(4, profile.cpu_cores)
        # Prefer disk cache to keep RAM under 50% target
        profile.recommended_cache = "disk"

        return profile

    def print_info(self) -> None:
        """Log detected hardware to stdout."""
        print(f"=== Hardware Profile ===")
        print(f"  GPU       : {self.gpu_name or 'N/A'}  ({self.vram_gb:.1f} GB VRAM)")
        print(f"  CPU cores : {self.cpu_cores}")
        print(f"  RAM       : {self.ram_gb:.1f} GB")
        print(f"  Workers   : {self.recommended_workers}")
        print(f"  Cache     : {self.recommended_cache}")
        print(f"=========================")

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
        # Fall back to nearest imgsz key
        if key not in self._VRAM_PER_SAMPLE:
            key = (model_family, 640) if (model_family, 640) in self._VRAM_PER_SAMPLE else ("yolo11s", 1024)

        gb_per_sample = self._VRAM_PER_SAMPLE[key]
        usable_vram = self.vram_gb * safety_margin

        # Reserve 4 GB for model weights + optimizer states + cuDNN workspace
        model_overhead = 4.0
        batch = int((usable_vram - model_overhead) / gb_per_sample)

        return max(min_batch, min(batch, max_batch))

    @staticmethod
    def _infer_family(model_path: str) -> str:
        """Infer model family string from a model path."""
        m = model_path.lower()
        if "yolo11n" in m:
            return "yolo11n"
        if "yolo11m" in m and "v2" in m:
            return "yolo11m-P2-v2"  # v2 = P2 + CoordAtt + LSK + DCNv4, heaviest
        if "yolo11m" in m and "p2" in m:
            return "yolo11m-P2"
        if "yolo11m" in m:
            return "yolo11m"
        if "yolo11s" in m and "v2" in m:
            return "yolo11s"  # use existing s-scale VRAM entries
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
    "workers": 16,             # overridden at runtime
    "cache": None,             # overridden at runtime — None lets auto-detect choose "ram" if RAM > 48GB
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
    "lr0": 0.001,              # Match C1 LR; cosine decay to 1e-5
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,    # Match C1 regularization
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
    "close_mosaic": 190,
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
    "copy_paste": 0.4,
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
    "close_mosaic": 0,
}


# ── Defect detector — Stage 5 v2 hard-negative fine-tune ─────────
#
# 2026-07-04 analysis showed that the first hard-negative run reduced raw
# detections but did not improve calibrated precision. The likely bottleneck
# was an overly conservative update budget: freeze[0-7] + lr=2e-5. This preset
# keeps the no-augmentation HN setting, but gives the neck/head and deeper
# backbone blocks enough freedom to learn the negative patterns.
DEFECT_HARD_NEGATIVE_V2_CONFIG: dict = {
    **_COMMON_BASE,
    **_DATALOADER,
    "amp": False,
    "cache": False,
    "epochs": 20,
    "imgsz": 1280,
    "batch": 4,
    "optimizer": "AdamW",
    "lr0": 5e-5,
    "lrf": 1.0,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 0,
    "warmup_momentum": 0.8,
    "warmup_bias_lr": 5e-5,
    "patience": 6,
    "save_period": 5,
    "cos_lr": False,
    "mosaic": 0.0,
    "mixup": 0.0,
    "cutmix": 0.0,
    "copy_paste": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    "degrees": 0.0,
    "translate": 0.0,
    "scale": 0.0,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "close_mosaic": 10,
    "freeze": [0, 1, 2, 3],
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

    # Respect explicit cache settings, e.g. Stage 5 v2 disables RAM cache to
    # avoid large image-cache spikes on 1280px crops.
    if config.get("cache") is None:
        config["cache"] = profile.recommended_cache

    # ── Batch size (VRAM-aware) ──
    imgsz = config.get("imgsz", 640)
    default_batch = config.get("batch", 16)
    recommended_batch = profile.recommend_batch_size(
        model_path=model_path, imgsz=imgsz
    )
    # Never go above what the preset specified as ceiling
    final_batch = min(recommended_batch, default_batch * 2)
    # But don't go below 4
    final_batch = max(final_batch, 4)
    config["batch"] = final_batch

    print(f"  → batch={final_batch}  workers={config['workers']}  "
          f"cache={config['cache']}")

    # ── OOM safeguard: warn if estimated VRAM usage is high ──
    if profile.vram_gb > 0:
        gb_per_sample = HardwareProfile._VRAM_PER_SAMPLE.get(
            (profile._infer_family(model_path), imgsz),
            HardwareProfile._VRAM_PER_SAMPLE.get(("yolo11s", 1024), 1.0),
        )
        est_vram = 2.0 + final_batch * gb_per_sample  # 2GB model overhead
        usage_pct = est_vram / profile.vram_gb * 100
        if usage_pct > 85:
            warnings.warn(
                f"Estimated VRAM usage: {est_vram:.1f} / {profile.vram_gb:.1f} GB "
                f"({usage_pct:.0f}%). "
                f"Consider reducing --batch or --imgsz if you see CUDA OOM errors.",
                stacklevel=2,
            )
        print(f"  → Estimated VRAM: {est_vram:.1f} / {profile.vram_gb:.1f} GB "
              f"({usage_pct:.0f}%)")

    return config
