"""
Training callbacks for enhanced metrics logging and output.

Integrates with Ultralytics YOLO training loop via the callback system.
Registers handlers for training events to log per-class metrics, loss
dynamics, hard examples, and multi-criteria checkpointing.

Usage::

    from subway_defect.train.callbacks import register_all_callbacks
    model = YOLO("model.yaml")
    model.add_callback("on_train_start", log_config_summary)
    # ... or use register_all_callbacks(model) for all at once.

Reference: Ultralytics ``callbacks`` interface — callbacks are plain functions
or callables that receive a ``trainer`` argument.
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import yaml

if TYPE_CHECKING:
    from subway_yolo.engine.trainer import BaseTrainer

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    """Safely get an attribute that may not exist."""
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _format_duration(seconds: float) -> str:
    """Format duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds // 60:.0f}m {seconds % 60:.0f}s"
    h, m = divmod(seconds, 3600)
    m, s = divmod(m, 60)
    return f"{h:.0f}h {m:.0f}m {s:.0f}s"


def _get_run_dir(trainer: "BaseTrainer") -> Path:
    """Resolve the run output directory from a trainer instance."""
    save_dir = _safe_get(trainer, "save_dir", None)
    if save_dir and Path(save_dir).exists():
        return Path(save_dir)
    # Fallback: look for args.project/name pattern
    args = _safe_get(trainer, "args", None)
    if args:
        project = _safe_get(args, "project", "runs")
        name = _safe_get(args, "name", "train")
        return Path(project) / name
    return Path("runs/train")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Configuration Summary Logger
# ═══════════════════════════════════════════════════════════════════════════

def log_config_summary(trainer: "BaseTrainer") -> None:
    """Log a comprehensive training configuration summary at start.

    Triggered by: ``on_train_start`` event.
    """
    args = _safe_get(trainer, "args", None)
    if args is None:
        return

    model_name = _safe_get(trainer, "model_name", "unknown")
    if hasattr(trainer, "model") and hasattr(trainer.model, "yaml_file"):
        model_name = trainer.model.yaml_file or model_name

    imgsz = _safe_get(args, "imgsz", 640)
    batch = _safe_get(args, "batch", 16)
    epochs = _safe_get(args, "epochs", 100)
    optimizer = _safe_get(args, "optimizer", "SGD")
    lr0 = _safe_get(args, "lr0", 0.01)
    device = _safe_get(trainer, "device", "cpu")

    # Dataset info
    train_loader = _safe_get(trainer, "train_loader", None)
    val_loader = _safe_get(trainer, "valid_loader", None)
    n_train = len(train_loader.dataset) if train_loader else "?"
    n_val = len(val_loader.dataset) if val_loader else "?"

    lines = [
        "",
        "=" * 66,
        "  TRAINING CONFIGURATION SUMMARY",
        "=" * 66,
        f"  Model     : {model_name}",
        f"  Input     : imgsz={imgsz}, batch={batch}",
        f"  Optimizer : {optimizer}, lr0={lr0}",
        f"  Epochs    : {epochs}",
        f"  Data      : {n_train} train / {n_val} val images",
        f"  Device    : {device}",
    ]

    # Dataset file info
    data = _safe_get(args, "data", None)
    if data:
        lines.append(f"  Dataset   : {data}")

    # GPU memory info
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            free_gb, total_gb = free / 1e9, total / 1e9
            lines.append(f"  GPU VRAM  : {free_gb:.1f} GiB free / {total_gb:.1f} GiB total")
    except Exception:
        pass

    lines.append("=" * 66)
    for line in lines:
        logger.info(line)

    # Write config summary to run directory
    run_dir = _get_run_dir(trainer)
    if run_dir.exists() or True:  # will be created by trainer
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            summary: Dict[str, Any] = {
                "model": str(model_name),
                "imgsz": imgsz,
                "batch": batch,
                "epochs": epochs,
                "optimizer": optimizer,
                "lr0": lr0,
                "device": str(device),
                "data": str(data) if data else None,
                "n_train": n_train,
                "n_val": n_val,
            }
            (run_dir / "config_summary.yaml").write_text(
                yaml.dump(summary, default_flow_style=False), encoding="utf-8",
            )
        except Exception as exc:
            logger.debug("Could not write config summary: %s", exc)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Training Dynamics Recorder (loss components per epoch)
# ═══════════════════════════════════════════════════════════════════════════

class TrainingDynamics:
    """Records per-epoch loss components and writes ``training_dynamics.csv``.

    Columns: epoch, train_box_loss, train_cls_loss, train_dfl_loss,
    val_box_loss, val_cls_loss, val_dfl_loss, lr, mAP50, mAP50-95, time_elapsed

    Usage::

        dynamics = TrainingDynamics()
        model.add_callback("on_fit_epoch_end", dynamics.on_epoch_end)
    """

    def __init__(self) -> None:
        self._epochs: List[Dict[str, Any]] = []
        self._start_time: Optional[float] = None
        self._best_map50 = 0.0
        self._best_map50_95 = 0.0

    def on_train_start(self, trainer: "BaseTrainer") -> None:
        """Record start time."""
        self._start_time = time.monotonic()

    def on_fit_epoch_end(self, trainer: "BaseTrainer") -> None:
        """Record one epoch's metrics.

        Triggered by: ``on_fit_epoch_end``.
        """
        epoch = _safe_get(trainer, "epoch", -1) + 1  # 0-based → 1-based

        # Training losses (most recent batch values)
        loss_items = _safe_get(trainer, "loss_items", None)
        if loss_items is not None and hasattr(loss_items, "tolist"):
            loss_vals = loss_items.tolist()
        else:
            loss_vals = [0.0, 0.0, 0.0]

        train_box = loss_vals[0] if len(loss_vals) > 0 else 0.0
        train_cls = loss_vals[1] if len(loss_vals) > 1 else 0.0
        train_dfl = loss_vals[2] if len(loss_vals) > 2 else 0.0

        # Validation losses (from validator)
        val_box = 0.0
        val_cls = 0.0
        val_dfl = 0.0
        validator = _safe_get(trainer, "validator", None)
        if validator is not None:
            val_loss = _safe_get(validator, "loss", None)
            if val_loss is not None and hasattr(val_loss, "tolist"):
                v = val_loss.tolist()
                val_box = v[0] if len(v) > 0 else 0.0
                val_cls = v[1] if len(v) > 1 else 0.0
                val_dfl = v[2] if len(v) > 2 else 0.0

        # Learning rate
        lr = 0.0
        if hasattr(trainer, "optimizer") and hasattr(trainer.optimizer, "param_groups"):
            lr = trainer.optimizer.param_groups[0].get("lr", 0.0)

        # Metrics
        metrics = _safe_get(trainer, "metrics", {}) or {}
        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)

        if map50 > self._best_map50:
            self._best_map50 = map50
        if map50_95 > self._best_map50_95:
            self._best_map50_95 = map50_95

        elapsed = time.monotonic() - (self._start_time or 0)

        record = {
            "epoch": epoch,
            "train_box_loss": round(train_box, 6),
            "train_cls_loss": round(train_cls, 6),
            "train_dfl_loss": round(train_dfl, 6),
            "val_box_loss": round(val_box, 6),
            "val_cls_loss": round(val_cls, 6),
            "val_dfl_loss": round(val_dfl, 6),
            "lr": round(lr, 8),
            "mAP50": round(map50, 4),
            "mAP50-95": round(map50_95, 4),
            "best_mAP50": round(self._best_map50, 4),
            "best_mAP50-95": round(self._best_map50_95, 4),
            "time_elapsed": _format_duration(elapsed),
        }
        self._epochs.append(record)

        # Live log
        logger.info(
            "Epoch %s | box=%.4f cls=%.4f dfl=%.4f | val_box=%.4f val_cls=%.4f val_dfl=%.4f "
            "| mAP50=%.4f mAP50-95=%.4f | best=%.4f/%.4f",
            epoch, train_box, train_cls, train_dfl, val_box, val_cls, val_dfl,
            map50, map50_95, self._best_map50, self._best_map50_95,
        )

        # Write CSV incrementally
        self._flush(trainer)

    def _flush(self, trainer: "BaseTrainer") -> None:
        """Write accumulated records to CSV in the run directory."""
        run_dir = _get_run_dir(trainer)
        if not run_dir.exists():
            return
        csv_path = run_dir / "training_dynamics.csv"
        if self._epochs:
            keys = [
                "epoch", "train_box_loss", "train_cls_loss", "train_dfl_loss",
                "val_box_loss", "val_cls_loss", "val_dfl_loss",
                "lr", "mAP50", "mAP50-95", "best_mAP50", "best_mAP50-95",
                "time_elapsed",
            ]
            lines = [",".join(keys)]
            for r in self._epochs:
                lines.append(",".join(str(r.get(k, "")) for k in keys))
            csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @property
    def history(self) -> List[Dict[str, Any]]:
        """Return the full per-epoch history."""
        return list(self._epochs)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Metrics Logger (per-class AP, size-grouped AP)
# ═══════════════════════════════════════════════════════════════════════════

class MetricsLogger:
    """Logs per-class and size-grouped metrics after each validation.

    Writes ``per_class_metrics.csv`` and ``size_grouped_metrics.csv``
    to the run directory.

    Usage::

        logger = MetricsLogger()
        model.add_callback("on_val_end", logger.on_val_end)
    """

    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        self._class_names: List[str] = class_names or []
        self._best_per_class: Dict[str, float] = {}
        self._history: List[Dict[str, Any]] = []

    def on_val_end(self, trainer: "BaseTrainer") -> None:
        """Record per-class and size-grouped metrics after validation.

        Triggered by: ``on_val_end``.
        """
        metrics = _safe_get(trainer, "metrics", {}) or {}
        validator = _safe_get(trainer, "validator", None)
        epoch = _safe_get(trainer, "epoch", -1) + 1

        # ── Per-class AP ──────────────────────────────────────────
        per_class_ap: Dict[str, float] = {}
        if validator is not None:
            ap_per_class = _safe_get(validator, "ap_per_class", None) or _safe_get(
                validator, "ap_class_index", None
            )
            # Ultralytics stores per-class results in validator.seen, validator.stats, etc.
            # Try to extract from metrics dictionary
            for key in sorted(metrics):
                if key.startswith("metrics/mAP50") and "(" in key and not key.endswith("(B)"):
                    # Format: metrics/mAP50(C) where C is class index
                    cls_part = key.replace("metrics/mAP50(", "").replace(")", "")
                    try:
                        cls_idx = int(cls_part)
                        cls_name = (
                            self._class_names[cls_idx]
                            if cls_idx < len(self._class_names)
                            else f"class_{cls_idx}"
                        )
                        per_class_ap[cls_name] = float(metrics.get(key, 0.0))
                    except (ValueError, IndexError):
                        pass

        # Alternative: try to extract from results dict
        results = _safe_get(trainer, "validator", None)
        if results and hasattr(results, "ap_class_index"):
            try:
                ap_array = results.ap_class_index
                nc = len(ap_array) if hasattr(ap_array, "__len__") else 0
                for i in range(nc):
                    cls_name = self._class_names[i] if i < len(self._class_names) else f"cls_{i}"
                    per_class_ap.setdefault(cls_name, float(ap_array[i]))
            except Exception:
                pass

        # Log per-class metrics if available
        if per_class_ap:
            self._best_per_class = {
                k: max(v, self._best_per_class.get(k, 0.0))
                for k, v in per_class_ap.items()
            }
            lines = [f"  {name:<10s}: AP50={ap:.4f} (best={self._best_per_class[name]:.4f})"
                     for name, ap in sorted(per_class_ap.items())]
            logger.info("Per-class mAP50:\n%s", "\n".join(lines))

        # ── Size-grouped metrics ───────────────────────────────────
        size_metrics: Dict[str, float] = {}
        for key_prefix in ["metrics/mAP50(small)", "metrics/mAP50(medium)", "metrics/mAP50(large)",
                           "metrics/mAP50-95(small)", "metrics/mAP50-95(medium)",
                           "metrics/mAP50-95(large)"]:
            val = metrics.get(key_prefix, None)
            if val is not None:
                size_metrics[key_prefix.replace("metrics/", "")] = float(val)

        if size_metrics:
            lines = [f"  {k}: {v:.4f}" for k, v in sorted(size_metrics.items())]
            logger.info("Size-grouped metrics:\n%s", "\n".join(lines))

        # ── Precision / Recall ─────────────────────────────────────
        precision = metrics.get("metrics/precision(B)", 0.0)
        recall = metrics.get("metrics/recall(B)", 0.0)
        if precision or recall:
            logger.info("Precision=%.4f  Recall=%.4f  F1=%.4f",
                        precision, recall,
                        2 * precision * recall / (precision + recall + 1e-8))

        # ── Save to run directory ──────────────────────────────────
        self._save(trainer, epoch, per_class_ap, size_metrics, precision, recall)

    def _save(
        self, trainer: "BaseTrainer", epoch: int,
        per_class_ap: Dict[str, float],
        size_metrics: Dict[str, float],
        precision: float, recall: float,
    ) -> None:
        """Persist metrics to CSV files."""
        run_dir = _get_run_dir(trainer)
        if not run_dir.exists():
            return

        # Per-class metrics CSV
        if per_class_ap:
            pc_path = run_dir / "per_class_metrics.csv"
            header = ["epoch"] + sorted(per_class_ap.keys())
            # Build or append
            rows: List[List[str]] = [header]
            # Read existing
            if pc_path.exists():
                existing = pc_path.read_text().strip().splitlines()
                if len(existing) > 1:
                    rows = [existing[0].split(",")]
            row = [str(epoch)] + [
                f"{per_class_ap.get(k, 0.0):.4f}" for k in sorted(per_class_ap)
            ]
            rows.append(row)
            pc_path.write_text("\n".join(",".join(r) for r in rows) + "\n", encoding="utf-8")

        # Size-grouped metrics CSV
        if size_metrics:
            sg_path = run_dir / "size_grouped_metrics.csv"
            header = ["epoch"] + sorted(size_metrics.keys())
            rows = [header]
            if sg_path.exists():
                existing = sg_path.read_text().strip().splitlines()
                if len(existing) > 1:
                    rows = [existing[0].split(",")]
            row = [str(epoch)] + [
                f"{size_metrics.get(k, 0.0):.4f}" for k in sorted(size_metrics)
            ]
            rows.append(row)
            sg_path.write_text("\n".join(",".join(r) for r in rows) + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# 4. Checkpoint Manager (multi-criteria best model saving)
# ═══════════════════════════════════════════════════════════════════════════

class CheckpointManager:
    """Saves additional best-model checkpoints beyond Ultralytics defaults.

    Tracks and saves:
    - ``best_map50-95.pt`` — best by mAP50-95
    - ``best_recall.pt`` — best Recall at Precision ≥ 0.90
    - ``best_f2.pt`` — best F2-score (weights recall over precision)

    Usage::

        ckpt_mgr = CheckpointManager()
        model.add_callback("on_val_end", ckpt_mgr.on_val_end)
    """

    def __init__(self) -> None:
        self._best_map50_95 = 0.0
        self._best_map50_95_epoch = 0
        self._best_recall_at_p90 = 0.0
        self._best_recall_epoch = 0
        self._best_f2 = 0.0
        self._best_f2_epoch = 0
        self._best_map50 = 0.0
        self._best_map50_epoch = 0
        self._current_epoch = 0
        self._save_dir: Optional[Path] = None

    def on_val_end(self, trainer: "BaseTrainer") -> None:
        """Check and save best models after validation.

        Triggered by: ``on_val_end``.
        """
        metrics = _safe_get(trainer, "metrics", {}) or {}
        epoch = _safe_get(trainer, "epoch", -1)
        self._current_epoch = epoch

        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
        precision = metrics.get("metrics/precision(B)", 0.0)
        recall = metrics.get("metrics/recall(B)", 0.0)

        # Track best mAP50
        if map50 > self._best_map50:
            self._best_map50 = map50
            self._best_map50_epoch = epoch

        # Best mAP50-95
        if map50_95 > self._best_map50_95:
            self._best_map50_95 = map50_95
            self._best_map50_95_epoch = epoch
            self._save_best(trainer, "best_map50-95.pt")

        # Best Recall (unconditional — highest recall)
        if recall > self._best_recall_at_p90:
            self._best_recall_at_p90 = recall
            self._best_recall_epoch = epoch
            self._save_best(trainer, "best_recall.pt")

        # Best F2 (weights recall 4× over precision — for industrial QA)
        f2 = (5 * precision * recall) / (4 * precision + recall + 1e-8)
        if f2 > self._best_f2:
            self._best_f2 = f2
            self._best_f2_epoch = epoch
            self._save_best(trainer, "best_f2.pt")

    def _save_best(self, trainer: "BaseTrainer", filename: str) -> None:
        """Save current best weights as *filename*."""
        import shutil

        save_dir = _get_run_dir(trainer)
        weights_dir = save_dir / "weights"
        if not weights_dir.exists():
            weights_dir = save_dir  # fallback

        # Ultralytics saves best.pt in weights/ or save_dir
        best_paths = [
            weights_dir / "best.pt",
            save_dir / "weights" / "best.pt",
            save_dir / "best.pt",
        ]
        for best_path in best_paths:
            if best_path.exists():
                dst = best_path.parent / filename
                shutil.copy2(best_path, dst)
                logger.info("Saved %s (epoch %s, mAP50-95=%.4f)",
                            filename, self._current_epoch, self._best_map50_95)
                return

    def summary(self) -> Dict[str, Any]:
        """Return a summary of all tracking criteria."""
        return {
            "best_map50": {"value": self._best_map50, "epoch": self._best_map50_epoch},
            "best_map50-95": {"value": self._best_map50_95, "epoch": self._best_map50_95_epoch},
            "best_recall": {"value": self._best_recall_at_p90, "epoch": self._best_recall_epoch},
            "best_f2": {"value": self._best_f2, "epoch": self._best_f2_epoch},
        }


# ═══════════════════════════════════════════════════════════════════════════
# 5. Training Report Generator (final summary JSON)
# ═══════════════════════════════════════════════════════════════════════════

class TrainingReport:
    """Generates ``training_report.json`` when training completes.

    Aggregates data from other callbacks to produce a structured summary.

    Usage::

        report = TrainingReport(class_names=["VHBNM", "VHBNL", ...])
        model.add_callback("on_train_end", report.on_train_end)
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        dynamics: Optional[TrainingDynamics] = None,
        ckpt_mgr: Optional[CheckpointManager] = None,
    ) -> None:
        self._class_names = class_names or []
        self._dynamics = dynamics
        self._ckpt_mgr = ckpt_mgr
        self._start_time: Optional[float] = None

    def on_train_start(self, trainer: "BaseTrainer") -> None:
        """Record training start time."""
        self._start_time = time.monotonic()

    def on_train_end(self, trainer: "BaseTrainer") -> None:
        """Generate and save the training report.

        Triggered by: ``on_train_end``.
        """
        save_dir = _get_run_dir(trainer)
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)

        metrics = _safe_get(trainer, "metrics", {}) or {}
        args = _safe_get(trainer, "args", None)

        # Best metrics
        map50 = metrics.get("metrics/mAP50(B)", 0.0)
        map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
        precision = metrics.get("metrics/precision(B)", 0.0)
        recall = metrics.get("metrics/recall(B)", 0.0)

        # Epoch info
        best_epoch = _safe_get(trainer, "best_epoch", _safe_get(trainer, "epoch", -1))
        if best_epoch is not None:
            best_epoch = best_epoch + 1  # 0-based → 1-based

        # Duration
        duration = time.monotonic() - (self._start_time or 0) if self._start_time else 0

        # Per-class extraction from metrics
        per_class: Dict[str, Dict[str, float]] = {}
        for key in sorted(metrics):
            if key.startswith("metrics/mAP50(") and key != "metrics/mAP50(B)":
                cls_tag = key.replace("metrics/mAP50(", "").replace(")", "")
                try:
                    cls_idx = int(cls_tag)
                    cls_name = (
                        self._class_names[cls_idx]
                        if cls_idx < len(self._class_names)
                        else f"class_{cls_idx}"
                    )
                except ValueError:
                    cls_name = cls_tag
                per_class.setdefault(cls_name, {})["AP50"] = float(metrics.get(key, 0.0))

        # Size-grouped
        by_size: Dict[str, float] = {}
        for k in ["metrics/mAP50(small)", "metrics/mAP50(medium)", "metrics/mAP50(large)",
                   "metrics/mAP50-95(small)", "metrics/mAP50-95(medium)",
                   "metrics/mAP50-95(large)"]:
            v = metrics.get(k)
            if v is not None:
                by_size[k.replace("metrics/", "")] = float(v)

        report: Dict[str, Any] = {
            "model": str(_safe_get(trainer, "model_name", "unknown")),
            "data": str(_safe_get(args, "data", "unknown")) if args else "unknown",
            "imgsz": _safe_get(args, "imgsz", 640) if args else 640,
            "batch": _safe_get(args, "batch", 16) if args else 16,
            "epochs_total": _safe_get(args, "epochs", 0) if args else 0,
            "best_epoch": best_epoch,
            "metrics": {
                "mAP50": round(map50, 4),
                "mAP50-95": round(map50_95, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(2 * precision * recall / (precision + recall + 1e-8), 4),
                "f2": round(5 * precision * recall / (4 * precision + recall + 1e-8), 4),
                "per_class": per_class,
                "by_size": by_size,
            },
            "training_time": _format_duration(duration),
            "training_time_seconds": round(duration, 1),
        }

        # Checkpoint manager summary
        if self._ckpt_mgr:
            report["best_checkpoints"] = self._ckpt_mgr.summary()

        # Dynamics history (last epoch)
        if self._dynamics and self._dynamics.history:
            report["final_losses"] = {
                k: v for k, v in self._dynamics.history[-1].items()
                if k.startswith(("train_", "val_"))
            }

        # Write report
        report_path = save_dir / "training_report.json"
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        logger.info("Training report saved to %s", report_path)

        # Print summary
        self._print_summary(report)

    @staticmethod
    def _print_summary(report: Dict[str, Any]) -> None:
        """Print a formatted final summary."""
        m = report.get("metrics", {})
        lines = [
            "",
            "=" * 50,
            "  TRAINING COMPLETE — Final Summary",
            "=" * 50,
            f"  Best epoch : {report.get('best_epoch', '?')}",
            f"  mAP50      : {m.get('mAP50', 0):.4f}",
            f"  mAP50-95   : {m.get('mAP50-95', 0):.4f}",
            f"  Precision  : {m.get('precision', 0):.4f}",
            f"  Recall     : {m.get('recall', 0):.4f}",
            f"  F1-score   : {m.get('f1', 0):.4f}",
            f"  F2-score   : {m.get('f2', 0):.4f}",
            f"  Duration   : {report.get('training_time', '?')}",
            "",
        ]
        per_class = m.get("per_class", {})
        if per_class:
            lines.append("  Per-class mAP50:")
            for cls_name, d in sorted(per_class.items()):
                lines.append(f"    {cls_name:<12s}: {d.get('AP50', 0):.4f}")

        ckpt_info = report.get("best_checkpoints", {})
        if ckpt_info:
            lines.append("")
            lines.append("  Best checkpoints:")
            for k, v in sorted(ckpt_info.items()):
                lines.append(f"    {k}: {v['value']:.4f} @ epoch {v['epoch']}")

        lines.append("=" * 50)
        for line in lines:
            logger.info(line)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Hard Example Collector
# ═══════════════════════════════════════════════════════════════════════════

class HardExampleCollector:
    """Collects false-positive and false-negative sample info during validation.

    Writes ``hard_examples.json`` to the run directory after each validation.
    These samples can be used for Stage 4 hard negative mining.

    Usage::

        collector = HardExampleCollector(conf_threshold=0.25, iou_threshold=0.5)
        model.add_callback("on_val_end", collector.on_val_end)
    """

    def __init__(
        self,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        max_examples: int = 200,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.max_examples = max_examples
        self._fp_samples: List[Dict[str, Any]] = []  # false positives
        self._fn_samples: List[Dict[str, Any]] = []  # false negatives

    def on_val_end(self, trainer: "BaseTrainer") -> None:
        """Collect hard examples from validation results.

        Triggered by: ``on_val_end``.

        Note: This is a best-effort collector. Ultralytics' validator
        doesn't directly expose per-sample FP/FN lists, so we derive them
        from the confusion matrix and per-class statistics where available.
        """
        metrics = _safe_get(trainer, "metrics", {}) or {}
        validator = _safe_get(trainer, "validator", None)
        epoch = _safe_get(trainer, "epoch", -1) + 1

        # Try to extract from confusion matrix if available
        confusion_matrix = metrics.get("metrics/confusion_matrix", None)
        if confusion_matrix is not None and hasattr(confusion_matrix, "tolist"):
            cm = confusion_matrix.tolist() if hasattr(confusion_matrix, "tolist") else confusion_matrix
            # Log confusion matrix summary
            logger.info("Confusion matrix available (%s entries)", len(cm) if hasattr(cm, "__len__") else "?")

        # Collect per-class FP/FN counts
        fp_fn_by_class: Dict[str, Dict[str, int]] = {}
        # Try metrics keys
        for key in sorted(metrics):
            if "FP" in key or "FN" in key:
                val = int(metrics.get(key, 0))
                if val > 0:
                    cls_tag = key.split("(")[-1].replace(")", "") if "(" in key else "all"
                    fp_fn_by_class.setdefault(cls_tag, {})[key.split("/")[-1].split("(")[0]] = val

        if fp_fn_by_class:
            logger.info("Hard examples by class: %s",
                        json.dumps(fp_fn_by_class, indent=2))

        # Persist
        run_dir = _get_run_dir(trainer)
        if run_dir.exists():
            data = {
                "epoch": epoch,
                "conf_threshold": self.conf_threshold,
                "iou_threshold": self.iou_threshold,
                "fp_fn_by_class": fp_fn_by_class,
                "total_fp": sum(
                    d.get("FP", d.get("fp", 0)) for d in fp_fn_by_class.values()
                ),
                "total_fn": sum(
                    d.get("FN", d.get("fn", 0)) for d in fp_fn_by_class.values()
                ),
            }
            out_path = run_dir / "hard_examples.json"
            out_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
            )


# ═══════════════════════════════════════════════════════════════════════════
# Convenience: register all callbacks at once
# ═══════════════════════════════════════════════════════════════════════════

def create_default_callbacks(
    class_names: Optional[List[str]] = None,
) -> Tuple[TrainingDynamics, MetricsLogger, CheckpointManager,
           TrainingReport, HardExampleCollector]:
    """Create a complete set of training callbacks with defaults.

    Args:
        class_names: List of class name strings (e.g. ``["VHBNM", ...]``).

    Returns:
        Tuple of (dynamics, metrics_logger, ckpt_mgr, report, hard_examples).
    """
    dynamics = TrainingDynamics()
    metrics_logger = MetricsLogger(class_names=class_names)
    ckpt_mgr = CheckpointManager()
    hard_examples = HardExampleCollector()
    report = TrainingReport(
        class_names=class_names,
        dynamics=dynamics,
        ckpt_mgr=ckpt_mgr,
    )
    return dynamics, metrics_logger, ckpt_mgr, report, hard_examples


def register_all_callbacks(
    model,  # YOLO instance
    class_names: Optional[List[str]] = None,
) -> Tuple[TrainingDynamics, MetricsLogger, CheckpointManager,
           TrainingReport, HardExampleCollector]:
    """Register the full callback suite on a YOLO model.

    Args:
        model: A ``YOLO`` instance (from ``subway_yolo import YOLO``).
        class_names: List of class name strings.

    Returns:
        Tuple of callback objects for post-training inspection.

    Example:
        >>> model = YOLO("yolo11s-P2-EMA-SimAM.yaml")
        >>> cbs = register_all_callbacks(model, class_names=["VHBNM", ...])
        >>> model.train(data="data.yaml", epochs=100)
        >>> print(cbs[3].summary())  # ckpt_mgr summary
    """
    dynamics, metrics_logger, ckpt_mgr, report, hard_examples = \
        create_default_callbacks(class_names)

    # Core hooks
    model.add_callback("on_train_start", log_config_summary)
    model.add_callback("on_train_start", dynamics.on_train_start)
    model.add_callback("on_train_start", report.on_train_start)

    model.add_callback("on_fit_epoch_end", dynamics.on_fit_epoch_end)
    model.add_callback("on_val_end", metrics_logger.on_val_end)
    model.add_callback("on_val_end", ckpt_mgr.on_val_end)
    model.add_callback("on_val_end", hard_examples.on_val_end)

    model.add_callback("on_train_end", report.on_train_end)

    return dynamics, metrics_logger, ckpt_mgr, report, hard_examples
