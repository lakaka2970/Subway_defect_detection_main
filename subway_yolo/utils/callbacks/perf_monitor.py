"""Performance monitoring callbacks for training optimization.

Records per-batch timing (data loading vs compute), per-epoch system metrics
(GPU utilization, VRAM, RAM, disk I/O), and writes structured logs for
post-hoc analysis and loop-driven optimization decisions.

Output files (under ``<save_dir>/perf_monitor/``):
    - ``metrics.jsonl`` — one JSON line per epoch with full metrics
    - ``status.json``   — current training state (for external loop readers)

Usage:
    from subway_yolo.utils.callbacks.perf_monitor import register_perf_monitor
    register_perf_monitor(trainer)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from subway_yolo.utils import LOGGER

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None


class PerfMonitor:
    """Collects timing and system metrics during training."""

    def __init__(self, save_dir: Path):
        self.dir = save_dir / "perf_monitor"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.metrics_file = self.dir / "metrics.jsonl"
        self.status_file = self.dir / "status.json"

        self._batch_start_time = 0.0
        self._epoch_start_time = 0.0
        self._batch_times: list[float] = []
        self._data_times: list[float] = []
        self._gpu_util_samples: list[float] = []
        self._prev_disk = None
        self._prev_time = 0.0
        self._train_start_time = 0.0
        self._pynvml = None
        self._nvml_handle = None

        self._system_logger = None
        try:
            from subway_yolo.utils.logger import SystemLogger
            self._system_logger = SystemLogger()
        except Exception:
            pass

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            pass

    def on_train_start(self, trainer):
        self._train_start_time = time.time()
        self._write_status(trainer, "training")

    def on_train_epoch_start(self, trainer):
        self._epoch_start_time = time.time()
        self._batch_times = []
        self._data_times = []
        self._gpu_util_samples = []
        if psutil:
            self._prev_disk = psutil.disk_io_counters()
            self._prev_time = time.time()

    def on_train_batch_start(self, trainer):
        self._batch_start_time = time.time()

    def on_train_batch_end(self, trainer):
        now = time.time()
        batch_elapsed = now - self._batch_start_time
        self._batch_times.append(batch_elapsed)
        if self._pynvml and self._nvml_handle:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                self._gpu_util_samples.append(util.gpu)
            except Exception:
                pass

    def on_fit_epoch_end(self, trainer):
        epoch_time = time.time() - self._epoch_start_time
        n_batches = len(self._batch_times)
        avg_batch = sum(self._batch_times) / max(n_batches, 1)

        metrics = {
            "epoch": trainer.epoch + 1,
            "total_epochs": trainer.epochs,
            "epoch_time_s": round(epoch_time, 2),
            "n_batches": n_batches,
            "avg_batch_ms": round(avg_batch * 1000, 1),
            "batch_size": trainer.batch_size,
            "imgsz": trainer.args.imgsz,
        }

        if torch and torch.cuda.is_available():
            metrics["vram_allocated_gb"] = round(torch.cuda.memory_allocated() / 1024**3, 2)
            metrics["vram_reserved_gb"] = round(torch.cuda.memory_reserved() / 1024**3, 2)
            props = torch.cuda.get_device_properties(0)
            metrics["vram_total_gb"] = round(props.total_memory / 1024**3, 1)

        if self._gpu_util_samples:
            metrics["gpu_util_avg_pct"] = round(sum(self._gpu_util_samples) / len(self._gpu_util_samples), 1)
            metrics["gpu_util_max_pct"] = max(self._gpu_util_samples)

        if self._system_logger:
            try:
                sys_metrics = self._system_logger.get_metrics(rates=True)
                metrics["cpu_pct"] = sys_metrics.get("cpu", 0)
                metrics["ram_pct"] = sys_metrics.get("ram", 0)
                disk = sys_metrics.get("disk", {})
                metrics["disk_read_mbs"] = disk.get("read_mbs", 0)
                metrics["disk_write_mbs"] = disk.get("write_mbs", 0)
                gpus = sys_metrics.get("gpus", {})
                if "0" in gpus:
                    gpu0 = gpus["0"]
                    metrics["gpu_temp_c"] = gpu0.get("temp", 0)
                    metrics["gpu_power_w"] = gpu0.get("power", 0)
            except Exception:
                pass

        if psutil:
            mem = psutil.virtual_memory()
            metrics["ram_used_gb"] = round(mem.used / 1024**3, 1)
            metrics["ram_total_gb"] = round(mem.total / 1024**3, 1)

        if hasattr(trainer, "tloss") and trainer.tloss is not None:
            tloss = trainer.tloss
            if hasattr(tloss, "mean"):
                metrics["train_loss"] = round(float(tloss.mean()), 4)

        if hasattr(trainer, "fitness") and trainer.fitness is not None:
            metrics["fitness"] = round(float(trainer.fitness), 4)

        with open(self.metrics_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        self._write_status(trainer, "training", metrics)

        ram_pct = metrics.get("ram_pct", 0)
        gpu_util = metrics.get("gpu_util_avg_pct", 0)
        if ram_pct > 50:
            LOGGER.warning(f"[PerfMon] RAM usage {ram_pct:.1f}% exceeds 50% target!")
        if gpu_util > 0 and gpu_util < 70:
            LOGGER.warning(f"[PerfMon] GPU utilization {gpu_util:.1f}% — possible data loading bottleneck")

    def on_train_end(self, trainer):
        total_time = time.time() - self._train_start_time
        self._write_status(trainer, "completed", {"total_train_time_s": round(total_time, 1)})

    def _write_status(self, trainer, state: str, extra: dict | None = None):
        status = {
            "state": state,
            "epoch": getattr(trainer, "epoch", 0) + 1,
            "total_epochs": getattr(trainer, "epochs", 0),
            "timestamp": time.time(),
        }
        if extra:
            status.update(extra)
        with open(self.status_file, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)


def register_perf_monitor(trainer):
    """Attach PerfMonitor callbacks to a trainer instance."""
    monitor = PerfMonitor(Path(trainer.save_dir))

    trainer.add_callback("on_train_start", monitor.on_train_start)
    trainer.add_callback("on_train_epoch_start", monitor.on_train_epoch_start)
    trainer.add_callback("on_train_batch_start", monitor.on_train_batch_start)
    trainer.add_callback("on_train_batch_end", monitor.on_train_batch_end)
    trainer.add_callback("on_fit_epoch_end", monitor.on_fit_epoch_end)
    trainer.add_callback("on_train_end", monitor.on_train_end)

    LOGGER.info(f"[PerfMon] Performance monitor registered → {monitor.dir}")
    return monitor
