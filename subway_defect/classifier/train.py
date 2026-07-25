"""
Training logic for the state classifier.

Supports binary (CBHPM: normal/missing) and multi-class (VHBNM/VHBNL:
normal/missing/loose/ambiguous) classification with:
- CosineAnnealing LR schedule
- EarlyStopping on validation loss
- Class-weighted loss for imbalanced data
- Best model checkpointing by macro-F1
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np

from .model import StateClassifier


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


def compute_metrics(
    all_preds: np.ndarray, all_labels: np.ndarray, num_classes: int
) -> Dict[str, float]:
    """Compute per-class and macro metrics."""
    metrics = {}
    per_class_f1 = []

    for c in range(num_classes):
        tp = np.sum((all_preds == c) & (all_labels == c))
        fp = np.sum((all_preds == c) & (all_labels != c))
        fn = np.sum((all_preds != c) & (all_labels == c))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[f"class_{c}_precision"] = precision
        metrics[f"class_{c}_recall"] = recall
        metrics[f"class_{c}_f1"] = f1
        per_class_f1.append(f1)

    metrics["macro_f1"] = np.mean(per_class_f1)
    metrics["accuracy"] = np.mean(all_preds == all_labels)
    return metrics


def train_classifier(
    model: StateClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    device: str = "0",
    save_path: Optional[str | Path] = None,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Train the state classifier.

    Args:
        model: StateClassifier model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        epochs: Maximum training epochs.
        lr: Initial learning rate.
        weight_decay: Weight decay for optimizer.
        patience: Early stopping patience.
        device: CUDA device string.
        save_path: Path to save best model checkpoint.
        class_names: Class names for logging.

    Returns:
        Best validation metrics dict.
    """
    if device not in ("", "cpu") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(f"cuda:{device}" if device not in ("", "cpu") else "cpu")
    model = model.to(dev)

    # Class-weighted loss
    num_classes = model.num_classes
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    early_stop = EarlyStopping(patience=patience)
    best_metrics: Dict[str, float] = {}
    best_f1 = 0.0

    print(f"  Training: {model.num_trainable_parameters:,} trainable params")
    print(f"  Epochs: {epochs}, LR: {lr}, Patience: {patience}")
    print(f"  Device: {dev}")
    print()

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(dev), labels.to(dev)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(dev), labels.to(dev)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                val_total += labels.size(0)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss /= val_total
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        metrics = compute_metrics(all_preds, all_labels, num_classes)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        print(
            f"  Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} macro_f1={metrics['macro_f1']:.4f} "
            f"acc={metrics['accuracy']:.3f} | "
            f"lr={lr_now:.6f} | {elapsed:.1f}s"
        )

        # Per-class F1
        if class_names:
            f1_strs = [
                f"{class_names[c]}={metrics.get(f'class_{c}_f1', 0):.3f}"
                for c in range(num_classes)
            ]
            print(f"           F1: {' | '.join(f1_strs)}")

        # Checkpoint best model
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            best_metrics = metrics.copy()
            best_metrics["epoch"] = epoch
            if save_path:
                model.save(save_path, meta={
                    "class_names": class_names,
                    "best_epoch": epoch,
                    "best_macro_f1": best_f1,
                })
                print(f"           → Saved best model (macro_f1={best_f1:.4f})")

        # Early stopping
        if early_stop(-val_loss):
            print(f"\n  Early stopping at epoch {epoch} (patience={patience})")
            break

    print(f"\n  Best macro-F1: {best_f1:.4f} at epoch {best_metrics.get('epoch', '?')}")
    return best_metrics
