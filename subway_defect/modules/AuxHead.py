"""Auxiliary component-type classification head for multi-task training.

Attaches to backbone P3 features and predicts which component types are
present in the image (multi-label).  This provides a structural learning
signal that helps the backbone learn discriminative component features,
improving downstream defect classification.

Usage::

    # In DetectionModel (auto-created when aux_head config is present)
    model = DetectionModel(cfg, nc=12)
    model.aux_head  # → AuxClassifyHead(in_channels=256, num_classes=9)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AuxClassifyHead(nn.Module):
    """Lightweight multi-label classification head for component types.

    Architecture: GAP → FC → BN → ReLU → Dropout → FC
    """

    def __init__(self, in_channels: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        mid = max(in_channels // 2, 128)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, mid),
            nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mid, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: (B, C, H, W) feature map → (B, num_classes) logits."""
        return self.head(x)
