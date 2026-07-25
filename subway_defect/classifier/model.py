"""
Lightweight state classifier model based on MobileNetV3-small.

Designed for second-stage verification of YOLO proposals. Classifies
128x128 crops into defect states (e.g., normal/missing for CBHPM,
or normal/missing/loose/ambiguous for VHBNM/VHBNL).

Architecture: MobileNetV3-small backbone (~2.5M params) + custom head.
Total model size < 5M parameters for fast inference (< 5ms per crop).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class StateClassifierHead(nn.Module):
    """Classification head with dropout and optional auxiliary regression."""

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class StateClassifier(nn.Module):
    """MobileNetV3-small based state classifier for defect verification.

    Args:
        num_classes: Number of output classes (2 for binary, 4 for state).
        pretrained: Use ImageNet-pretrained backbone. Default: True.
        dropout: Dropout rate in classification head. Default: 0.3.
        freeze_backbone: Freeze backbone layers for initial training.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes

        weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Remove the original classifier, keep features
        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # MobileNetV3-small last channel is 576
        last_channel = 576
        self.classifier = StateClassifierHead(last_channel, num_classes, dropout)

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, 128, 128) normalized.

        Returns:
            Logits tensor (B, num_classes).
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict class and confidence.

        Returns:
            (predicted_class, confidence) tensors.
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, dim=1)
        return predicted, confidence

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str | Path, meta: Optional[Dict] = None) -> None:
        """Save model checkpoint with metadata."""
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "num_classes": self.num_classes,
            "num_parameters": self.num_parameters,
        }
        if meta:
            checkpoint["meta"] = meta
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "StateClassifier":
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        num_classes = checkpoint.get("num_classes", 2)
        model = cls(num_classes=num_classes, pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model
