"""
Large Selective Kernel (LSK) Module.

Reference: Large Selective Kernel Network for Remote Sensing Object
Detection (ICCV 2023)
https://arxiv.org/abs/2303.09030

Adaptively selects receptive field size per spatial location via two
parallel branches with different kernel sizes. A channel-wise selection
mechanism (GAP → FC → ReLU → FC → Sigmoid) produces per-branch weights,
and the weighted sum gives the final output.

Key advantage: large, adaptive receptive fields capture context around
small defects without fixed kernel-size trade-offs.
"""

import torch
import torch.nn as nn


class LSK(nn.Module):
    """Large Selective Kernel Module.

    Two parallel branches with different effective receptive fields
    (5×5 and 7×7, each decomposed into stacked 3×3 convolutions) are
    combined via a learned channel-wise selection gate.

    Attributes:
        branch_large (nn.Sequential): 5×5 branch (two 3×3 convolutions).
        branch_larger (nn.Sequential): 7×7 branch (three 3×3 convolutions).
        gate (nn.Sequential): Channel-wise selection (GAP → FC → ReLU → FC → Sigmoid).

    Args:
        channels (int): Number of input (and output) feature channels.

    Shape:
        - Input:  (B, C, H, W)
        - Output: (B, C, H, W)
    """

    def __init__(self, channels: int):
        """Initialize LSK with dual-branch kernels and selection gate.

        Args:
            channels (int): Input channel count.
        """
        super().__init__()

        # Branch 1: effective 5×5 via two 3×3 convolutions
        self.branch_large = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.BatchNorm2d(channels),
        )

        # Branch 2: effective 7×7 via three 3×3 convolutions
        self.branch_larger = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels),
            nn.BatchNorm2d(channels),
        )

        # Channel-wise selection gate: GAP → FC → ReLU → FC → Sigmoid
        mid = max(channels // 4, 8)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels * 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply large selective kernel attention.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Kernel-selected feature map of same shape.
        """
        b, c, _, _ = x.shape

        # Dual-branch feature extraction
        feat_large = self.branch_large(x)    # (B, C, H, W)
        feat_larger = self.branch_larger(x)  # (B, C, H, W)

        # Channel-wise selection weights
        weights = self.gate(x)                       # (B, 2*C)
        weights = weights.view(b, 2, c, 1, 1)       # (B, 2, C, 1, 1)

        # Weighted combination
        stacked = torch.stack([feat_large, feat_larger], dim=1)  # (B, 2, C, H, W)
        out = (stacked * weights).sum(dim=1)                     # (B, C, H, W)

        return x + out
