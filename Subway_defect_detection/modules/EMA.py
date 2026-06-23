# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Efficient Multi-Scale Attention (EMA) Module.

Reference: EMA: Efficient Multi-Scale Attention (ICASSP 2023)
https://arxiv.org/abs/2305.13563

Encodes spatial information along X and Y directions separately, then fuses
cross-group features through 1x1 and 3x3 convolutions for small-object
spatial enhancement. Designed for the P3 detection branch.
"""

import torch
import torch.nn as nn


class EMA(nn.Module):
    """Efficient Multi-Scale Attention Module.

    Performs separate average pooling along X (width) and Y (height)
    dimensions to preserve spatial position information, enabling
    attention sensitive to object location.

    Attributes:
        gn (nn.GroupNorm): Input normalization across groups.
        conv1x1 (nn.Conv2d): 1x1 cross-group feature interaction.
        conv3x3 (nn.Conv2d): 3x3 spatial refinement convolution.
        sigmoid (nn.Sigmoid): Gating activation.

    Args:
        channels (int): Number of input feature channels. Must be divisible
            by `groups`.
        groups (int): Group count for GroupNorm normalization and channel
            divisibility constraint. The 1x1 and 3x3 convolutions operate on
            all channels jointly for efficiency. Default: 4.
        kernel_size (int): Kernel size for spatial refinement convolution.
            Default: 3.

    Shape:
        - Input:  (B, C, H, W)
        - Output: (B, C, H, W)
    """

    def __init__(self, channels: int, groups: int = 4, kernel_size: int = 3):
        """Initialize EMA with normalization, cross-group, and refinement layers.

        Args:
            channels (int): Input channel count (must be divisible by groups).
            groups (int): Number of groups for GroupNorm.
            kernel_size (int): Kernel size for the spatial refinement conv.
        """
        super().__init__()
        self.groups = groups
        assert channels % groups == 0, (
            f"channels ({channels}) must be divisible by groups ({groups})"
        )

        self.gn = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.conv1x1 = nn.Conv2d(channels, channels, 1, 1, 0)
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size, 1, kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale attention to input feature map.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Attention-modulated feature map of same shape.
        """
        _, _, h, w = x.shape

        normalized = self.gn(x)

        # Dual-direction spatial encoding
        x_pool = normalized.mean(dim=2, keepdim=True)  # (B, C, 1, W)
        y_pool = normalized.mean(dim=3, keepdim=True)  # (B, C, H, 1)

        # Cross-group interaction and spatial fusion
        x_attn = self.conv1x1(x_pool)
        y_attn = self.conv1x1(y_pool)
        fused = x_attn.expand(-1, -1, h, -1) + y_attn.expand(-1, -1, -1, w)

        # Spatial refinement and gating
        attention = self.sigmoid(self.conv3x3(fused))
        return x * attention
