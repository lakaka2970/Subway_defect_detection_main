# modules/EMA.py
"""
Efficient Multi-Scale Attention (EMA) Module.

Reference:
    EMA: Efficient Multi-Scale Attention, ICASSP 2023

This implementation creates a lightweight multi-scale attention mechanism
that encodes spatial information along X and Y directions separately,
then fuses cross-group features through 1x1 convolutions.

Designed for the P3 detection branch of YOLO11 to enhance
small-object spatial localization in catenary defect detection.
"""

import torch
import torch.nn as nn


class EMA(nn.Module):
    """Efficient Multi-Scale Attention Module.

    Preserves spatial position information by performing separate
    average pooling along X (width) and Y (height) dimensions,
    enabling attention to be sensitive to object location.

    Architecture:
        1. GroupNorm for input normalization
        2. X-direction AvgPool (1xW) + Y-direction AvgPool (Hx1)
        3. 1x1 Conv for cross-group feature interaction
        4. 3x3 Conv for local spatial refinement
        5. Sigmoid gating for attention weights

    Args:
        channels (int): Number of input feature channels.
        groups (int): Number of groups for GroupNorm normalization and channel
            divisibility. The attention layers (1x1 and 3x3 convolutions) operate on
            all channels jointly rather than per-group, which is a valid simplification
            for efficiency while maintaining expressive cross-channel interaction.
            Default: 4.
        kernel_size (int): Kernel size for spatial refinement conv. Default: 3.

    Shape:
        - Input:  (B, C, H, W)
        - Output: (B, C, H, W) — same shape as input

    Examples:
        >>> ema = EMA(channels=256)
        >>> x = torch.randn(1, 256, 128, 128)
        >>> output = ema(x)
        >>> assert output.shape == x.shape
    """

    def __init__(self, channels: int, groups: int = 4, kernel_size: int = 3):
        super().__init__()
        self.channels = channels
        self.groups = groups
        assert channels % groups == 0, (
            f"channels ({channels}) must be divisible by groups ({groups})"
        )
        # Input normalization
        self.gn = nn.GroupNorm(num_groups=groups, num_channels=channels)

        # Cross-group interaction: 1x1 conv
        self.conv1x1 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # Spatial refinement: 3x3 conv
        self.conv3x3 = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Attention-modulated feature map.
        """
        _, c, h, w = x.shape

        # Step 1: Normalize
        normalized = self.gn(x)

        # Step 2: Dual-direction spatial encoding
        x_pool = normalized.mean(dim=2, keepdim=True)  # (B, C, 1, W)
        y_pool = normalized.mean(dim=3, keepdim=True)  # (B, C, H, 1)

        # Step 3: Cross-group 1x1 interaction
        x_attn = self.conv1x1(x_pool)  # (B, C, 1, W)
        y_attn = self.conv1x1(y_pool)  # (B, C, H, 1)

        # Step 4: Expand and fuse
        x_expanded = x_attn.expand(-1, -1, h, -1)  # (B, C, H, W)
        y_expanded = y_attn.expand(-1, -1, -1, w)  # (B, C, H, W)
        fused = x_expanded + y_expanded

        # Step 5: Spatial refinement
        refined = self.conv3x3(fused)  # (B, C, H, W)

        # Step 6: Sigmoid gating
        attention = self.sigmoid(refined)  # (B, C, H, W), values in [0, 1]

        return x * attention
