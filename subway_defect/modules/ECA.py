"""
Efficient Channel Attention (ECA) Module.

Reference: ECA-Net: Efficient Channel Attention for Deep Convolutional
Neural Networks (CVPR 2020 / ICCV 2019 Workshop)
https://arxiv.org/abs/1910.03151

A lightweight channel attention mechanism that uses 1D convolution of
adaptive kernel size instead of dimensionality reduction, avoiding
information loss while adding negligible parameters (< 100 per layer).

Suitable as a drop-in replacement for heavier attention modules (SE,
CBAM) or as a lightweight alternative to SimAM at coarser scales (P4/P5)
where spatial attention is less critical.
"""

import math

import torch
import torch.nn as nn


class ECA(nn.Module):
    """Efficient Channel Attention Module.

    Computes channel-wise attention via 1D convolution with adaptively
    sized kernel. No dimensionality reduction — the 1D conv operates
    directly on the channel-wise aggregated feature.

    Kernel size ``k`` is derived from channel count:

        k = | (log2(C) / gamma + b / gamma) |_odd

    where |·|_odd rounds to the nearest odd integer. This makes the
    receptive field proportional to channel dimensionality.

    Attributes:
        k (int): Adaptive 1D convolution kernel size (always odd).
        conv (nn.Conv1d): 1D convolution over channels.
        sigmoid (nn.Sigmoid): Gating activation.

    Args:
        channels (int): Number of input channels. The 1D kernel size is
            derived automatically from this value.
        gamma (int): Scaling factor for kernel-size computation.
            Default: 2 (from paper).
        b (int): Bias term for kernel-size computation. Default: 1.

    Shape:
        - Input:  (B, C, H, W)
        - Output: (B, C, H, W)

    Examples:
        >>> eca = ECA(256)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> y = eca(x)  # shape unchanged
    """

    def __init__(self, channels: int = None, gamma: int = 2, b: int = 1):
        """Initialize ECA with adaptive 1D kernel.

        Args:
            channels: Input channel count. Must be provided by YOLO
                ``parse_model`` (receives from ``args=[c2, *args]``).
            gamma: Scaling factor for kernel-size computation.
            b: Bias for kernel-size computation.
        """
        super().__init__()
        if channels is None:
            import logging
            logging.getLogger(__name__).warning(
                "ECA received channels=None — YOLO parse_model may not have "
                "passed c2 correctly. Falling back to 256. Check the model "
                "YAML that this module is not missing args."
            )
            channels = 256
        self.channels = channels

        # Adaptive kernel size: k = | (log2(C) + b) / gamma |_odd
        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 == 1 else t + 1
        self.k = k

        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply efficient channel attention.

        Args:
            x: Input feature map of shape (B, C, H, W).

        Returns:
            Attention-modulated feature map of same shape.
        """
        # Global average pooling → (B, C, 1)
        gap = x.mean(dim=[2, 3], keepdim=False)  # (B, C)

        # 1D conv over channels (no dimension reduction)
        # Reshape: (B, C) → (B, 1, C)
        y = gap.unsqueeze(1)
        y = self.conv(y)
        y = y.squeeze(1)  # (B, C)

        # Sigmoid gate and broadcast
        attention = self.sigmoid(y).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return x * attention

    def extra_repr(self) -> str:
        """Return extra repr string for print(model)."""
        return f"channels={self.channels}, kernel_size={self.k}"
