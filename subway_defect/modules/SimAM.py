"""
Simple, Parameter-Free Attention Module (SimAM).

Reference: SimAM: A Simple, Parameter-Free Attention Module for
Convolutional Neural Networks (ICML 2021)
https://proceedings.mlr.press/v139/yang21o.html

Based on neuroscience spatial inhibition theory: important neurons exhibit
inhibitory effects on surrounding neurons. The energy function identifies
neurons that are "surprisingly different" from their neighborhood — ideal
for detecting local anomalies like missing bolts in regular structures.

Key advantage: ZERO parameters — no overfitting risk on small datasets.
"""

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """Simple, Parameter-Free Attention Module.

    Computes a 3-D attention weight map via energy function:

        e_t = 4 * (sigma^2 + lambda) / ((t - mu)^2 + 2*sigma^2 + 2*lambda)

    where t is the target neuron, mu/sigma^2 are the channel-wise mean and
    variance, and lambda is a regularization constant. The attention weight
    ``1 / (1 + e_t)`` is bounded in (0, 1) without any sigmoid.

    Neurons that strongly deviate from their channel mean receive high
    attention; background/noise neurons receive low attention.

    Attributes:
        lambda_e (float): Regularization constant for the energy function.

    Args:
        channels (int): Accepted for YOLO parse_model compatibility (receives
            input channel count from ``args=[c2, *args]``). Not used internally.
        lambda_e (float): Regularization constant. Larger values make attention
            more uniform. Default: ``1e-4``.

    Shape:
        - Input:  (B, C, H, W)
        - Output: (B, C, H, W)
    """

    def __init__(self, channels: int = None, lambda_e: float = 1e-4):
        """Initialize SimAM.

        Args:
            channels (int): Input channels (ignored, for parse_model compat).
            lambda_e (float): Energy function regularization constant.
        """
        super().__init__()
        self.lambda_e = lambda_e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply parameter-free energy-based attention.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Attention-modulated feature map of same shape.
        """
        # Channel-wise mean and variance
        mu = x.mean(dim=[2, 3], keepdim=True)
        sigma_sq = ((x - mu) ** 2).mean(dim=[2, 3], keepdim=True)

        # Energy function (eq. 4 from paper)
        numerator = 4.0 * (sigma_sq + self.lambda_e)
        denominator = (x - mu) ** 2 + 2.0 * sigma_sq + 2.0 * self.lambda_e

        # Attention: 1 / (1 + e_t), naturally bounded in (0, 1)
        return x / (1.0 + numerator / denominator)
