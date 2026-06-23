"""
Simple, Parameter-Free Attention Module (SimAM).

Reference:
    SimAM: A Simple, Parameter-Free Attention Module for Convolutional
    Neural Networks, ICML 2021

Based on neuroscience theory of spatial inhibition: important neurons
exhibit inhibitory effects on surrounding neurons. The energy function
identifies neurons that are "surprisingly different" from their
neighborhood — exactly the property needed for detecting local
anomalies like missing bolts in an otherwise regular structure.

Key advantage: ZERO parameters — no overfitting risk on small datasets.
"""

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """Simple, Parameter-Free Attention Module.

    Computes a 3D attention weight map based on the energy function:
        e_t = 4 * (sigma^2 + lambda) / ((t - mu)^2 + 2*sigma^2 + 2*lambda)

    where:
        t   = target neuron value at a spatial position
        mu  = mean of all neurons in the same channel
        sigma^2 = variance of all neurons in the same channel
        lambda  = regularization constant (default 1e-4)

    The attention weight is: 1 / (1 + e_t), bound in (0, 1).

    Neurons that strongly deviate from their channel's mean receive
    high attention (close to 1); background/noise neurons receive
    low attention (close to 0).

    Args:
        channels: Accepted for compatibility with YOLO parse_model
                  (receives input channel count from args=[c2, *args]).
                  Not used internally — SimAM is parameter-free.
        lambda_e (float): Regularization constant. Larger values make
                          attention more uniform. Default: 1e-4.

    Shape:
        - Input:  (B, C, H, W)
        - Output: (B, C, H, W)

    Examples:
        >>> simam = SimAM()
        >>> x = torch.randn(1, 256, 64, 64)
        >>> output = simam(x)
        >>> assert output.shape == x.shape
        >>> sum(p.numel() for p in simam.parameters())  # 0
    """

    def __init__(self, channels: int = None, lambda_e: float = 1e-4):
        super().__init__()
        self.lambda_e = lambda_e
        # channels accepted for parse_model compatibility (not used internally)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Attention-modulated feature map.
        """
        # Spatial size per channel
        n = x.shape[2] * x.shape[3]  # H * W

        # Channel-wise mean: (B, C, 1, 1)
        mu = x.mean(dim=[2, 3], keepdim=True)

        # Channel-wise variance (population variance, matches paper)
        sigma_sq = ((x - mu) ** 2).mean(dim=[2, 3], keepdim=True)

        # Energy function e_t for all spatial positions
        numerator = 4.0 * (sigma_sq + self.lambda_e)
        denominator = (x - mu) ** 2 + 2.0 * sigma_sq + 2.0 * self.lambda_e
        energy = numerator / denominator  # (B, C, H, W)

        # Attention: 1/(1+energy), bound in (0, 1)
        attention = 1.0 / (1.0 + energy)  # (B, C, H, W)

        return x * attention
