"""
Coordinate Attention (CoordAtt) Module.

Reference: Coordinate Attention for Efficient Mobile Network Design
(CVPR 2021)
https://arxiv.org/abs/2103.02907

Encodes spatial information in channel attention via directional pooling:
pools along width (1×W) to capture horizontal position and along height
(H×1) to capture vertical position. The two pooled features are fused
through a shared bottleneck, then split into two independent attention
maps that modulate the input along each spatial axis.

Key advantage over standard channel attention (SE): preserves precise
positional information, enabling the network to attend to *where*
defects are, not just *which* channels matter.
"""

import torch
import torch.nn as nn


class CoordAtt(nn.Module):
    """Coordinate Attention Module.

    Decomposes channel attention into two 1-D feature encodings along
    the horizontal and vertical directions, preserving positional
    information that is lost in global average pooling.

    Attributes:
        reduce_conv (nn.Sequential): Shared 1×1 bottleneck (Conv → BN → ReLU)
            that fuses the concatenated directional features.
        conv_h (nn.Conv2d): 1×1 conv producing the horizontal attention map.
        conv_w (nn.Conv2d): 1×1 conv producing the vertical attention map.
        sigmoid (nn.Sigmoid): Gating activation for attention maps.

    Args:
        channels (int): Number of input feature channels.
        reduction (int): Reduction ratio for the bottleneck. The
            intermediate channel count is ``max(channels // reduction, 8)``.
            Default: 32.

    Shape:
        - Input:  (B, C, H, W)
        - Output: (B, C, H, W)
    """

    def __init__(self, channels: int, reduction: int = 32):
        """Initialize CoordAtt with directional pooling and attention layers.

        Args:
            channels (int): Input channel count.
            reduction (int): Bottleneck reduction ratio.
        """
        super().__init__()
        mid = max(channels // reduction, 8)

        # Shared bottleneck: 1×1 Conv → BN → ReLU
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )

        # Separate 1×1 convs for horizontal and vertical attention
        self.conv_h = nn.Conv2d(mid, channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid, channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply coordinate attention to input feature map.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C, H, W).

        Returns:
            torch.Tensor: Attention-modulated feature map of same shape.
        """
        b, c, h, w = x.shape

        # Directional pooling
        x_h = x.mean(dim=3, keepdim=True)          # (B, C, H, 1) — encode vertical
        x_w = x.mean(dim=2, keepdim=True)          # (B, C, 1, W) — encode horizontal

        # Concatenate along spatial dim and pass through shared bottleneck
        # x_h is (B, C, H, 1), x_w transposed is (B, C, 1, W) → cat → (B, C, H+W, 1)
        y = torch.cat([x_h, x_w.permute(0, 1, 3, 2)], dim=2)  # (B, C, H+W, 1)
        y = self.reduce_conv(y)                                 # (B, mid, H+W, 1)

        # Split back into horizontal and vertical components
        x_h_feat, x_w_feat = y.split([h, w], dim=2)            # (B, mid, H, 1), (B, mid, W, 1)
        x_w_feat = x_w_feat.permute(0, 1, 3, 2)               # (B, mid, 1, W)

        # Generate attention maps
        attn_h = self.sigmoid(self.conv_h(x_h_feat))           # (B, C, H, 1)
        attn_w = self.sigmoid(self.conv_w(x_w_feat))           # (B, C, 1, W)

        return x * attn_h * attn_w
