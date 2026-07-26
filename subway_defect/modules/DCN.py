"""
Deformable Convolution v2 (DCN) Module.

Reference: Deformable ConvNets v2: More Deformable, Better Results
(CVPR 2019)
https://arxiv.org/abs/1811.11168

Learns per-sample spatial offsets and modulation scalars so that
convolution sampling locations adapt to object geometry. This is
especially useful for irregular defect shapes (cracks, corrosion)
where fixed-grid sampling is suboptimal.

Uses ``torchvision.ops.deform_conv2d`` when available (GPU-optimized
CUDA kernel), with a pure-PyTorch fallback for environments without
torchvision or on CPU-only setups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d as _tv_deform_conv2d

    _HAS_TORCHVISION_DCN = True
except ImportError:
    _HAS_TORCHVISION_DCN = False


class DeformConv2d(nn.Module):
    """Deformable Convolution v2.

    Augments a standard convolution with learned 2-D offsets and
    modulation scalars per sampling location, enabling the receptive
    field to deform according to input content.

    Attributes:
        offset_conv (nn.Conv2d): Predicts 2*K² offset channels.
        mod_conv (nn.Conv2d): Predicts K² modulation scalars.
        weight (nn.Parameter): Convolution kernel of shape
            (out_channels, in_channels // groups, kH, kW).
        bias (nn.Parameter): Optional bias of shape (out_channels,).

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Convolution kernel size. Default: 3.
        stride (int): Convolution stride. Default: 1.
        padding (int): Zero-padding added to both sides. Default: 1.
        dilation (int): Kernel dilation. Default: 1.
        groups (int): Number of blocked connections. Default: 1.
        deformable_groups (int): Number of deformable offset groups.
            Default: 4.

    Shape:
        - Input:  (B, in_channels, H, W)
        - Output: (B, out_channels, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        deformable_groups: int = 4,
    ):
        """Initialize DeformConv2d with offset/modulation branches and weight.

        Args:
            in_channels (int): Input channel count.
            out_channels (int): Output channel count.
            kernel_size (int): Size of the convolving kernel.
            stride (int): Stride of the convolution.
            padding (int): Zero-padding size.
            dilation (int): Kernel dilation factor.
            groups (int): Number of convolution groups.
            deformable_groups (int): Number of deformable offset groups.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.deformable_groups = deformable_groups

        kk = kernel_size * kernel_size

        # Offset branch: 2*K² channels (dy, dx per sampling location)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * deformable_groups * kk,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        # Modulation branch: K² scalars per group
        self.mod_conv = nn.Conv2d(
            in_channels,
            deformable_groups * kk,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )

        # Main convolution weight
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self._init_weights()

    def _init_weights(self):
        """Initialize offset/modulation convs to zero and weight via Kaiming."""
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mod_conv.weight)
        nn.init.zeros_(self.mod_conv.bias)
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply deformable convolution to input feature map.

        Args:
            x (torch.Tensor): Input feature map of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output feature map of shape (B, C_out, H_out, W_out).
        """
        offset = self.offset_conv(x)
        mask = torch.sigmoid(self.mod_conv(x))

        if _HAS_TORCHVISION_DCN:
            return _tv_deform_conv2d(
                x,
                offset,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                mask=mask,
            )

        # Pure-PyTorch fallback
        return self._deform_conv2d_fallback(x, offset, mask)

    def _deform_conv2d_fallback(
        self, x: torch.Tensor, offset: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Pure-PyTorch deformable convolution fallback.

        Uses grid_sample to gather features at offset locations, then
        performs grouped convolution on the gathered features.

        Args:
            x (torch.Tensor): Input (B, C_in, H, W).
            offset (torch.Tensor): Offsets (B, 2*dg*K², H_out, W_out).
            mask (torch.Tensor): Modulation (B, dg*K², H_out, W_out).

        Returns:
            torch.Tensor: Output (B, C_out, H_out, W_out).
        """
        b, _, h_out, w_out = offset.shape[:1] + offset.shape[2:]
        k = self.kernel_size
        dg = self.deformable_groups
        kk = k * k

        # Build base grid of sampling centers
        shift_y, shift_x = torch.meshgrid(
            torch.arange(0, h_out, device=x.device, dtype=x.dtype) * self.stride,
            torch.arange(0, w_out, device=x.device, dtype=x.dtype) * self.stride,
            indexing="ij",
        )
        shift_y = shift_y.unsqueeze(0).expand(b, -1, -1)  # (B, H_out, W_out)
        shift_x = shift_x.unsqueeze(0).expand(b, -1, -1)

        # Kernel offsets (dilation-aware)
        ky = torch.arange(0, k, device=x.device, dtype=x.dtype) * self.dilation
        kx = torch.arange(0, k, device=x.device, dtype=x.dtype) * self.dilation
        grid_y, grid_x = torch.meshgrid(ky, kx, indexing="ij")
        grid_y = grid_y.reshape(-1)  # (K²,)
        grid_x = grid_x.reshape(-1)

        # Reshape offset: (B, dg, 2, K², H_out, W_out)
        offset = offset.view(b, dg, 2, kk, h_out, w_out)
        mask = mask.view(b, dg, kk, h_out, w_out)

        # Compute sampling locations per group
        out_channels_per_group = self.out_channels // self.groups
        in_channels_per_group = self.in_channels // self.groups

        output = x.new_zeros(b, self.out_channels, h_out, w_out)

        for g in range(self.groups):
            # Map group g to deformable group
            dg_idx = g % dg
            off_g = offset[:, dg_idx]  # (B, 2, K², H_out, W_out)
            mask_g = mask[:, dg_idx]   # (B, K², H_out, W_out)

            # Absolute sampling positions
            sample_y = shift_y.unsqueeze(1).unsqueeze(1) + grid_y.view(1, 1, kk, 1, 1) + off_g[:, 0]
            sample_x = shift_x.unsqueeze(1).unsqueeze(1) + grid_x.view(1, 1, kk, 1, 1) + off_g[:, 1]

            # Normalize to [-1, 1] for grid_sample
            _, _, h_in, w_in = x.shape
            norm_y = 2.0 * sample_y / max(h_in - 1, 1) - 1.0  # (B, 1, K², H_out, W_out)
            norm_x = 2.0 * sample_x / max(w_in - 1, 1) - 1.0

            # Sample each kernel position
            x_g = x[:, g * in_channels_per_group : (g + 1) * in_channels_per_group]
            gathered = []
            for ki in range(kk):
                grid_ki = torch.stack([norm_x[:, 0, ki], norm_y[:, 0, ki]], dim=-1)  # (B, H_out, W_out, 2)
                sampled = F.grid_sample(
                    x_g, grid_ki, mode="bilinear", padding_mode="zeros", align_corners=True
                )  # (B, C_in/g, H_out, W_out)
                sampled = sampled * mask_g[:, ki].unsqueeze(1)  # apply modulation
                gathered.append(sampled)

            # Stack and convolve: (B, C_in/g * K², H_out, W_out) → 1×1 conv
            feat = torch.cat(gathered, dim=1)
            w_g = self.weight[g * out_channels_per_group : (g + 1) * out_channels_per_group]
            w_g = w_g.view(out_channels_per_group, -1, 1, 1)
            output[:, g * out_channels_per_group : (g + 1) * out_channels_per_group] = (
                F.conv2d(feat, w_g)
            )

        output = output + self.bias.view(1, -1, 1, 1)
        return output
