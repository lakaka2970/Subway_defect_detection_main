# tests/test_attention_modules.py
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Subway_defect_detection"))

import pytest
import torch
from modules.EMA import EMA


class TestEMA:
    """Test suite for EMA (Efficient Multi-Scale Attention) module."""

    @pytest.fixture
    def input_tensor(self):
        """Create a standard input tensor: BCHW."""
        return torch.randn(2, 128, 64, 64)

    def test_ema_forward_shape(self, input_tensor):
        """EMA output shape matches input shape."""
        ema = EMA(channels=128)
        output = ema(input_tensor)
        assert output.shape == input_tensor.shape, (
            f"Expected shape {input_tensor.shape}, got {output.shape}"
        )

    def test_ema_forward_dtype(self, input_tensor):
        """EMA output dtype matches input dtype."""
        ema = EMA(channels=128)
        output = ema(input_tensor)
        assert output.dtype == input_tensor.dtype

    def test_ema_not_inplace(self, input_tensor):
        """EMA does not modify input in-place."""
        original = input_tensor.clone()
        ema = EMA(channels=128)
        _ = ema(input_tensor)
        assert torch.equal(input_tensor, original), "Input tensor was modified in-place"

    def test_ema_different_channels(self):
        """EMA works with various channel counts."""
        for c in [64, 128, 256, 512]:
            ema = EMA(channels=c)
            x = torch.randn(1, c, 32, 32)
            output = ema(x)
            assert output.shape == x.shape

    def test_ema_different_spatial_sizes(self):
        """EMA works with various spatial dimensions."""
        ema = EMA(channels=128)
        for h, w in [(16, 16), (32, 64), (128, 128), (256, 256)]:
            x = torch.randn(1, 128, h, w)
            output = ema(x)
            assert output.shape == x.shape

    def test_ema_attention_range(self, input_tensor):
        """EMA output is bounded after sigmoid attention modulation."""
        ema = EMA(channels=128)
        output = ema(input_tensor)
        # Output should be input modulated by sigmoid weights in [0, 1]
        # So output magnitude should not greatly exceed input magnitude
        assert output.abs().max() <= input_tensor.abs().max() * 1.1, (
            "EMA output values out of expected range"
        )

    def test_ema_gradient_flow(self):
        """Gradients flow through EMA module."""
        ema = EMA(channels=64)
        x = torch.randn(1, 64, 32, 32, requires_grad=True)
        output = ema(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None, "No gradient flowing through EMA"
        assert not (x.grad == 0).all(), "Gradient is all zeros through EMA"

    def test_ema_train_eval_consistent(self, input_tensor):
        """EMA has same output shape in train and eval mode."""
        ema = EMA(channels=128)
        ema.train()
        out_train = ema(input_tensor)
        ema.eval()
        with torch.no_grad():
            out_eval = ema(input_tensor)
        assert out_train.shape == out_eval.shape

    def test_ema_groups_parameter(self):
        """EMA works with different group counts."""
        ema = EMA(channels=256, groups=8)
        x = torch.randn(1, 256, 32, 32)
        output = ema(x)
        assert output.shape == x.shape

    def test_ema_kernel_size_parameter(self):
        """EMA works with non-default kernel_size."""
        ema = EMA(channels=128, kernel_size=5)
        x = torch.randn(1, 128, 32, 32)
        output = ema(x)
        assert output.shape == x.shape


class TestSimAM:
    """Test suite for SimAM (Simple Parameter-Free Attention) module."""

    @pytest.fixture
    def input_tensor(self):
        """Create a standard input tensor: BCHW."""
        return torch.randn(2, 128, 64, 64)

    def test_simam_forward_shape(self, input_tensor):
        """SimAM output shape matches input shape."""
        from modules.SimAM import SimAM
        simam = SimAM()
        output = simam(input_tensor)
        assert output.shape == input_tensor.shape

    def test_simam_forward_dtype(self, input_tensor):
        """SimAM output dtype matches input dtype."""
        from modules.SimAM import SimAM
        simam = SimAM()
        output = simam(input_tensor)
        assert output.dtype == input_tensor.dtype

    def test_simam_not_inplace(self, input_tensor):
        """SimAM does not modify input in-place."""
        from modules.SimAM import SimAM
        original = input_tensor.clone()
        simam = SimAM()
        _ = simam(input_tensor)
        assert torch.equal(input_tensor, original), "Input was modified in-place"

    def test_simam_zero_parameters(self):
        """SimAM has zero trainable parameters."""
        from modules.SimAM import SimAM
        simam = SimAM()
        num_params = sum(p.numel() for p in simam.parameters())
        assert num_params == 0, f"SimAM should have 0 params, has {num_params}"

    def test_simam_different_input_sizes(self):
        """SimAM works with various input dimensions."""
        from modules.SimAM import SimAM
        simam = SimAM()
        for c, h, w in [(64, 32, 32), (128, 64, 64), (256, 128, 128), (512, 16, 16)]:
            x = torch.randn(1, c, h, w)
            output = simam(x)
            assert output.shape == x.shape

    def test_simam_attention_enhances_features(self, input_tensor):
        """SimAM highlights spatially salient regions.

        A synthetic input with a bright spot in the center should have
        that spot's relative intensity preserved or enhanced.
        """
        from modules.SimAM import SimAM
        simam = SimAM()
        x = torch.zeros(1, 8, 32, 32)
        x[:, :, 14:18, 14:18] = 5.0  # bright center

        output = simam(x)
        center_out = output[:, :, 14:18, 14:18].mean()
        edge_out = output[:, :, 0:4, 0:4].mean()
        assert center_out > edge_out, (
            "SimAM should preserve relative feature saliency"
        )

    def test_simam_gradient_flow(self):
        """Gradients flow through SimAM despite zero parameters."""
        from modules.SimAM import SimAM
        simam = SimAM()
        x = torch.randn(1, 64, 32, 32, requires_grad=True)
        output = simam(x)
        loss = output.sum()
        loss.backward()
        assert x.grad is not None, "No gradient flowing through SimAM"
        assert not (x.grad == 0).all(), "Gradient is all zeros through SimAM"

    def test_simam_lambda_sensitivity(self):
        """SimAM with different lambda values produces different outputs."""
        from modules.SimAM import SimAM
        x = torch.randn(2, 16, 32, 32)
        out_small = SimAM(lambda_e=1e-6)(x.clone())
        out_large = SimAM(lambda_e=1e-2)(x.clone())
        assert not torch.allclose(out_small, out_large, atol=1e-4), (
            "Different lambda_e should produce different outputs"
        )

    def test_simam_train_eval_consistent(self, input_tensor):
        """SimAM produces consistent shapes in train and eval modes."""
        from modules.SimAM import SimAM
        simam = SimAM()
        simam.train()
        out_train = simam(input_tensor)
        simam.eval()
        with torch.no_grad():
            out_eval = simam(input_tensor)
        assert out_train.shape == out_eval.shape
