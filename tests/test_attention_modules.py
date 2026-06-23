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
