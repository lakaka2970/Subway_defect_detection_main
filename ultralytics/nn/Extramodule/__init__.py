# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics Extra Neural Network Modules.

This module provides custom attention and feature extraction modules
that extend the standard Ultralytics model components.
"""

# Only import from modules that actually exist in this directory
from .ECA import ECA  # Efficient Channel Attention (implemented)

# CBAM, ChannelAttention, SpatialAttention are defined in ultralytics.nn.modules.conv
# ADown is defined in ultralytics.nn.modules.block
# These are imported through the standard module system — do NOT re-import here
# to avoid circular imports and shadowing.

# Additional custom modules (EMA, SimAM) will be bridged in later tasks.

__all__ = ["ECA"]
