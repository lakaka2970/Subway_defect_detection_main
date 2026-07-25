"""
Ultralytics Extra Neural Network Modules.

Extended module registry that bridges Ultralytics' native module
resolution (via tasks.py globals()) with custom domain-specific
modules for subway catenary defect detection.
"""

# Standard extra modules
from .ECA import ECA  # Efficient Channel Attention

# CBAM is defined in ultralytics.nn.modules.conv — re-export here so
# parse_model's ``globals()`` lookup resolves it for YAML-based configs.
from ..modules.conv import CBAM

# Project custom attention modules — imported from the sibling
# subway_defect package. After ``pip install -e .`` both packages are
# importable, and tasks.py's ``from .extra_module import *`` exposes
# these in parse_model's globals() for YAML-based module resolution.
from subway_defect.modules.EMA import EMA       # Efficient Multi-Scale Attention
from subway_defect.modules.SimAM import SimAM   # Simple Parameter-Free Attention

__all__ = [
    "CBAM",
    "ECA",
    "EMA",
    "SimAM",
]
