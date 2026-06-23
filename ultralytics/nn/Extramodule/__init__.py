# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics Extra Neural Network Modules.

Extended module registry that bridges Ultralytics' native module
resolution (via tasks.py globals()) with custom domain-specific
modules for subway catenary defect detection.
"""

import sys
from pathlib import Path

# Standard extra modules
from .ECA import ECA  # Efficient Channel Attention (implemented in Extramodule)

# Attention modules bridged from ultralytics core.
# CBAM is defined in ultralytics.nn.modules.conv but not imported in tasks.py
# directly, so we re-export it here to satisfy parse_model's globals() resolution
# and the elif m in {CBAM} guard.
from ..modules.conv import CBAM

# Custom attention modules — bridged from the project's modules/ package.
# These must be importable from ultralytics.nn.Extramodule so that
# tasks.py's `from .Extramodule import *` exposes them in parse_model's
# globals() namespace for YAML-based module resolution.
_project_root = Path(__file__).parent.parent.parent.parent
_modules_parent = _project_root / "Subway_defect_detection"
if str(_modules_parent) not in sys.path:
    sys.path.insert(0, str(_modules_parent))

from modules.EMA import EMA       # Efficient Multi-Scale Attention
from modules.SimAM import SimAM   # Simple Parameter-Free Attention

# Pre-existing parse_model elif guards reference CA, SE, and MLLAttention,
# but these classes are not yet implemented in the codebase.  Define them
# as None so the `elif m in {None}` guards are harmless (always False)
# rather than raising NameError for every non-base module.
CA = None  # Coordinate Attention — not yet implemented
SE = None  # Squeeze-and-Excitation — not yet implemented
MLLAttention = None  # Multi-Level Local Attention — not yet implemented

__all__ = [
    "CBAM",
    "CA",
    "ECA",
    "EMA",
    "MLLAttention",
    "SE",
    "SimAM",
]
