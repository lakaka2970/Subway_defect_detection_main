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

__all__ = [
    "ECA",
    "EMA",
    "SimAM",
]
