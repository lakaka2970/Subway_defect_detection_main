"""
Subway Defect Detection — Custom Neural Network Modules.

Domain-specific modules for the catenary defect detection model,
including attention mechanisms optimized for small object detection
in high-resolution infrastructure imagery.
"""

from .CoordAtt import CoordAtt
from .DCN import DeformConv2d
from .EMA import EMA
from .LSK import LSK
from .SimAM import SimAM

__all__ = ["CoordAtt", "DeformConv2d", "EMA", "LSK", "SimAM"]
