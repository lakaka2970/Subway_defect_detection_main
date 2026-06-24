"""
Subway Defect Detection — Custom Neural Network Modules.

Domain-specific modules for the catenary defect detection model,
including attention mechanisms optimized for small object detection
in high-resolution infrastructure imagery.
"""

from .EMA import EMA
from .SimAM import SimAM

__all__ = ["EMA", "SimAM"]
