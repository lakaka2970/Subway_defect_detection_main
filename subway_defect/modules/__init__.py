"""
Subway Defect Detection — Custom Neural Network Modules.

Domain-specific modules for the catenary defect detection model,
including attention mechanisms optimized for small object detection
in high-resolution infrastructure imagery.

Attention modules:
    - SimAM:  Parameter-free energy-based spatial attention (ICML 2021)
              Best at P2/P3 for local anomaly detection in regular structures.
    - EMA:    Efficient Multi-Scale Attention (ICASSP 2023)
              X/Y directional pooling for spatial position encoding.
    - ECA:    Efficient Channel Attention (CVPR 2020)
              Lightweight 1D-conv channel attention — P4/P5 alternative.
"""

from .ECA import ECA
from .EMA import EMA
from .SimAM import SimAM

__all__ = ["ECA", "EMA", "SimAM"]
