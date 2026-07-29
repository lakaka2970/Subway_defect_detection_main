"""
Inference pipeline modules for subway catenary defect detection.
"""

from .cascade import CascadeClassifier
from .slicer import SmartSlicer
from .two_stage import TwoStagePipeline
from .wbf_fusion import WBFFusion

__all__ = ["CascadeClassifier", "SmartSlicer", "TwoStagePipeline", "WBFFusion"]
