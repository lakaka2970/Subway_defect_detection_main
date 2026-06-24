"""
Shared utility functions for the defect detection pipeline.
"""

import numpy as np


def box_iou(box1: dict, box2: dict) -> float:
    """Compute IoU between two normalized (x, y, w, h) boxes.

    Args:
        box1: Dict with keys x, y, w, h (normalized center + size).
        box2: Dict with keys x, y, w, h (normalized center + size).

    Returns:
        IoU value in [0, 1].
    """
    b1 = box1["box"] if "box" in box1 else box1
    b2 = box2["box"] if "box" in box2 else box2

    x1_min = b1["x"] - b1["w"] / 2
    x1_max = b1["x"] + b1["w"] / 2
    y1_min = b1["y"] - b1["h"] / 2
    y1_max = b1["y"] + b1["h"] / 2

    x2_min = b2["x"] - b2["w"] / 2
    x2_max = b2["x"] + b2["w"] / 2
    y2_min = b2["y"] - b2["h"] / 2
    y2_max = b2["y"] + b2["h"] / 2

    inter_x = max(0.0, min(x1_max, x2_max) - max(x1_min, x2_min))
    inter_y = max(0.0, min(y1_max, y2_max) - max(y1_min, y2_min))
    inter = inter_x * inter_y

    area1 = b1["w"] * b1["h"]
    area2 = b2["w"] * b2["h"]
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0
