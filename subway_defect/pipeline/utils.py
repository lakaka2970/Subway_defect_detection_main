"""
Shared utility functions for the defect detection pipeline.
"""

import json
from pathlib import Path
from typing import Dict, Mapping, Union

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


def load_class_thresholds(path: Union[str, Path]) -> Dict[str, float]:
    """Load per-class confidence thresholds from a calibration JSON file.

    Supported schemas:
    - ``{"VHBNM": 0.20, ...}``
    - ``{"VHBNM": {"recommended_threshold": 0.20, ...}, ...}``

    Args:
        path: JSON file produced by a calibration step.

    Returns:
        Mapping from class name to confidence threshold.

    Raises:
        ValueError: If a class entry cannot be interpreted as a threshold.
    """
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)

    thresholds: Dict[str, float] = {}
    for class_name, value in payload.items():
        if isinstance(value, Mapping):
            if "recommended_threshold" not in value:
                raise ValueError(
                    f"Threshold entry for {class_name!r} lacks "
                    "'recommended_threshold'"
                )
            value = value["recommended_threshold"]
        try:
            threshold = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Threshold entry for {class_name!r} is not numeric: {value!r}"
            ) from exc
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold entry for {class_name!r} must be in [0, 1], "
                f"got {threshold}"
            )
        thresholds[str(class_name)] = threshold

    return thresholds
