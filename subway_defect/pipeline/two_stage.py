"""
Two-stage inference pipeline: ROI proposer + defect detector.

Stage 1: YOLO11n-ROI detects structural regions on downsampled image.
Stage 2: YOLO11s/m-EMA-SimAM detects defects on full-res ROI tiles.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch

from .slicer import SmartSlicer
from .utils import box_iou


class TwoStagePipeline:
    """Two-stage catenary defect detection pipeline.

    Args:
        roi_model: YOLO model for Stage 1 (structural region detection).
        defect_model: YOLO model for Stage 2 (defect detection).
        slice_size: Tile size for Stage 2. Default: 1024.
        overlap: Overlap ratio for slicing (both stages). Default: 0.15.
        roi_conf: Confidence threshold for Stage 1. Default: 0.15.
        defect_conf: Confidence threshold for Stage 2. Default: 0.40.
        downsample_ratio: Downsample factor for Stage 1. Default: 8.
        device: Torch device string. Default: ``"0"``.
    """

    def __init__(
        self,
        roi_model,
        defect_model,
        slice_size: int = 1024,
        overlap: float = 0.15,
        roi_conf: float = 0.15,
        defect_conf: float = 0.40,
        downsample_ratio: int = 8,
        device: str = "0",
    ):
        # Fallback to CPU if requested CUDA device is unavailable
        if device not in ("", "cpu") and not torch.cuda.is_available():
            print("Warning: CUDA not available, falling back to CPU")
            device = "cpu"

        self.roi_model = roi_model
        self.defect_model = defect_model
        self.slice_size = slice_size
        self.overlap = overlap
        self.roi_conf = roi_conf
        self.defect_conf = defect_conf
        self.downsample_ratio = downsample_ratio
        self.device = device

        self.slicer = SmartSlicer(slice_size=slice_size, overlap=overlap)

    def infer(self, image: np.ndarray) -> Dict[str, Any]:
        """Run two-stage inference on a single image.

        Args:
            image: Input image (H, W, 3) uint8 BGR.

        Returns:
            Dict with keys: ``defects`` (list of detection dicts),
            ``total_time_ms``, ``stage1_time_ms``, ``stage2_time_ms``,
            ``num_roi_regions``, ``image_size``.
        """
        if image is None or not isinstance(image, np.ndarray):
            return {
                "defects": [], "total_time_ms": 0, "stage1_time_ms": 0,
                "stage2_time_ms": 0, "num_roi_regions": 0,
                "image_size": (0, 0), "error": "Invalid input image",
            }

        t_start = time.time()
        h, w = image.shape[:2]

        try:
            # ── Stage 1: ROI detection ──
            t1 = time.time()
            roi_boxes = self._detect_roi_regions(image, h, w)
            stage1_ms = (time.time() - t1) * 1000

            # ── Stage 2: Defect detection ──
            t2 = time.time()
            defects = self._detect_defects(image, roi_boxes)
            stage2_ms = (time.time() - t2) * 1000

            total_ms = (time.time() - t_start) * 1000

            return {
                "defects": defects,
                "total_time_ms": total_ms,
                "stage1_time_ms": stage1_ms,
                "stage2_time_ms": stage2_ms,
                "num_roi_regions": len(roi_boxes) if roi_boxes is not None else 0,
                "image_size": (w, h),
            }
        except Exception as e:
            total_ms = (time.time() - t_start) * 1000
            return {
                "defects": [], "total_time_ms": total_ms,
                "stage1_time_ms": 0, "stage2_time_ms": 0,
                "num_roi_regions": 0, "image_size": (w, h),
                "error": str(e),
            }

    def _detect_roi_regions(self, image, h, w):
        """Stage 1: Detect structural regions on downsampled image."""
        ratio = self.downsample_ratio
        small_h, small_w = h // ratio, w // ratio
        small_img = cv2.resize(image, (small_w, small_h))

        results = self.roi_model(
            small_img, conf=self.roi_conf, verbose=False, device=self.device
        )

        if len(results) == 0 or results[0].boxes is None:
            return None

        boxes = results[0].boxes.xyxy.cpu().numpy()
        # Scale back to original coordinates
        boxes *= ratio
        return boxes

    def _detect_defects(self, image, roi_boxes):
        """Stage 2: Detect defects on full-resolution ROI tiles."""
        if roi_boxes is not None and len(roi_boxes) > 0:
            tiles = list(self.slicer.roi_tiles(image, roi_boxes))
        else:
            tiles = list(self.slicer.iter_tiles(image))

        all_defects = []
        for tile, row, col, x0, y0 in tiles:
            results = self.defect_model(
                tile, conf=self.defect_conf, verbose=False, device=self.device
            )
            if len(results) == 0 or results[0].boxes is None:
                continue
            boxes = results[0].boxes
            for i in range(len(boxes.cls)):
                x, y, bw, bh = boxes.xywh[i].cpu().numpy()
                all_defects.append({
                    "box": {
                        "x": float((x0 + x) / image.shape[1]),
                        "y": float((y0 + y) / image.shape[0]),
                        "w": float(bw / image.shape[1]),
                        "h": float(bh / image.shape[0]),
                    },
                    "confidence": float(boxes.conf[i]),
                    "class_id": int(boxes.cls[i]),
                    "class_name": self.defect_model.names.get(
                        int(boxes.cls[i]), str(int(boxes.cls[i]))),
                    "source_tile": {"row": row, "col": col},
                })

        # Simple NMS to merge overlapping detections from adjacent tiles
        return self._merge_overlapping(all_defects)

    def _merge_overlapping(self, defects, iou_threshold=0.5):
        """Merge duplicate detections from overlapping tile regions."""
        if len(defects) < 2:
            return defects

        # Sort by confidence descending
        defects = sorted(defects, key=lambda d: d["confidence"], reverse=True)
        kept = []
        used = set()

        for i, d1 in enumerate(defects):
            if i in used:
                continue
            merged = {**d1}
            for j, d2 in enumerate(defects):
                if j <= i or j in used:
                    continue
                # Calculate IoU in normalized coords
                b1 = d1["box"]
                b2 = d2["box"]
                if d1["class_id"] == d2["class_id"] and box_iou(b1, b2) > iou_threshold:
                    used.add(j)
                    # Average the box coordinates
                    for k in ("x", "y", "w", "h"):
                        merged["box"][k] = (merged["box"][k] + b2[k]) / 2
                    merged["confidence"] = max(merged["confidence"], d2["confidence"])
            kept.append(merged)

        return kept

