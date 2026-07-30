"""
Cascade classifier for false-positive suppression.

After YOLO detection, crops each candidate bounding box (with 1.75x
context) and runs a per-class binary MobileNetV3-small classifier.
Detections whose classifier predicts "normal" above a confidence
threshold are rejected as false positives.

Designed to plug into both the tile-based batch inference script
(``scripts/run_inference.py``) and the interactive ``TwoStagePipeline``.

Usage::

    from subway_defect.pipeline.cascade import CascadeClassifier

    cascade = CascadeClassifier(
        weights_dir="weights",
        device="0",
        confidence_threshold=0.55,
    )
    # Filter a list of detection dicts (from nms_merge_detections)
    kept, rejected = cascade.filter_detections(image_bgr, detections)

Class-to-weight mapping (12-class dataset indexing)::

    SVHBNM (cls 2) → classifier_svhbnm.pt
    SVHBNL (cls 3) → classifier_svhbnl.pt
    SVHTNL (cls 4) → classifier_svhtnl.pt
    CBHPM  (cls 5) → classifier_cbhpm.pt
    CBVPM  (cls 6) → classifier_cbvpm.pt
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Default class → weight filename mapping (12-class dataset indexing) ──
DEFAULT_CLASSIFIER_MAP: Dict[int, str] = {
    0: "classifier_vhb_level1.pt",    # VHBNM → hierarchical L1 (normal vs defective)
    1: "classifier_vhb_level1.pt",    # VHBNL → hierarchical L1 (shared)
    2: "classifier_svhbnm.pt",        # SVHBNM
    3: "classifier_svhbnl.pt",        # SVHBNL
    4: "classifier_svhtnl.pt",        # SVHTNL
    5: "classifier_cbhpm.pt",         # CBHPM
    6: "classifier_cbvpm.pt",         # CBVPM
    # 7: RHTBNM — pending data collection
    # 8: RHTBNL — pending data collection
    # 9: BSBM — pending data collection
    10: "classifier_insd.pt",         # INSD (high FP class, needs FP reduction)
    # 11: DRPS — already excellent, no classifier needed
}

# Hierarchical classifier config: classes that use L1 → L2 two-stage verification
HIERARCHICAL_MAP: Dict[int, dict] = {
    0: {"l1": "classifier_vhb_level1.pt", "l2": "classifier_vhb_level2.pt"},  # VHBNM
    1: {"l1": "classifier_vhb_level1.pt", "l2": "classifier_vhb_level2.pt"},  # VHBNL
}

# States that indicate the detection is a false positive
REJECT_STATES = {"normal", "background", "negative"}


class CascadeClassifier:
    """Multi-class cascade filter for YOLO false-positive suppression.

    Loads one binary classifier per YOLO class that has a trained
    verifier.  At inference time, each detection whose class has a
    classifier is cropped (with context), classified, and optionally
    rejected.

    Args:
        weights_dir: Directory containing classifier ``.pt`` files.
        class_map: Mapping from YOLO class ID → weight filename.
            Defaults to ``DEFAULT_CLASSIFIER_MAP``.
        device: Torch device string (e.g. ``"0"`` or ``"cpu"``).
        confidence_threshold: Minimum classifier confidence for
            rejection.  If the classifier predicts a reject-state with
            confidence ≥ this value, the detection is suppressed.
        context_scale: Bbox expansion factor for cropping. Default 1.75.
        enabled: Master switch — when ``False``, :meth:`filter_detections`
            is a no-op passthrough.
    """

    def __init__(
        self,
        weights_dir: str | Path = "weights",
        class_map: Optional[Dict[int, str]] = None,
        device: str = "0",
        confidence_threshold: float = 0.55,
        context_scale: float = 1.75,
        enabled: bool = True,
    ):
        self.weights_dir = Path(weights_dir)
        self.class_map = class_map or dict(DEFAULT_CLASSIFIER_MAP)
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.context_scale = context_scale
        self.enabled = enabled

        # Lazily loaded reasoners keyed by class ID
        self._reasoners: Dict[int, Any] = {}
        self._load_failures: Dict[int, str] = {}

        if self.enabled:
            self._preload_classifiers()

    # ── Public API ────────────────────────────────────────────────────────

    def filter_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Filter detections through per-class cascade classifiers.

        Args:
            image: Full-resolution BGR image (H, W, 3) uint8 — the same
                image the detections were produced from.
            detections: List of detection dicts, each containing at
                minimum ``x1, y1, x2, y2, conf, cls`` keys (pixel
                coordinates, as produced by ``nms_merge_detections``).

        Returns:
            (kept, rejected): Two lists of detection dicts.  Rejected
            dicts gain extra keys ``cascade_state`` and
            ``cascade_confidence`` for diagnostics.
        """
        if not self.enabled or not detections:
            return list(detections), []

        kept: List[Dict[str, Any]] = []
        rejected: List[Dict[str, Any]] = []

        for det in detections:
            cls_id = int(det["cls"])

            # ── Hierarchical path (VHBNM/VHBNL: L1 → L2) ──
            if cls_id in HIERARCHICAL_MAP:
                result = self._hierarchical_classify(image, det, cls_id)
                if result is not None:
                    state, state_conf = result
                    det["cascade_state"] = state
                    det["cascade_confidence"] = state_conf
                    if state in REJECT_STATES and state_conf >= self.confidence_threshold:
                        rejected.append(det)
                    else:
                        kept.append(det)
                    continue

            # ── Standard single-classifier path ──
            reasoner = self._get_reasoner(cls_id)

            if reasoner is None:
                # No classifier for this class — keep unconditionally
                kept.append(det)
                continue

            # Build the normalized box dict expected by ClassifierReasoner
            h, w = image.shape[:2]
            x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            box_norm = {"x": cx, "y": cy, "w": bw, "h": bh}

            # Run classifier
            result = reasoner(image, {"box": box_norm})
            state = str(result.get("state", "")).lower()
            state_conf = float(result.get("confidence", 0.0))

            det["cascade_state"] = state
            det["cascade_confidence"] = state_conf

            if state in REJECT_STATES and state_conf >= self.confidence_threshold:
                rejected.append(det)
            else:
                kept.append(det)

        return kept, rejected

    @property
    def available_classes(self) -> List[int]:
        """Class IDs with successfully loaded classifiers."""
        return sorted(self._reasoners.keys())

    @property
    def failed_classes(self) -> Dict[int, str]:
        """Class IDs whose classifier failed to load, with error messages."""
        return dict(self._load_failures)

    def summary(self) -> str:
        """Human-readable summary of cascade state."""
        lines = [f"CascadeClassifier (enabled={self.enabled})"]
        lines.append(f"  weights_dir: {self.weights_dir}")
        lines.append(f"  confidence_threshold: {self.confidence_threshold}")
        lines.append(f"  context_scale: {self.context_scale}")
        lines.append(f"  device: {self.device}")
        for cls_id in sorted(self.class_map):
            fname = self.class_map[cls_id]
            if cls_id in self._reasoners:
                lines.append(f"  cls {cls_id} ({fname}): ✓ loaded")
            elif cls_id in self._load_failures:
                lines.append(f"  cls {cls_id} ({fname}): ✗ {self._load_failures[cls_id]}")
            else:
                lines.append(f"  cls {cls_id} ({fname}): not loaded")
        return "\n".join(lines)

    # ── Internal ──────────────────────────────────────────────────────────

    def _hierarchical_classify(
        self, image: np.ndarray, det: Dict[str, Any], cls_id: int,
    ) -> Optional[Tuple[str, float]]:
        """Two-stage hierarchical classification for VHBNM/VHBNL.

        Level 1: normal vs defective (high-precision gate)
        Level 2: missing vs loose (defect type discrimination)

        Returns (state, confidence) or None if classifiers unavailable.
        """
        hier = HIERARCHICAL_MAP.get(cls_id)
        if hier is None:
            return None

        h, w = image.shape[:2]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        box_norm = {
            "x": (x1 + x2) / 2.0 / w, "y": (y1 + y2) / 2.0 / h,
            "w": (x2 - x1) / w, "h": (y2 - y1) / h,
        }

        # Level 1: normal vs defective
        l1 = self._get_hierarchical_reasoner(hier["l1"])
        if l1 is None:
            return None
        l1_result = l1(image, {"box": box_norm})
        l1_state = str(l1_result.get("state", "")).lower()
        l1_conf = float(l1_result.get("confidence", 0.0))

        if l1_state == "normal" and l1_conf >= self.confidence_threshold:
            return ("normal", l1_conf)

        # Level 2: missing vs loose
        l2 = self._get_hierarchical_reasoner(hier["l2"])
        if l2 is None:
            return ("defective", l1_conf)
        l2_result = l2(image, {"box": box_norm})
        l2_state = str(l2_result.get("state", "")).lower()
        l2_conf = float(l2_result.get("confidence", 0.0))

        # Map L2 result to a cascade state
        if l2_state == "missing":
            return ("missing", l2_conf)
        elif l2_state == "loose":
            return ("loose", l2_conf)
        return ("defective", min(l1_conf, l2_conf))

    def _get_hierarchical_reasoner(self, fname: str):
        """Get or lazily load a reasoner by filename (for hierarchical classifiers)."""
        key = f"_hier_{fname}"
        if hasattr(self, key):
            return getattr(self, key)

        weight_path = self.weights_dir / fname
        if not weight_path.exists():
            logger.warning("Cascade: hierarchical weight not found: %s", weight_path)
            setattr(self, key, None)
            return None

        try:
            from subway_defect.classifier.inference import ClassifierReasoner

            reasoner = ClassifierReasoner(
                weights_path=weight_path,
                context_scale=self.context_scale,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
            )
            setattr(self, key, reasoner)
            logger.info("Cascade: loaded hierarchical classifier %s", fname)
            return reasoner
        except Exception as e:
            logger.warning("Cascade: hierarchical load error %s: %s", fname, e)
            setattr(self, key, None)
            return None

    def _preload_classifiers(self) -> None:
        """Attempt to load all classifiers at init time."""
        for cls_id, fname in self.class_map.items():
            self._get_reasoner(cls_id)

    def _get_reasoner(self, cls_id: int):
        """Get or lazily load the ClassifierReasoner for a class ID."""
        if cls_id in self._reasoners:
            return self._reasoners[cls_id]
        if cls_id in self._load_failures:
            return None  # Already tried and failed

        fname = self.class_map.get(cls_id)
        if fname is None:
            return None

        weight_path = self.weights_dir / fname
        if not weight_path.exists():
            msg = f"weight file not found: {weight_path}"
            logger.warning("Cascade: %s", msg)
            self._load_failures[cls_id] = msg
            return None

        try:
            from subway_defect.classifier.inference import ClassifierReasoner

            reasoner = ClassifierReasoner(
                weights_path=weight_path,
                context_scale=self.context_scale,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
            )
            self._reasoners[cls_id] = reasoner
            logger.info("Cascade: loaded classifier for cls %d from %s", cls_id, fname)
            return reasoner
        except Exception as e:
            msg = f"load error: {e}"
            logger.warning("Cascade: cls %d %s", cls_id, msg)
            self._load_failures[cls_id] = msg
            return None
