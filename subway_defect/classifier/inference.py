"""
Inference wrapper for the state classifier.

Implements the StateReasoner protocol expected by TwoStagePipeline:
    Callable[[np.ndarray, Dict[str, Any]], Dict[str, Any]]

The reasoner crops the detection region from the tile with 1.5-2.0x
context, runs the classifier, and returns state + confidence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from .model import StateClassifier
from .dataset import INPUT_SIZE, IMAGENET_MEAN, IMAGENET_STD


# State labels for different classifier configurations
CBHPM_STATES = ["normal", "missing"]
VHBNM_VHBNL_STATES = ["normal", "missing", "loose", "ambiguous"]

# States that indicate false positive (should be rejected)
REJECT_STATES = {"normal", "background", "negative"}


class ClassifierReasoner:
    """State classifier reasoner for TwoStagePipeline integration.

    Implements the StateReasoner protocol:
        __call__(tile: np.ndarray, detection: Dict) -> Dict

    Args:
        weights_path: Path to trained classifier checkpoint.
        class_names: State class names (e.g., ["normal", "missing"]).
        context_scale: Context expansion factor around bbox (default: 1.75).
        device: CUDA device string.
        reject_states: Set of state names that trigger FP rejection.
        confidence_threshold: Minimum confidence for state-based rejection.
    """

    def __init__(
        self,
        weights_path: str | Path,
        class_names: Optional[List[str]] = None,
        context_scale: float = 1.75,
        device: str = "0",
        reject_states: Optional[set] = None,
        confidence_threshold: float = 0.50,
    ):
        if device not in ("", "cpu") and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(
            f"cuda:{device}" if device not in ("", "cpu") else "cpu"
        )
        self.context_scale = context_scale
        self.reject_states = reject_states or REJECT_STATES
        self.confidence_threshold = confidence_threshold

        # Load model
        self.model = StateClassifier.load(weights_path, device=str(self.device))
        self.model.to(self.device)
        self.model.eval()

        # Infer class names from checkpoint or use provided
        if class_names:
            self.class_names = class_names
        else:
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            meta = checkpoint.get("meta", {})
            self.class_names = meta.get("class_names", CBHPM_STATES)

        self.num_classes = len(self.class_names)

        # Preprocessing transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __call__(self, tile: np.ndarray, detection: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a detection proposal.

        Args:
            tile: Image tile (H, W, 3) uint8 BGR containing the detection.
            detection: Detection dict with 'box' key containing normalized
                {x, y, w, h} coordinates.

        Returns:
            Dict with 'state' (str) and 'confidence' (float) keys.
        """
        try:
            crop = self._extract_context_crop(tile, detection["box"])
            if crop is None or crop.size == 0:
                return {"state": "unknown", "confidence": 0.0}

            # Preprocess
            # Convert BGR to RGB for the transform
            crop_rgb = crop[:, :, ::-1].copy()
            tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

            # Inference
            with torch.no_grad():
                logits = self.model(tensor)
                probs = F.softmax(logits, dim=1)
                confidence, predicted = torch.max(probs, dim=1)

            state_idx = predicted.item()
            state_conf = confidence.item()
            state_name = self.class_names[state_idx] if state_idx < len(self.class_names) else "unknown"

            return {
                "state": state_name,
                "confidence": state_conf,
                "class_probs": {
                    self.class_names[i]: float(probs[0, i])
                    for i in range(self.num_classes)
                },
            }

        except Exception:
            return {"state": "unknown", "confidence": 0.0}

    def _extract_context_crop(
        self, tile: np.ndarray, box: Dict[str, float]
    ) -> Optional[np.ndarray]:
        """Extract a context crop around the detection bbox.

        Args:
            tile: Full tile image (H, W, 3).
            box: Normalized box dict {x, y, w, h} (center + size).

        Returns:
            Cropped region as np.ndarray, or None if invalid.
        """
        h, w = tile.shape[:2]

        # Convert normalized center-format to pixel coords
        cx = box["x"] * w
        cy = box["y"] * h
        bw = box["w"] * w
        bh = box["h"] * h

        # Expand by context_scale
        ctx_w = bw * self.context_scale
        ctx_h = bh * self.context_scale

        # Ensure minimum crop size
        ctx_w = max(ctx_w, 32)
        ctx_h = max(ctx_h, 32)

        # Compute crop bounds
        x1 = int(max(0, cx - ctx_w / 2))
        y1 = int(max(0, cy - ctx_h / 2))
        x2 = int(min(w, cx + ctx_w / 2))
        y2 = int(min(h, cy + ctx_h / 2))

        if x2 <= x1 or y2 <= y1:
            return None

        return tile[y1:y2, x1:x2]

    def batch_classify(
        self, tile: np.ndarray, detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Classify multiple detections in batch for efficiency.

        Args:
            tile: Image tile (H, W, 3) uint8 BGR.
            detections: List of detection dicts.

        Returns:
            List of state dicts (one per detection).
        """
        crops = []
        valid_indices = []

        for i, det in enumerate(detections):
            crop = self._extract_context_crop(tile, det["box"])
            if crop is not None and crop.size > 0:
                crop_rgb = crop[:, :, ::-1].copy()
                crops.append(self.transform(crop_rgb))
                valid_indices.append(i)

        results = [{"state": "unknown", "confidence": 0.0}] * len(detections)

        if not crops:
            return results

        # Batch inference
        batch = torch.stack(crops).to(self.device)
        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)

        for batch_idx, det_idx in enumerate(valid_indices):
            state_idx = predictions[batch_idx].item()
            state_conf = confidences[batch_idx].item()
            state_name = self.class_names[state_idx] if state_idx < len(self.class_names) else "unknown"
            results[det_idx] = {
                "state": state_name,
                "confidence": state_conf,
            }

        return results
