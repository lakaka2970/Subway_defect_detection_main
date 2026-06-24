"""
Weighted Boxes Fusion (WBF) for dual-GPU ensemble results.

Reference: "Weighted Boxes Fusion: Ensembling boxes from different
models" (arXiv:1910.13302).
"""

from typing import Dict, List

import numpy as np


class WBFFusion:
    """Weighted Boxes Fusion for dual-model defect detection ensemble.

    Fuses detection results from two models (ECA variant + P2 variant)
    to reduce false positives while maintaining recall.

    Args:
        iou_threshold: IoU threshold for matching boxes across models.
            Default: 0.55.
        dual_conf_threshold: Minimum avg confidence when both models
            detect the same object. Default: 0.50.
        single_conf_threshold: Minimum confidence when only one model
            detects an object. Default: 0.75.
        final_conf_threshold: Minimum fused confidence for output.
            Default: 0.60.
        weights: Optional (w1, w2) weights for model A and model B.
            Default: equal weighting (1.0, 1.0).
    """

    def __init__(
        self,
        iou_threshold: float = 0.55,
        dual_conf_threshold: float = 0.50,
        single_conf_threshold: float = 0.75,
        final_conf_threshold: float = 0.60,
        weights: tuple = (1.0, 1.0),
    ):
        self.iou_threshold = iou_threshold
        self.dual_conf_threshold = dual_conf_threshold
        self.single_conf_threshold = single_conf_threshold
        self.final_conf_threshold = final_conf_threshold
        self.weights = weights

    def fuse(self, detections_a: List[Dict],
             detections_b: List[Dict]) -> List[Dict]:
        """Fuse two lists of detection dicts.

        Args:
            detections_a: Detections from model A (ECA variant).
            detections_b: Detections from model B (P2 variant).

        Returns:
            Fused detection list.
        """
        w_a, w_b = self.weights

        # Build lists with source tracking
        all_dets = []
        for d in detections_a:
            d = dict(d)
            d["_source"] = "a"
            all_dets.append(d)
        for d in detections_b:
            d = dict(d)
            d["_source"] = "b"
            all_dets.append(d)

        if not all_dets:
            return []

        # Sort by confidence desc
        all_dets.sort(key=lambda d: d["confidence"], reverse=True)

        fused = []
        matched_indices = set()

        for i, det_i in enumerate(all_dets):
            if i in matched_indices:
                continue

            # Find matching detections from other model(s)
            cluster = [det_i]
            cluster_indices = [i]

            for j in range(i + 1, len(all_dets)):
                if j in matched_indices:
                    continue
                det_j = all_dets[j]
                if det_i["class_id"] != det_j["class_id"]:
                    continue
                if self._box_iou(det_i["box"], det_j["box"]) > self.iou_threshold:
                    cluster.append(det_j)
                    cluster_indices.append(j)

            # Mark all cluster members as matched
            for idx in cluster_indices:
                matched_indices.add(idx)

            # Validate and fuse cluster
            has_a = any(d["_source"] == "a" for d in cluster)
            has_b = any(d["_source"] == "b" for d in cluster)
            both_detected = has_a and has_b

            # Compute weighted avg confidence
            avg_conf = np.mean([d["confidence"] for d in cluster])

            if both_detected and avg_conf >= self.dual_conf_threshold:
                valid = True
            elif not both_detected and avg_conf >= self.single_conf_threshold:
                valid = True
            else:
                valid = False

            if not valid:
                continue

            # Fuse box coordinates
            fused_box = self._fuse_boxes(cluster)

            if avg_conf < self.final_conf_threshold:
                continue

            fused.append({
                "box": fused_box,
                "confidence": float(avg_conf),
                "class_id": det_i["class_id"],
                "class_name": det_i["class_name"],
                "dual_detected": both_detected,
            })

        return fused

    def _fuse_boxes(self, cluster):
        """Weighted average of box coordinates."""
        total_w = 0.0
        for d in cluster:
            w = self.weights[0] if d["_source"] == "a" else self.weights[1]
            total_w += w * d["confidence"]

        fused = {"x": 0.0, "y": 0.0, "w": 0.0, "h": 0.0}
        for d in cluster:
            w = self.weights[0] if d["_source"] == "a" else self.weights[1]
            weight = w * d["confidence"] / total_w if total_w > 0 else 1.0
            for k in ("x", "y", "w", "h"):
                fused[k] += d["box"][k] * weight

        return fused

    @staticmethod
    def _box_iou(b1, b2):
        """IoU of two normalized xywh boxes."""
        x1, y1, w1, h1 = b1["x"], b1["y"], b1["w"], b1["h"]
        x2, y2, w2, h2 = b2["x"], b2["y"], b2["w"], b2["h"]

        l1, t1 = x1 - w1 / 2, y1 - h1 / 2
        r1, b1 = x1 + w1 / 2, y1 + h1 / 2
        l2, t2 = x2 - w2 / 2, y2 - h2 / 2
        r2, b2 = x2 + w2 / 2, y2 + h2 / 2

        inter_l = max(l1, l2)
        inter_t = max(t1, t2)
        inter_r = min(r1, r2)
        inter_b = min(b1, b2)

        if inter_r <= inter_l or inter_b <= inter_t:
            return 0.0

        inter_area = (inter_r - inter_l) * (inter_b - inter_t)
        area1 = w1 * h1
        area2 = w2 * h2
        return inter_area / (area1 + area2 - inter_area)
