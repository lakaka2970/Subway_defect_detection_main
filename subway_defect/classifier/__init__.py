"""
Two-stage state classifier for catenary defect detection.

This module implements a lightweight CNN classifier that acts as a
second-stage verifier after YOLO detection. It classifies YOLO proposals
into defect states (normal/missing/loose/ambiguous) to suppress false
positives while maintaining recall.

Architecture: MobileNetV3-small backbone (< 5M params) with a custom
classification head. Input: 128x128 crop around YOLO proposal (1.5-2.0x
context).

Usage::

    from subway_defect.classifier import StateClassifier, ClassifierReasoner

    # Training
    classifier = StateClassifier(num_classes=2)
    classifier.fit(train_loader, val_loader, epochs=30)

    # Inference (plugs into TwoStagePipeline)
    reasoner = ClassifierReasoner("weights/classifier_cbhpm.pt")
    pipeline = TwoStagePipeline(..., state_reasoner=reasoner)
"""

from .model import StateClassifier
from .inference import ClassifierReasoner

__all__ = ["StateClassifier", "ClassifierReasoner"]
