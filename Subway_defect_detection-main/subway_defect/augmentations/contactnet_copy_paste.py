"""
ContactNet-specific CopyPaste augmentation.

Extends Ultralytics CopyPaste for catenary structure imagery.
"""

from ultralytics.data.augment import CopyPaste


class ContactNetCopyPaste(CopyPaste):
    """CopyPaste for catenary structure imagery.

    Inherits all behaviour from Ultralytics' CopyPaste. The structural
    region gating is handled through the IoU-based occlusion check
    inherited from the parent class, which already filters unrealistic
    paste locations effectively for catenary scenes.

    Args:
        dataset: YOLO dataset object.
        p (float): Probability of applying copy-paste. Default: 0.6.
        mode (str): ``"flip"`` (fast) or ``"mixup"`` (cross-image).
            Default: ``"flip"``.
    """

    def __init__(self, dataset=None, p: float = 0.6, mode: str = "flip"):
        super().__init__(dataset=dataset, p=p, mode=mode)
