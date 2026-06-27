"""
ContactNet augmentation modules for subway catenary training.

Scene augmentations (6 functions):
  - tunnelize: dark tunnel lighting simulation
  - sunlitize: bright outdoor sunlight + shadows
  - motion_blur: directional vehicle-motion blur
  - weather_augment: rain/fog overlay
  - vibration_blur: high-frequency micro-vibration (Gaussian + pixel shift)
  - white_balance_shift: camera white-balance / colour temperature drift

Defect copy-paste (offline, small-target only):
  - copy_paste_defects: extract <32px defect patches + paste to other images
"""

from .contactnet_copy_paste import ContactNetCopyPaste
from .defect_copy_paste import copy_paste_defects
from .scene import (
    motion_blur,
    sunlitize,
    tunnelize,
    vibration_blur,
    weather_augment,
    white_balance_shift,
)

__all__ = [
    "ContactNetCopyPaste",
    "copy_paste_defects",
    "motion_blur",
    "sunlitize",
    "tunnelize",
    "vibration_blur",
    "weather_augment",
    "white_balance_shift",
]
