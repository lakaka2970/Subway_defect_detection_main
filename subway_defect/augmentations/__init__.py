"""
ContactNet augmentation modules for subway catenary training.
"""

from .contactnet_copy_paste import ContactNetCopyPaste
from .scene import (
    glare_augment,
    motion_blur,
    night_augment,
    sunlitize,
    tunnelize,
    vibration_blur,
    weather_augment,
    white_balance_shift,
)

__all__ = [
    "ContactNetCopyPaste",
    "glare_augment",
    "motion_blur",
    "night_augment",
    "sunlitize",
    "tunnelize",
    "vibration_blur",
    "weather_augment",
    "white_balance_shift",
]
