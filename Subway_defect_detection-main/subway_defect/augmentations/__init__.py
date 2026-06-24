"""
ContactNet augmentation modules for subway catenary training.
"""

from .contactnet_copy_paste import ContactNetCopyPaste
from .scene import motion_blur, sunlitize, tunnelize, weather_augment

__all__ = [
    "ContactNetCopyPaste",
    "motion_blur",
    "sunlitize",
    "tunnelize",
    "weather_augment",
]
