"""
ContactNet augmentation modules for subway catenary training.
"""

from .background_replacement import BackgroundReplacer, replace_backgrounds_batch
from .contactnet_copy_paste import ContactNetCopyPaste
from .degradation import (
    background_blur,
    defocus_blur,
    jpeg_compress,
    resolution_degrade,
)
from .grid_mask import GridMaskTransform, grid_mask
from .local_contrast import (
    clahe_contrast,
    defect_glare,
    defect_shadow,
    elastic_deform,
)
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
    "BackgroundReplacer",
    "ContactNetCopyPaste",
    "GridMaskTransform",
    "background_blur",
    "clahe_contrast",
    "defect_glare",
    "defect_shadow",
    "defocus_blur",
    "elastic_deform",
    "glare_augment",
    "grid_mask",
    "jpeg_compress",
    "motion_blur",
    "night_augment",
    "replace_backgrounds_batch",
    "resolution_degrade",
    "sunlitize",
    "tunnelize",
    "vibration_blur",
    "weather_augment",
    "white_balance_shift",
]
