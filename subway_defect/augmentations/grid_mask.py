"""GridMask data augmentation (Chen et al., 2020, arXiv:2001.04086).

Creates a regular grid of masked (zeroed or noise-filled) square regions,
forcing the model to not rely on any single spatial region.  This is
especially useful when defect cues are small and localised — the network
must learn redundant features across the object.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate_image(img: np.ndarray, name: str = "img") -> None:
    """Validate that *img* is a uint8, 3-channel BGR array."""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"{name} must be a numpy ndarray, got {type(img).__name__}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"{name} must be (H, W, 3), got shape {img.shape}"
        )
    if img.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8, got {img.dtype}")


# ---------------------------------------------------------------------------
# Core GridMask function
# ---------------------------------------------------------------------------


def grid_mask(
    img: np.ndarray,
    d_range: Tuple[int, int] = (64, 128),
    ratio: float = 0.6,
    mode: int = 0,
    prob: float = 0.5,
) -> np.ndarray:
    """Apply GridMask augmentation.

    Creates a regular grid of masked (zeroed or noised) square regions.
    Forces the model to not rely on any single spatial region.

    Args:
        img: ``(H, W, 3)`` uint8 BGR image.
        d_range: ``(min_d, max_d)`` grid cell size in pixels.  A value is
            sampled uniformly from this range on each call.
        ratio: Fraction of each cell that is masked (0–1).
        mode: ``0`` = zero-fill masked regions, ``1`` = random-noise-fill.
        prob: Probability of applying the mask.  With probability
            ``1 - prob`` the image is returned unchanged.

    Returns:
        Augmented image, same shape and dtype.
    """
    _validate_image(img)

    rng = np.random.default_rng()

    # 1. Stochastic gate
    if rng.random() > prob:
        return img.copy()

    h, w = img.shape[:2]

    # 2. Sample grid cell size
    min_d, max_d = d_range
    d = int(rng.integers(min_d, max_d + 1))
    d = max(d, 2)  # guard against degenerate sizes

    # 3. Mask unit size within each cell
    l = int(d * ratio + 0.5)
    l = max(1, min(l, d))  # clamp to [1, d]

    # 4. Random offset so the grid is not always aligned to the origin
    dx = int(rng.integers(0, d))
    dy = int(rng.integers(0, d))

    # 5. Build binary keep-mask (1 = keep, 0 = mask)
    #    For every d×d cell the top-left l×l sub-region is masked.
    mask = np.ones((h, w), dtype=np.float32)
    for y in range(-dy, h, d):
        y_start = max(y, 0)
        y_end = min(y + l, h)
        if y_end <= y_start:
            continue
        for x in range(-dx, w, d):
            x_start = max(x, 0)
            x_end = min(x + l, w)
            if x_end <= x_start:
                continue
            mask[y_start:y_end, x_start:x_end] = 0.0

    # 6/7. Apply mask
    out = img.copy()
    if mode == 0:
        # Zero-fill masked regions
        out = (out.astype(np.float32) * mask[..., None]).astype(np.uint8)
    elif mode == 1:
        # Random-noise-fill masked regions
        noise = rng.integers(0, 256, size=img.shape, dtype=np.uint8)
        mask_bool = np.broadcast_to(mask[..., None] == 0.0, img.shape)
        out[mask_bool] = noise[mask_bool]
    else:
        raise ValueError(f"mode must be 0 or 1, got {mode}")

    return out


# ---------------------------------------------------------------------------
# Ultralytics-compatible transform wrapper
# ---------------------------------------------------------------------------


class GridMaskTransform:
    """Ultralytics-compatible transform wrapper for GridMask.

    Can be inserted into the YOLO augmentation pipeline via the
    ``transforms`` argument or by monkey-patching.

    Args:
        d_range: ``(min_d, max_d)`` grid cell size in pixels.
        ratio: Fraction of each cell that is masked.
        mode: ``0`` = zero-fill, ``1`` = noise-fill.
        prob: Probability of applying the mask.
    """

    def __init__(
        self,
        d_range: Tuple[int, int] = (64, 128),
        ratio: float = 0.6,
        mode: int = 0,
        prob: float = 0.5,
    ) -> None:
        self.d_range = d_range
        self.ratio = ratio
        self.mode = mode
        self.prob = prob

    def __call__(self, labels: dict) -> dict:
        """Apply GridMask to an Ultralytics *labels* dict.

        The dict is expected to contain an ``"img"`` key with a
        ``(H, W, 3)`` uint8 BGR array.  The image is replaced in-place
        and the dict is returned.

        Args:
            labels: Ultralytics labels dictionary.

        Returns:
            The same dict with ``labels["img"]`` augmented.
        """
        labels["img"] = grid_mask(
            labels["img"],
            d_range=self.d_range,
            ratio=self.ratio,
            mode=self.mode,
            prob=self.prob,
        )
        return labels

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"d_range={self.d_range}, ratio={self.ratio}, "
            f"mode={self.mode}, prob={self.prob})"
        )
