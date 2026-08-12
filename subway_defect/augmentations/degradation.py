"""Imaging-chain degradation augmentations for subway catenary imagery.

Complements ``scene.py`` (environment simulation) with degradations that
happen between the scene and the recorded pixels:

* ``resolution_degrade`` — effective defect pixel count reduction
  (downscale → upscale), simulating distant / small-on-sensor targets.
* ``defocus_blur`` — lens defocus (Gaussian), simulating missed focus on
  moving inspection trains.
* ``jpeg_compress`` — low-quality JPEG re-encode, simulating transmission
  and archive compression artifacts.
* ``background_blur`` — depth-of-field simulation: defect regions stay
  sharp while everything outside the bounding boxes is strongly blurred.

All image-only functions accept and return ``np.ndarray`` (H, W, 3) uint8
BGR and never modify the input.  Box-aware functions take YOLO-format
labels ``[cls, cx, cy, bw, bh]`` and return them **unchanged** — these
augmentations never move or resize annotations.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np


def _validate_image(img: np.ndarray) -> None:
    """Validate that *img* is an (H, W, 3) uint8 ndarray."""
    if not isinstance(img, np.ndarray):
        raise ValueError(f"Expected a numpy ndarray, got {type(img).__name__}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected an image with shape (H, W, 3), got {img.shape}")
    if img.dtype != np.uint8:
        raise ValueError(f"Expected dtype uint8, got {img.dtype}")


def resolution_degrade(
    img: np.ndarray, down_factor: Optional[float] = None
) -> np.ndarray:
    """Reduce the effective pixel count of every object in the image.

    Downscales by ``down_factor`` (INTER_AREA) and upscales back to the
    original dimensions (INTER_LINEAR).  Defect targets lose fine detail
    exactly as they would when photographed from farther away, while the
    image size — and therefore all normalised bounding boxes — is preserved.

    Args:
        img: Input BGR image (H, W, 3) uint8.
        down_factor: Downscale factor.  ``None`` samples uniformly from
            [2.0, 4.0].  Values <= 1.0 return an unchanged copy.

    Returns:
        Degraded image, same shape and dtype.
    """
    _validate_image(img)
    if down_factor is None:
        down_factor = np.random.uniform(2.0, 4.0)
    if down_factor <= 1.0:
        return img.copy()

    h, w = img.shape[:2]
    small_w = max(1, int(round(w / down_factor)))
    small_h = max(1, int(round(h / down_factor)))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def defocus_blur(img: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """Simulate lens defocus with a Gaussian blur.

    Args:
        img: Input BGR image (H, W, 3) uint8.
        sigma: Gaussian sigma.  ``None`` samples uniformly from [1.5, 4.0].

    Returns:
        Blurred image, same shape and dtype.
    """
    _validate_image(img)
    if sigma is None:
        sigma = np.random.uniform(1.5, 4.0)
    if sigma <= 0:
        return img.copy()
    ksize = int(np.ceil(sigma * 3)) * 2 + 1  # ±3σ covers 99.7% of the kernel
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def jpeg_compress(img: np.ndarray, quality: Optional[int] = None) -> np.ndarray:
    """Re-encode as low-quality JPEG to simulate transmission artifacts.

    Args:
        img: Input BGR image (H, W, 3) uint8.
        quality: JPEG quality in [1, 100].  ``None`` samples uniformly from
            [35, 75].

    Returns:
        Compressed image, same shape and dtype.
    """
    _validate_image(img)
    if quality is None:
        quality = int(np.random.randint(35, 76))
    quality = int(np.clip(quality, 1, 100))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if out is None:
        raise RuntimeError("JPEG decoding failed")
    return out


def _boxes_to_mask(
    boxes: List[List[float]], w: int, h: int, pad_frac: float
) -> np.ndarray:
    """Rasterise YOLO boxes into a binary focus mask with per-box padding.

    Args:
        boxes: ``[cls, cx, cy, bw, bh]`` in normalised YOLO format.
        w: Image width in pixels.
        h: Image height in pixels.
        pad_frac: Padding as a fraction of each box's own size (applied to
            both axes).  Keeps box-edge pixels on the sharp side.

    Returns:
        uint8 mask (H, W), 255 inside the padded boxes.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        cx, cy, bw, bh = box[1], box[2], box[3], box[4]
        pw = bw * pad_frac
        ph = bh * pad_frac
        x1 = int((cx - bw / 2 - pw) * w)
        y1 = int((cy - bh / 2 - ph) * h)
        x2 = int((cx + bw / 2 + pw) * w)
        y2 = int((cy + bh / 2 + ph) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def background_blur(
    img: np.ndarray,
    boxes: List[List[float]],
    sigma: Optional[float] = None,
    pad_frac: float = 0.15,
) -> Tuple[np.ndarray, List[List[float]]]:
    """Simulate shallow depth of field: sharp defects, blurred background.

    Everything outside the (padded) defect bounding boxes is strongly
    Gaussian-blurred; inside stays sharp.  A feathered transition mask
    avoids a hard cut at box borders.

    Args:
        img: Input BGR image (H, W, 3) uint8.
        boxes: YOLO-format labels ``[cls, cx, cy, bw, bh]``.
        sigma: Blur sigma for the out-of-focus regions.  ``None`` samples
            uniformly from [4.0, 8.0].
        pad_frac: Box padding (fraction of box size) kept on the sharp side.

    Returns:
        ``(new_img, boxes)`` — blurred image and the **unchanged** labels.
        If ``boxes`` is empty the original image is returned unmodified
        (there is no focus anchor).
    """
    _validate_image(img)
    if not boxes:
        return img.copy(), list(boxes)
    if sigma is None:
        sigma = np.random.uniform(4.0, 8.0)

    h, w = img.shape[:2]
    focus_mask = _boxes_to_mask(boxes, w, h, pad_frac)

    # Feather the binary mask into soft [0, 1] focus weights.
    feather_k = max(3, int(np.ceil(sigma * 2)) * 2 + 1)
    weights = cv2.GaussianBlur(
        focus_mask.astype(np.float32) / 255.0, (feather_k, feather_k), 0
    )[..., None]

    ksize = int(np.ceil(sigma * 3)) * 2 + 1
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

    out = img.astype(np.float32) * weights + blurred.astype(np.float32) * (1.0 - weights)
    return np.clip(out, 0, 255).astype(np.uint8), list(boxes)
