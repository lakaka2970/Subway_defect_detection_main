"""Local contrast / geometry augmentations for subway catenary imagery.

Three fresh augmentations for the train_data_3 pipeline, complementing
``scene.py`` (environment) and ``degradation.py`` (imaging chain):

* ``clahe_contrast`` — contrast-limited adaptive histogram equalisation on
  the LAB luminance channel.  Reveals low-contrast defect texture without
  the global wash-out of plain histogram equalisation.
* ``elastic_deform`` — smooth, Gaussian-filtered random displacement field
  that warps both the image **and** the bounding boxes.  Simulates the
  subtle geometric variation of a moving inspection camera and different
  viewing angles on loose / small parts.  The only augmentation here that
  moves annotations.
* ``defect_shadow`` — places a soft occlusion shadow **near** (not on) each
  defect with a small per-box probability, teaching the model to recognise
  defects under partial illumination while keeping the box itself clean.

All functions accept and return ``np.ndarray`` (H, W, 3) uint8 BGR and
never mutate their input.  Boxes use YOLO ``[cls, cx, cy, bw, bh]`` format.
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


def clahe_contrast(
    img: np.ndarray,
    clip_limit: Optional[float] = None,
    tile: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """CLAHE on the LAB luminance channel (local contrast perturbation).

    Args:
        img: Input BGR image (H, W, 3) uint8.
        clip_limit: CLAHE clip limit.  ``None`` samples uniformly from
            [1.5, 4.0].
        tile: CLAHE tile grid size.  ``None`` samples from {(4,4), (8,8),
            (12,12)}.

    Returns:
        Contrast-enhanced image, same shape and dtype.
    """
    _validate_image(img)
    img = img.copy()
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if clip_limit is None:
        clip_limit = float(np.random.uniform(1.5, 4.0))
    if tile is None:
        s = int(np.random.choice([4, 8, 12]))
        tile = (s, s)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def _smooth_field(field: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian-smooth a displacement field (kernel sized to ±3σ)."""
    ksize = int(np.ceil(sigma * 3)) * 2 + 1
    return cv2.GaussianBlur(field, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def _sample_field(field: np.ndarray, px: float, py: float) -> float:
    """Bilinear-sample a (H, W) field at float pixel coord (px, py)."""
    h, w = field.shape
    px = float(np.clip(px, 0.0, w - 1.0))
    py = float(np.clip(py, 0.0, h - 1.0))
    x0, y0 = int(np.floor(px)), int(np.floor(py))
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    fx, fy = px - x0, py - y0
    return float(
        field[y0, x0] * (1.0 - fx) * (1.0 - fy)
        + field[y0, x1] * fx * (1.0 - fy)
        + field[y1, x0] * (1.0 - fx) * fy
        + field[y1, x1] * fx * fy
    )


def elastic_deform(
    img: np.ndarray,
    boxes: Optional[List[List[float]]] = None,
    alpha: Optional[float] = None,
    sigma: Optional[float] = None,
) -> Tuple[np.ndarray, List[List[float]]]:
    """Elastically warp the image and its bounding boxes together.

    A random displacement field (uniform in [-1, 1]) is Gaussian-smoothed
    by ``sigma`` and scaled by ``alpha`` (pixel units).  The image is
    remapped and each box's four corners are pushed through the same field;
    the new bounding box is the axis-aligned rectangle of the displaced
    corners.  Boxes that shrink below half their original area are dropped.

    Args:
        img: Input BGR image (H, W, 3) uint8.
        boxes: YOLO-format labels ``[cls, cx, cy, bw, bh]`` (normalised).
            ``None``/empty still warps the image, returning ``[]``.
        alpha: Displacement magnitude in pixels.  ``None`` samples [2.0, 5.0].
        sigma: Gaussian smoothing radius.  ``None`` samples [8.0, 20.0].

    Returns:
        ``(new_img, new_boxes)`` — warped image and transformed labels.
    """
    _validate_image(img)
    boxes = list(boxes or [])
    if alpha is None:
        alpha = float(np.random.uniform(2.0, 5.0))
    if sigma is None:
        sigma = float(np.random.uniform(8.0, 20.0))

    h, w = img.shape[:2]

    dx = _smooth_field(
        np.random.uniform(-1.0, 1.0, (h, w)).astype(np.float32), sigma
    ) * alpha
    dy = _smooth_field(
        np.random.uniform(-1.0, 1.0, (h, w)).astype(np.float32), sigma
    ) * alpha

    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_grid + dx).astype(np.float32)
    map_y = (y_grid + dy).astype(np.float32)
    out = cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    new_boxes: List[List[float]] = []
    for box in boxes:
        cls, cx, cy, bw, bh = box[0], box[1], box[2], box[3], box[4]
        x1 = (cx - bw / 2.0) * w
        x2 = (cx + bw / 2.0) * w
        y1 = (cy - bh / 2.0) * h
        y2 = (cy + bh / 2.0) * h

        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        displaced = []
        for px, py in corners:
            nx = px + _sample_field(dx, px, py)
            ny = py + _sample_field(dy, px, py)
            displaced.append((nx, ny))

        nxs = [p[0] for p in displaced]
        nys = [p[1] for p in displaced]
        nx1, nx2 = min(nxs), max(nxs)
        ny1, ny2 = min(nys), max(nys)

        # Re-normalise and clamp to the image frame.
        ncx = (nx1 + nx2) / 2.0 / w
        ncy = (ny1 + ny2) / 2.0 / h
        nbw = max(0.0, (nx2 - nx1) / w)
        nbh = max(0.0, (ny2 - ny1) / h)
        ncx = float(np.clip(ncx, 0.0, 1.0))
        ncy = float(np.clip(ncy, 0.0, 1.0))

        old_area = bw * bh
        new_area = nbw * nbh
        if new_area <= 0 or old_area <= 0:
            continue
        if new_area < 0.5 * old_area:
            continue  # box collapsed under deformation → drop

        new_boxes.append([cls, ncx, ncy, nbw, nbh])

    return out, new_boxes


def defect_glare(
    img: np.ndarray,
    boxes: List[List[float]],
    p_per_box: float = 0.5,
) -> np.ndarray:
    """Place specular glare **near** defect locations.

    Reflectance from polished metal is most misleading where the defect
    actually is, so glare blobs/streaks are anchored to (the vicinity of)
    each box rather than scattered randomly over the frame.  The box centre
    is offset by a random amount within a couple of box-sizes, keeping the
    highlight adjacent to — not necessarily on — the defect.

    Args:
        img: Input BGR image (H, W, 3) uint8.
        boxes: YOLO-format labels ``[cls, cx, cy, bw, bh]``.
        p_per_box: Per-box probability of adding a glare blob.

    Returns:
        Augmented image, same shape and dtype.  Boxes are unchanged.
    """
    _validate_image(img)
    if not boxes:
        return img.copy()
    img = img.copy()
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    for box in boxes:
        if np.random.random() >= p_per_box:
            continue
        cx, cy, bw, bh = box[1], box[2], box[3], box[4]
        box_w = bw * w
        box_h = bh * h
        # Offset from the box centre by up to ~1.5 box sizes.
        ox = (np.random.uniform(-1.5, 1.5) * box_w)
        oy = (np.random.uniform(-1.5, 1.5) * box_h)
        px = float(np.clip(cx * w + ox, 1, w - 1))
        py = float(np.clip(cy * h + oy, 1, h - 1))

        radius = max(8, int(min(box_w, box_h) * np.random.uniform(0.8, 1.6)))
        intensity = float(np.random.uniform(0.35, 0.85))

        yy, xx = np.ogrid[:h, :w]
        dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
        blob = np.exp(-(dist ** 2) / (2.0 * (radius / 2.0) ** 2))
        blob = np.clip(blob * intensity, 0, 1)

        tint = np.array([
            np.random.uniform(0.9, 1.0),
            np.random.uniform(0.9, 1.0),
            np.random.uniform(0.95, 1.0),
        ], dtype=np.float32).reshape(1, 1, 3)
        out = out * (1 - blob[..., None]) + 255.0 * blob[..., None] * tint

    return np.clip(out, 0, 255).astype(np.uint8)


def defect_shadow(
    img: np.ndarray,
    boxes: List[List[float]],
    p_per_box: float = 0.15,
) -> np.ndarray:
    """Place soft occlusion shadows near (not on) each defect.

    For each box, with probability ``p_per_box``, a soft dark bar is drawn
    just outside the box (top/bottom/left/right, chosen at random) to
    simulate partial illumination around the defect.  The box itself is
    never darkened — only the surrounding metal.

    Args:
        img: Input BGR image (H, W, 3) uint8.
        boxes: YOLO-format labels ``[cls, cx, cy, bw, bh]``.
        p_per_box: Per-box probability of adding a shadow.

    Returns:
        Augmented image, same shape and dtype.  Boxes are unchanged.
    """
    _validate_image(img)
    if not boxes:
        return img.copy()
    img = img.copy()
    h, w = img.shape[:2]
    out = img.astype(np.float32)

    for box in boxes:
        if np.random.random() >= p_per_box:
            continue
        cx, cy, bw, bh = box[1], box[2], box[3], box[4]
        bx1 = int((cx - bw / 2.0) * w)
        bx2 = int((cx + bw / 2.0) * w)
        by1 = int((cy - bh / 2.0) * h)
        by2 = int((cy + bh / 2.0) * h)
        box_w = max(1, bx2 - bx1)
        box_h = max(1, by2 - by1)

        side = np.random.choice(["top", "bottom", "left", "right"])
        gap = np.random.randint(2, max(3, int(box_h * 0.6)))

        # Shadow bar dimensions (long axis along the box edge).
        bar_len = int(box_w * np.random.uniform(1.5, 3.0))
        bar_thick = int(box_h * np.random.uniform(0.8, 1.5))

        if side == "top":
            sx = int(np.clip(bx1 + box_w / 2 - bar_len / 2, 0, w - 1))
            ex = min(w, sx + bar_len)
            ey = max(0, by1 - gap)
            sy = max(0, ey - bar_thick)
        elif side == "bottom":
            sx = int(np.clip(bx1 + box_w / 2 - bar_len / 2, 0, w - 1))
            ex = min(w, sx + bar_len)
            sy = min(h, by2 + gap)
            ey = min(h, sy + bar_thick)
        elif side == "left":
            sx = max(0, bx1 - gap - bar_thick)
            ex = max(0, bx1 - gap)
            sy = int(np.clip(by1 + box_h / 2 - bar_len / 2, 0, h - 1))
            ey = min(h, sy + bar_len)
        else:  # right
            sx = min(w, bx2 + gap)
            ex = min(w, bx2 + gap + bar_thick)
            sy = int(np.clip(by1 + box_h / 2 - bar_len / 2, 0, h - 1))
            ey = min(h, sy + bar_len)

        if ex <= sx or ey <= sy:
            continue

        strength = float(np.random.uniform(0.25, 0.5))
        bar = np.ones((h, w), dtype=np.float32)
        bar[sy:ey, sx:ex] = 1.0 - strength
        # Feather the edges so the shadow is soft.
        k = max(3, (bar_thick // 2) * 2 + 1)
        bar = cv2.GaussianBlur(bar, (k, k), 0)
        out = out * bar[..., None]

    return np.clip(out, 0, 255).astype(np.uint8)
