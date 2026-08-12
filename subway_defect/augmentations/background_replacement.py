"""ML-enhanced background replacement augmentation.

Breaks background-label correlation by extracting defect foregrounds with
GrabCut (graph-cut segmentation) and compositing them onto diverse background
images via Poisson seamless cloning.  This directly addresses workshop
overfitting: all defect images share a limited set of workshop backgrounds,
but the model deploys in subway tunnels with very different surroundings.

Pipeline per image:
1. Union bounding box of all defect boxes → expand by ``fg_padding``
2. GrabCut (GC_INIT_WITH_RECT) at reduced scale (``grabcut_max_side``) →
   binary foreground mask, upsampled back to full resolution.  Full-image
   GrabCut on 8K inspection frames is prohibitively slow; the foreground
   boundary is smooth enough that a ≤1024 px working scale loses nothing.
3. Morphological cleanup (open + close, 3×3 kernel)
4. ``min_fg_ratio`` sanity check (fall back to rect mask if GrabCut failed)
5. ``cv2.seamlessClone`` (NORMAL_CLONE) onto a randomly chosen background —
   only when the image is at most ``max_poisson_pixels`` large; beyond that
   the Poisson solver is too slow / memory-hungry and a feathered
   alpha-blend is used instead.
6. Feathered alpha-blend fallback if Poisson blending is skipped or fails

Labels (bounding boxes) are **unchanged** — only background pixels change.
"""

from __future__ import annotations

import shutil
import threading
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}


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
# GrabCut foreground extraction
# ---------------------------------------------------------------------------


def _union_bbox(
    boxes: List[List[float]], w: int, h: int
) -> Tuple[int, int, int, int]:
    """Compute the union pixel bounding box of YOLO-format boxes.

    Args:
        boxes: List of ``[cls, cx, cy, bw, bh]`` in normalised YOLO format.
        w: Image width in pixels.
        h: Image height in pixels.

    Returns:
        ``(x1, y1, x2, y2)`` clamped to image bounds.
    """
    x1_min, y1_min = w, h
    x2_max, y2_max = 0, 0
    for box in boxes:
        cx, cy, bw, bh = box[1], box[2], box[3], box[4]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        x1_min = min(x1_min, x1)
        y1_min = min(y1_min, y1)
        x2_max = max(x2_max, x2)
        y2_max = max(y2_max, y2)
    return (
        max(0, x1_min),
        max(0, y1_min),
        min(w, x2_max),
        min(h, y2_max),
    )


def _grabcut_mask(
    img: np.ndarray,
    rect: Tuple[int, int, int, int],
    iters: int = 5,
    min_fg_ratio: float = 0.05,
) -> np.ndarray:
    """Extract a binary foreground mask via GrabCut.

    Args:
        img: BGR uint8 image.
        rect: ``(x1, y1, x2, y2)`` initialisation rectangle.
        iters: Number of GrabCut iterations.
        min_fg_ratio: If the foreground covers less than this fraction of
            the rect area, GrabCut probably failed — fall back to a filled
            rect mask.

    Returns:
        Binary mask (uint8, 0 or 255) with the same ``(H, W)`` as *img*.
    """
    h, w = img.shape[:2]
    x1, y1, x2, y2 = rect
    rw, rh = x2 - x1, y2 - y1

    # GrabCut needs a rect that is strictly inside the image and has
    # positive width/height.  Below ~8 px per side the GMMs cannot be
    # estimated reliably — fall back to the rect mask directly.
    if rw < 8 or rh < 8:
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        return mask

    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    gc_rect = (x1, y1, rw, rh)
    try:
        cv2.grabCut(img, mask, gc_rect, bgd_model, fgd_model, iters,
                     cv2.GC_INIT_WITH_RECT)
    except cv2.error as exc:
        warnings.warn(f"GrabCut failed ({exc}); using rect-only mask")
        mask[y1:y2, x1:x2] = 255
        return mask

    # Foreground = definite FG + probable FG
    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype(np.uint8)

    # Morphological cleanup: open (remove noise) then close (fill gaps)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Sanity check: if foreground is suspiciously small, GrabCut likely
    # failed — fall back to the full rect.
    rect_area = rw * rh
    fg_area = int(np.count_nonzero(fg_mask))
    if rect_area > 0 and fg_area / rect_area < min_fg_ratio:
        warnings.warn(
            f"GrabCut foreground ratio {fg_area / rect_area:.3f} < "
            f"{min_fg_ratio}; falling back to rect-only mask"
        )
        fg_mask = np.zeros((h, w), dtype=np.uint8)
        fg_mask[y1:y2, x1:x2] = 255

    return fg_mask


# ---------------------------------------------------------------------------
# BackgroundReplacer
# ---------------------------------------------------------------------------


class BackgroundReplacer:
    """Replace image backgrounds using GrabCut segmentation + Poisson blending.

    Breaks background-label correlation by compositing defect foregrounds
    onto diverse background images from a pool.

    Args:
        background_dir: Directory containing background images
            (``.jpg`` / ``.png``).
        fg_padding: Pixels to expand the union bounding box before
            initialising GrabCut.
        grabcut_iters: Number of GrabCut iterations.
        poisson_blend: If ``True``, use ``cv2.seamlessClone`` for compositing
            (falls back to alpha blending on failure).
        alpha: Alpha-blend factor used when Poisson blending is disabled or
            fails.
        bg_resize: Backgrounds are resized so their longest side equals this
            value before per-call resizing to the target image dimensions.
        grabcut_max_side: GrabCut runs on a downscaled copy of the image
            whose longest side is at most this value (mask is upsampled
            afterwards).  Keeps segmentation tractable on 8K frames.
        max_poisson_pixels: Images with more pixels than this skip
            ``seamlessClone`` and use feathered alpha blending directly.
        seed: Seed for the local RNG (``np.random.default_rng``).
    """

    def __init__(
        self,
        background_dir: Path,
        fg_padding: int = 15,
        grabcut_iters: int = 5,
        poisson_blend: bool = True,
        alpha: float = 0.9,
        bg_resize: int = 1024,
        grabcut_max_side: int = 1024,
        max_poisson_pixels: int = 4_000_000,
        seed: int = 42,
    ) -> None:
        self.background_dir = Path(background_dir)
        self.fg_padding = fg_padding
        self.grabcut_iters = grabcut_iters
        self.poisson_blend = poisson_blend
        self.alpha = alpha
        self.bg_resize = bg_resize
        self.grabcut_max_side = grabcut_max_side
        self.max_poisson_pixels = max_poisson_pixels
        self._rng = np.random.default_rng(seed)
        self._rng_lock = threading.Lock()

        # Discover background images (recursive — pools are often organised
        # as images/train + images/val subdirectories).
        if not self.background_dir.is_dir():
            raise FileNotFoundError(
                f"Background directory not found: {self.background_dir}"
            )
        self._bg_paths: List[Path] = sorted(
            p
            for p in self.background_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES
        )
        if not self._bg_paths:
            raise ValueError(
                f"No background images (.jpg/.png) found in "
                f"{self.background_dir}"
            )

    # -- background pool (lazy LRU cache, up to 50) -------------------------

    @lru_cache(maxsize=50)
    def _load_bg(self, index: int) -> np.ndarray:
        """Load and pre-resize a background image (cached)."""
        bg = cv2.imread(str(self._bg_paths[index]))
        if bg is None:
            raise IOError(f"Cannot read background: {self._bg_paths[index]}")
        # Resize longest side to bg_resize
        h, w = bg.shape[:2]
        scale = self.bg_resize / max(h, w)
        if scale < 1.0:
            bg = cv2.resize(
                bg,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        return bg

    def random_background(self) -> np.ndarray:
        """Return a randomly selected background from the pool.

        Thread-safe: the RNG draw is serialised so concurrent callers
        (e.g. a ThreadPoolExecutor batch pipeline) cannot corrupt the
        shared generator state.
        """
        with self._rng_lock:
            idx = int(self._rng.integers(0, len(self._bg_paths)))
        return self._load_bg(idx)

    def _random_bg(self) -> np.ndarray:
        """Return a randomly selected background from the pool."""
        return self.random_background()

    # -- core API -----------------------------------------------------------

    def replace_background(
        self,
        img: np.ndarray,
        boxes: List[List[float]],
        target_bg: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """Replace the background of *img*, keeping defect labels unchanged.

        Args:
            img: ``(H, W, 3)`` uint8 BGR image.
            boxes: YOLO-format labels ``[cls, cx, cy, bw, bh]``.
            target_bg: Optional pre-loaded background image.  If ``None``,
                a random background is drawn from the pool.

        Returns:
            ``(new_img, boxes)`` — the composited image and the **unchanged**
            label list.
        """
        _validate_image(img)

        # No boxes → nothing to segment; return original
        if not boxes:
            return img.copy(), list(boxes)

        h, w = img.shape[:2]

        # 1. Union bounding box + padding
        x1, y1, x2, y2 = _union_bbox(boxes, w, h)
        x1 = max(0, x1 - self.fg_padding)
        y1 = max(0, y1 - self.fg_padding)
        x2 = min(w, x2 + self.fg_padding)
        y2 = min(h, y2 + self.fg_padding)

        # 2–5. GrabCut foreground mask.  Runs at reduced scale on large
        # frames (8K inspection images) — full-resolution GrabCut there
        # takes minutes per image; defect boundaries are smooth enough
        # that a <=1024 px working scale loses nothing.
        long_side = max(h, w)
        gc_scale = min(1.0, self.grabcut_max_side / long_side)
        if gc_scale < 1.0:
            small = cv2.resize(
                img,
                (max(1, int(round(w * gc_scale))),
                 max(1, int(round(h * gc_scale)))),
                interpolation=cv2.INTER_AREA,
            )
            gc_rect = (
                int(round(x1 * gc_scale)),
                int(round(y1 * gc_scale)),
                max(1, int(round(x2 * gc_scale))),
                max(1, int(round(y2 * gc_scale))),
            )
            fg_small = _grabcut_mask(small, gc_rect, iters=self.grabcut_iters)
            fg_mask = cv2.resize(fg_small, (w, h), interpolation=cv2.INTER_LINEAR)
            fg_mask = np.where(fg_mask >= 128, 255, 0).astype(np.uint8)
        else:
            fg_mask = _grabcut_mask(
                img, (x1, y1, x2, y2),
                iters=self.grabcut_iters,
            )

        # 6. Select background
        bg = target_bg if target_bg is not None else self._random_bg()
        _validate_image(bg, "target_bg")

        # 7. Resize background to match image dimensions
        bg_resized = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)

        # 8. Composite via Poisson seamless cloning (skipped on very large
        # images — the Poisson solver scales poorly beyond a few MP).
        result: Optional[np.ndarray] = None
        use_poisson = self.poisson_blend and (h * w <= self.max_poisson_pixels)
        if use_poisson:
            # seamlessClone needs a single-channel mask with 255 for the
            # region to blend and a centre point strictly inside the image.
            # Compute the centroid of the foreground mask.
            ys, xs = np.where(fg_mask > 0)
            if len(xs) > 0:
                cx = int(np.clip(xs.mean(), 1, w - 2))
                cy = int(np.clip(ys.mean(), 1, h - 2))
                try:
                    result = cv2.seamlessClone(
                        img, bg_resized, fg_mask, (cx, cy), cv2.NORMAL_CLONE,
                    )
                except cv2.error as exc:
                    warnings.warn(
                        f"seamlessClone failed ({exc}); "
                        f"falling back to alpha blend"
                    )

        # 9. Feathered alpha-blend fallback
        if result is None:
            # Feather width scales with the image so composite borders stay
            # soft on 8K frames (a fixed 15 px kernel looks like a hard cut).
            feather_k = max(3, min(61, (min(h, w) // 100) | 1))
            alpha_mask = cv2.GaussianBlur(
                fg_mask.astype(np.float32) / 255.0, (feather_k, feather_k), 0
            )[..., None]
            result = (
                img.astype(np.float32) * alpha_mask * self.alpha
                + bg_resized.astype(np.float32) * (1.0 - alpha_mask * self.alpha)
            )
            result = np.clip(result, 0, 255).astype(np.uint8)

        # 10. Labels are unchanged
        return result, list(boxes)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def replace_backgrounds_batch(
    img_dir: Path,
    label_dir: Path,
    output_img_dir: Path,
    output_label_dir: Path,
    background_dir: Path,
    replace_prob: float = 0.5,
    seed: int = 42,
) -> Dict[str, int]:
    """Process a directory of images, replacing backgrounds on a fraction.

    For each image with a corresponding label file, the background is
    replaced with probability ``replace_prob``.  Images that are not
    modified are copied as-is.  Labels are always copied unchanged.

    Args:
        img_dir: Directory of source images.
        label_dir: Directory of YOLO label ``.txt`` files.
        output_img_dir: Destination for output images.
        output_label_dir: Destination for output labels.
        background_dir: Directory of background images for the pool.
        replace_prob: Probability of replacing the background for each
            image that has labels.
        seed: RNG seed.

    Returns:
        Stats dict with ``total``, ``replaced``, ``copied``, ``skipped``.
    """
    img_dir = Path(img_dir)
    label_dir = Path(label_dir)
    output_img_dir = Path(output_img_dir)
    output_label_dir = Path(output_label_dir)

    output_img_dir.mkdir(parents=True, exist_ok=True)
    output_label_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    replacer = BackgroundReplacer(background_dir, seed=seed)

    image_files = sorted(
        f for f in img_dir.iterdir() if f.suffix.lower() in _IMG_SUFFIXES
    )

    stats: Dict[str, int] = {
        "total": len(image_files),
        "replaced": 0,
        "copied": 0,
        "skipped": 0,
    }

    for idx, img_path in enumerate(image_files):
        lbl_path = label_dir / (img_path.stem + ".txt")

        # No label file → copy image, skip augmentation
        if not lbl_path.exists():
            shutil.copy2(img_path, output_img_dir / img_path.name)
            stats["skipped"] += 1
            continue

        # Parse labels
        lines = [
            ln.strip()
            for ln in lbl_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        boxes: List[List[float]] = []
        for ln in lines:
            parts = ln.split()
            if len(parts) >= 5:
                boxes.append([float(p) for p in parts[:5]])

        # Decide whether to replace
        if not boxes or rng.random() > replace_prob:
            shutil.copy2(img_path, output_img_dir / img_path.name)
            shutil.copy2(lbl_path, output_label_dir / lbl_path.name)
            stats["copied"] += 1
            continue

        # Load and augment
        img = cv2.imread(str(img_path))
        if img is None:
            shutil.copy2(img_path, output_img_dir / img_path.name)
            shutil.copy2(lbl_path, output_label_dir / lbl_path.name)
            stats["skipped"] += 1
            continue

        new_img, _ = replacer.replace_background(img, boxes)

        # Save with _bg suffix
        out_name = img_path.stem + "_bg" + img_path.suffix
        cv2.imwrite(str(output_img_dir / out_name), new_img)
        # Labels unchanged
        shutil.copy2(lbl_path, output_label_dir / (img_path.stem + "_bg.txt"))

        # Also copy the original
        shutil.copy2(img_path, output_img_dir / img_path.name)
        shutil.copy2(lbl_path, output_label_dir / lbl_path.name)

        stats["replaced"] += 1

        if (idx + 1) % 200 == 0:
            print(
                f"    [{idx + 1}/{len(image_files)}] "
                f"replaced={stats['replaced']}"
            )

    print(
        f"  Done: {stats['replaced']} backgrounds replaced, "
        f"{stats['copied']} copied, {stats['skipped']} skipped"
    )
    return stats
