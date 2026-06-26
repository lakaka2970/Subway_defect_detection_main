"""
Smart image slicer for ultra-high-resolution catenary imagery.

Handles 127MP (~13000x9800) images by slicing into overlapping
1024x1024 tiles for GPU-efficient inference.
"""

import math
from typing import Iterator, Tuple

import numpy as np


class SmartSlicer:
    """Slice a large image into overlapping tiles for inference.

    Args:
        slice_size (int): Tile edge length in pixels. Default: 1024.
        overlap (float): Overlap ratio between adjacent tiles (0-1).
            Default: 0.15.
        min_roi_overlap (float): Overlap ratio for key regions.
            Default: 0.25.

    Example:
        >>> slicer = SmartSlicer(slice_size=1024, overlap=0.15)
        >>> img = np.zeros((9800, 13000, 3), dtype=np.uint8)
        >>> tiles = list(slicer.iter_tiles(img))
        >>> len(tiles)  # ~180 tiles
    """

    def __init__(self, slice_size: int = 1024, overlap: float = 0.15,
                 min_roi_overlap: float = 0.25):
        self.slice_size = slice_size
        self.overlap = overlap
        self.min_roi_overlap = min_roi_overlap
        self._stride = int(slice_size * (1 - overlap))

    @property
    def stride(self) -> int:
        return self._stride

    def iter_tiles(self, img: np.ndarray
                   ) -> Iterator[Tuple[np.ndarray, int, int, int, int]]:
        """Yield (tile, row, col, x0, y0) for every slice in the image.

        Args:
            img: Input image (H, W, 3) uint8.

        Yields:
            Tuple of (tile_array, row_index, col_index, x0_pixel, y0_pixel).
        """
        h, w = img.shape[:2]
        s = self.slice_size
        stride = self.stride

        n_cols = max(1, math.ceil((w - s) / stride) + 1)
        n_rows = max(1, math.ceil((h - s) / stride) + 1)

        for row in range(n_rows):
            y0 = min(row * stride, h - s)
            y0 = max(0, y0)
            for col in range(n_cols):
                x0 = min(col * stride, w - s)
                x0 = max(0, x0)
                tile = img[y0:y0 + s, x0:x0 + s]
                yield tile, row, col, x0, y0

    def tile_count(self, h: int, w: int) -> int:
        """Return number of tiles for an image of given dimensions."""
        s = self.slice_size
        stride = self.stride
        n_cols = max(1, math.ceil((w - s) / stride) + 1)
        n_rows = max(1, math.ceil((h - s) / stride) + 1)
        return n_rows * n_cols

    def roi_tile_count(self, h: int, w: int, roi_boxes: np.ndarray) -> int:
        """Return number of tiles that intersect with any ROI box."""
        if roi_boxes is None or len(roi_boxes) == 0:
            return self.tile_count(h, w)

        s = self.slice_size
        stride = self.stride
        n_cols = max(1, math.ceil((w - s) / stride) + 1)
        n_rows = max(1, math.ceil((h - s) / stride) + 1)

        count = 0
        for row in range(n_rows):
            y0 = max(0, min(row * stride, h - s))
            y1 = y0 + s
            for col in range(n_cols):
                x0 = max(0, min(col * stride, w - s))
                x1 = x0 + s
                for box in roi_boxes:
                    rx0, ry0, rx1, ry1 = box
                    if x0 < rx1 and x1 > rx0 and y0 < ry1 and y1 > ry0:
                        count += 1
                        break
        return count

    def roi_tiles(self, img: np.ndarray,
                  roi_boxes: np.ndarray) -> Iterator[Tuple]:
        """Yield tiles that intersect with any ROI bounding box.

        Args:
            img: Input image (H, W, 3) uint8.
            roi_boxes: (N, 4) array of [x0, y0, x1, y1] pixel coords.

        Yields:
            Same format as iter_tiles, but only ROI-overlapping tiles.
        """
        if roi_boxes is None or len(roi_boxes) == 0:
            yield from self.iter_tiles(img)
            return

        h, w = img.shape[:2]
        s = self.slice_size
        stride = self.stride
        n_cols = max(1, math.ceil((w - s) / stride) + 1)
        n_rows = max(1, math.ceil((h - s) / stride) + 1)

        for row in range(n_rows):
            y0 = max(0, min(row * stride, h - s))
            y1 = y0 + s
            for col in range(n_cols):
                x0 = max(0, min(col * stride, w - s))
                x1 = x0 + s
                # Check intersection with any ROI
                for box in roi_boxes:
                    rx0, ry0, rx1, ry1 = box
                    if x0 < rx1 and x1 > rx0 and y0 < ry1 and y1 > ry0:
                        tile = img[y0:y1, x0:x1]
                        yield tile, row, col, x0, y0
                        break
