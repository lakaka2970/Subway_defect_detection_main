import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Subway_defect_detection"))

import cv2
import numpy as np
import pytest
from augmentations.scene import motion_blur, sunlitize, tunnelize, weather_augment


class TestSceneAugmentations:
    @pytest.fixture
    def img(self):
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    def test_tunnelize_shape_dtype(self, img):
        result = tunnelize(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_sunlitize_shape_dtype(self, img):
        result = sunlitize(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_motion_blur_shape_dtype(self, img):
        result = motion_blur(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_weather_shape_dtype(self, img):
        result = weather_augment(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_tunnelize_idempotent_shape(self, img):
        r1 = tunnelize(img)
        r2 = tunnelize(img)
        assert r1.shape == r2.shape

    def test_motion_blur_changes_image(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[56:72, 56:72] = 255
        result = motion_blur(img)
        assert result.sum() > 0

    def test_weather_not_inplace(self, img):
        original = img.copy()
        weather_augment(img)
        assert np.array_equal(img, original), "Input was modified in-place"


class TestContactNetCopyPaste:
    def test_import(self):
        from augmentations.contactnet_copy_paste import ContactNetCopyPaste
        assert ContactNetCopyPaste is not None

    def test_init_defaults(self):
        from augmentations.contactnet_copy_paste import ContactNetCopyPaste
        cp = ContactNetCopyPaste(dataset=None, p=0.6, mode="flip")
        assert cp.p == 0.6
        assert cp.mode == "flip"
