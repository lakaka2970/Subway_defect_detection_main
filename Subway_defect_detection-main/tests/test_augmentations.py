import cv2
import numpy as np
import pytest
from subway_defect.augmentations.scene import (
    motion_blur, sunlitize, tunnelize, weather_augment,
)


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
        from subway_defect.augmentations.contactnet_copy_paste import ContactNetCopyPaste
        assert ContactNetCopyPaste is not None

    def test_init_defaults(self):
        from subway_defect.augmentations.contactnet_copy_paste import ContactNetCopyPaste
        cp = ContactNetCopyPaste(dataset=None, p=0.6, mode="flip")
        assert cp.p == 0.6
        assert cp.mode == "flip"


class TestTrainingConfigs:
    def test_configs_loadable(self):
        from subway_defect.train.configs import (
            ROI_TRAIN_CONFIG, DEFECT_WARMUP_CONFIG,
            DEFECT_FULL_TRAIN_CONFIG, DEFECT_FINETUNE_CONFIG,
        )
        for name, cfg in [
            ("ROI", ROI_TRAIN_CONFIG),
            ("Warmup", DEFECT_WARMUP_CONFIG),
            ("Full", DEFECT_FULL_TRAIN_CONFIG),
            ("Finetune", DEFECT_FINETUNE_CONFIG),
        ]:
            assert "epochs" in cfg, f"{name}: missing epochs"
            assert "imgsz" in cfg, f"{name}: missing imgsz"
            assert "batch" in cfg, f"{name}: missing batch"
            assert "optimizer" in cfg, f"{name}: missing optimizer"
            assert "device" in cfg, f"{name}: missing device"

    def test_scripts_importable(self):
        from subway_defect.train.train_roi import main as roi_main
        from subway_defect.train.train_defect import main as defect_main
        assert callable(roi_main)
        assert callable(defect_main)


class TestPipelineIntegration:
    def test_all_augmentations_on_synthetic_image(self):
        """All scene augmentations handle a structured synthetic image."""
        img = np.zeros((320, 320, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 80), (220, 240), (128, 128, 128), -1)
        cv2.rectangle(img, (130, 100), (190, 130), (200, 200, 200), -1)

        for aug in [tunnelize, sunlitize, motion_blur, weather_augment]:
            result = aug(img.copy())
            assert result.shape == img.shape
            assert result.dtype == np.uint8

    def test_synthetic_import(self):
        """Synthetic generation module is importable."""
        from subway_defect.synthetic.defect_synthesis import generate_missing_defect
        assert callable(generate_missing_defect)
