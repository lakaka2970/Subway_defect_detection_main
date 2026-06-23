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


class TestTrainingConfigs:
    @staticmethod
    def _import_module(rel_path, mod_name):
        """Import a module by its path relative to the Subway_defect_detection
        package, avoiding the root-level ``train.py`` name collision."""
        import importlib.util
        pkg_root = Path(__file__).parent.parent / "Subway_defect_detection"
        spec = importlib.util.spec_from_file_location(
            mod_name, str(pkg_root / rel_path),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        return mod

    def test_configs_loadable(self):
        configs = self._import_module("train/configs.py", "train_configs")
        for name, cfg in [
            ("ROI", configs.ROI_TRAIN_CONFIG),
            ("Warmup", configs.DEFECT_WARMUP_CONFIG),
            ("Full", configs.DEFECT_FULL_TRAIN_CONFIG),
            ("Finetune", configs.DEFECT_FINETUNE_CONFIG),
        ]:
            assert "epochs" in cfg, f"{name}: missing epochs"
            assert "imgsz" in cfg, f"{name}: missing imgsz"
            assert "batch" in cfg, f"{name}: missing batch"
            assert "optimizer" in cfg, f"{name}: missing optimizer"
            assert "device" in cfg, f"{name}: missing device"

    def test_scripts_importable(self):
        roi_mod = self._import_module("train/train_roi.py", "train_roi")
        defect_mod = self._import_module("train/train_defect.py", "train_defect")
        assert callable(roi_mod.main)
        assert callable(defect_mod.main)
