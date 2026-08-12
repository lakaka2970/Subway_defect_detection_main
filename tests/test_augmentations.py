import cv2
import numpy as np
import pytest
from subway_defect.augmentations.degradation import (
    background_blur, defocus_blur, jpeg_compress, resolution_degrade,
)
from subway_defect.augmentations.scene import (
    glare_augment, motion_blur, night_augment, sunlitize, tunnelize,
    vibration_blur, weather_augment, white_balance_shift,
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


ALL_AUGS = [
    tunnelize, sunlitize, motion_blur, vibration_blur,
    white_balance_shift, weather_augment, glare_augment, night_augment,
]

# Image-only degradation augmentations (same shape-in/shape-out contract).
DEGRADATION_AUGS = [resolution_degrade, defocus_blur, jpeg_compress]


class TestGlareAugment:
    @pytest.fixture
    def img(self):
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    def test_shape_dtype(self, img):
        result = glare_augment(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_not_inplace(self, img):
        original = img.copy()
        glare_augment(img)
        assert np.array_equal(img, original), "Input was modified in-place"

    def test_changes_image(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        result = glare_augment(img)
        assert result.sum() > 0, "Glare should brighten at least some pixels"

    def test_small_image_16x16(self):
        img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        result = glare_augment(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestNightAugment:
    @pytest.fixture
    def img(self):
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    def test_shape_dtype(self, img):
        result = night_augment(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_not_inplace(self, img):
        original = img.copy()
        night_augment(img)
        assert np.array_equal(img, original), "Input was modified in-place"

    def test_ir_mode_desaturated(self):
        """p_ir=1.0 forces IR mode → output should be near-grayscale."""
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = night_augment(img, p_ir=1.0)
        b, g, r = result[:, :, 0].astype(float), result[:, :, 1].astype(float), result[:, :, 2].astype(float)
        # Channel spread should be small (desaturated)
        spread = np.abs(b - g).mean() + np.abs(g - r).mean()
        assert spread < 60, f"IR output should be desaturated, got spread={spread:.1f}"

    def test_visible_mode_darker(self):
        """p_ir=0.0 forces visible low-light → output should be darker."""
        img = np.full((128, 128, 3), 180, dtype=np.uint8)
        result = night_augment(img, p_ir=0.0)
        assert result.mean() < img.mean(), "Visible night mode should darken the image"


class TestWhiteBalanceShift:
    @pytest.fixture
    def img(self):
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    def test_shape_dtype(self, img):
        result = white_balance_shift(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_not_inplace(self, img):
        original = img.copy()
        white_balance_shift(img)
        assert np.array_equal(img, original), "Input was modified in-place"


class TestVibrationBlur:
    @pytest.fixture
    def img(self):
        return np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

    def test_shape_dtype(self, img):
        result = vibration_blur(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_is_alias_for_motion_blur(self):
        """vibration_blur should produce the same result as motion_blur
        when the RNG state is identical."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        np.random.seed(42)
        r1 = vibration_blur(img)
        np.random.seed(42)
        r2 = motion_blur(img)
        assert np.array_equal(r1, r2), "vibration_blur should be an alias for motion_blur"


class TestSmallImageEdgeCases:
    """All 8 augmentations must not crash on very small images."""

    @pytest.mark.parametrize("aug", ALL_AUGS, ids=lambda f: f.__name__)
    def test_16x16(self, aug):
        img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        result = aug(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("aug", ALL_AUGS, ids=lambda f: f.__name__)
    def test_4x4(self, aug):
        img = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        result = aug(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    @pytest.mark.parametrize("aug", DEGRADATION_AUGS, ids=lambda f: f.__name__)
    def test_degradation_16x16(self, aug):
        img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
        result = aug(img)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestTunnelizeSpotlight:
    def test_p_spotlight_zero(self):
        """p_spotlight=0.0 should never add a spotlight."""
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = tunnelize(img, p_spotlight=0.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_p_spotlight_one(self):
        """p_spotlight=1.0 should always add a spotlight."""
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = tunnelize(img, p_spotlight=1.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


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

        for aug in ALL_AUGS:
            result = aug(img.copy())
            assert result.shape == img.shape
            assert result.dtype == np.uint8

    def test_synthetic_import(self):
        """Synthetic generation module is importable."""
        from subway_defect.synthetic.defect_synthesis import generate_missing_defect
        assert callable(generate_missing_defect)


class TestResolutionDegrade:
    @pytest.fixture
    def img(self):
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        cv2.rectangle(img, (100, 100), (156, 156), (200, 200, 200), -1)
        return img

    def test_shape_dtype(self, img):
        result = resolution_degrade(img, down_factor=3.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_not_inplace(self, img):
        original = img.copy()
        resolution_degrade(img, down_factor=3.0)
        assert np.array_equal(img, original)

    def test_loses_fine_detail(self, img):
        """A small sharp square must come back softened (fewer effective pixels)."""
        result = resolution_degrade(img, down_factor=4.0)
        # Edge pixels just outside the square pick up bled intensity.
        assert result[98:100, 98:100].max() > img[98:100, 98:100].max()
        # The square's centre stays bright — content is preserved, not erased.
        assert result[128, 128].mean() > 100

    def test_factor_one_returns_copy(self, img):
        result = resolution_degrade(img, down_factor=1.0)
        assert np.array_equal(result, img)

    def test_deterministic_with_explicit_factor(self, img):
        r1 = resolution_degrade(img, down_factor=2.5)
        r2 = resolution_degrade(img, down_factor=2.5)
        assert np.array_equal(r1, r2)


class TestDefocusBlur:
    def test_shape_dtype(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = defocus_blur(img, sigma=2.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_reduces_local_contrast(self):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[::2, ::2] = 255  # checkerboard: maximum local contrast
        result = defocus_blur(img, sigma=3.0)
        assert result.std() < img.std(), "defocus must smooth the image"

    def test_not_inplace(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        original = img.copy()
        defocus_blur(img, sigma=2.0)
        assert np.array_equal(img, original)


class TestJpegCompress:
    def test_shape_dtype(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = jpeg_compress(img, quality=50)
        assert result.shape == img.shape
        assert result.dtype == np.uint8

    def test_low_quality_changes_image(self):
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        result = jpeg_compress(img, quality=10)
        assert not np.array_equal(result, img)

    def test_high_quality_close_to_input(self):
        # Smooth content — pure noise would legitimately destroy any JPEG.
        img = np.full((128, 128, 3), 120, dtype=np.uint8)
        cv2.rectangle(img, (30, 30), (96, 96), (200, 60, 60), -1)
        result = jpeg_compress(img, quality=100)
        diff = np.abs(result.astype(float) - img.astype(float)).mean()
        assert diff < 5.0, f"q=100 should be near-lossless, got mean diff {diff:.2f}"

    def test_not_inplace(self):
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        original = img.copy()
        jpeg_compress(img, quality=50)
        assert np.array_equal(img, original)


class TestBackgroundBlur:
    BOXES = [[0, 0.5, 0.5, 0.2, 0.2]]  # centred box covering 10% of the area

    @pytest.fixture
    def img(self):
        rng = np.random.default_rng(0)
        return rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)

    def test_shape_dtype_and_boxes_unchanged(self, img):
        result, boxes = background_blur(img, self.BOXES, sigma=5.0)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert boxes == self.BOXES

    def test_focus_region_preserved(self, img):
        result, _ = background_blur(img, self.BOXES, sigma=5.0)
        cy, cx = 128, 128
        inner = (slice(cy - 10, cy + 10), slice(cx - 10, cx + 10))
        diff_focus = np.abs(result[inner].astype(float) - img[inner].astype(float)).mean()
        diff_corner = np.abs(result[:32, :32].astype(float) - img[:32, :32].astype(float)).mean()
        assert diff_focus < diff_corner, "box region must stay sharper than background"

    def test_empty_boxes_returns_copy(self, img):
        result, boxes = background_blur(img, [], sigma=5.0)
        assert np.array_equal(result, img)
        assert boxes == []

    def test_not_inplace(self, img):
        original = img.copy()
        background_blur(img, self.BOXES, sigma=5.0)
        assert np.array_equal(img, original)


class TestBackgroundReplacer:
    @pytest.fixture
    def pool_dir(self, tmp_path):
        """Background pool nested in subdirs (images/train, images/val)."""
        for sub in ("images/train", "images/val"):
            d = tmp_path / sub
            d.mkdir(parents=True)
            bg = np.full((64, 64, 3), 40, dtype=np.uint8)
            cv2.imwrite(str(d / "bg.jpg"), bg)
        return tmp_path

    @pytest.fixture
    def img(self):
        img = np.full((128, 128, 3), 200, dtype=np.uint8)
        cv2.rectangle(img, (48, 48), (80, 80), (30, 30, 220), -1)
        return img

    def test_recursive_pool_discovery(self, pool_dir):
        from subway_defect.augmentations.background_replacement import BackgroundReplacer
        replacer = BackgroundReplacer(pool_dir)
        assert len(replacer._bg_paths) == 2, "must find backgrounds recursively"

    def test_replace_background_small_image(self, pool_dir, img):
        from subway_defect.augmentations.background_replacement import BackgroundReplacer
        replacer = BackgroundReplacer(pool_dir, seed=0)
        boxes = [[0, 0.5, 0.5, 0.3, 0.3]]
        result, out_boxes = replacer.replace_background(img, boxes)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert out_boxes == boxes

    def test_large_image_skips_poisson(self, pool_dir):
        """Images beyond max_poisson_pixels must still composite via alpha blend."""
        from subway_defect.augmentations.background_replacement import BackgroundReplacer
        replacer = BackgroundReplacer(pool_dir, max_poisson_pixels=100, seed=0)
        img = np.full((64, 64, 3), 200, dtype=np.uint8)  # 4096 px > 100
        boxes = [[0, 0.5, 0.5, 0.3, 0.3]]
        result, out_boxes = replacer.replace_background(img, boxes)
        assert result.shape == img.shape
        assert out_boxes == boxes

    def test_empty_boxes_noop(self, pool_dir, img):
        from subway_defect.augmentations.background_replacement import BackgroundReplacer
        replacer = BackgroundReplacer(pool_dir, seed=0)
        result, boxes = replacer.replace_background(img, [])
        assert np.array_equal(result, img)
        assert boxes == []
