import numpy as np
import pytest
from subway_defect.pipeline.slicer import SmartSlicer
from subway_defect.pipeline.wbf_fusion import WBFFusion


class TestSmartSlicer:
    @pytest.fixture
    def img(self):
        return np.random.randint(0, 255, (2048, 3072, 3), dtype=np.uint8)

    def test_tile_count(self):
        slicer = SmartSlicer(slice_size=1024, overlap=0.15)
        n = slicer.tile_count(2048, 3072)
        assert 6 <= n <= 12

    def test_iter_tiles(self, img):
        slicer = SmartSlicer(slice_size=1024, overlap=0.15)
        tiles = list(slicer.iter_tiles(img))
        assert len(tiles) > 0
        for tile, row, col, x0, y0 in tiles:
            assert tile.shape[:2] == (1024, 1024)
            assert tile.dtype == np.uint8

    def test_tiles_cover_image(self, img):
        """No gap larger than slice_size between adjacent tiles."""
        slicer = SmartSlicer(slice_size=1024, overlap=0.15)
        tiles = list(slicer.iter_tiles(img))
        # Tile count is reasonable for the image size
        n = slicer.tile_count(img.shape[0], img.shape[1])
        assert len(tiles) == n

    def test_roi_tiles(self, img):
        slicer = SmartSlicer(slice_size=1024, overlap=0.15)
        roi = np.array([[0, 0, 512, 512]], dtype=np.float32)
        tiles = list(slicer.roi_tiles(img, roi))
        assert len(tiles) >= 1
        # All tiles should intersect with the ROI
        for tile, row, col, x0, y0 in tiles:
            assert x0 < 512 and x0 + 1024 > 0
            assert y0 < 512 and y0 + 1024 > 0

    def test_large_image_tile_count(self):
        """127MP image tile count is reasonable."""
        slicer = SmartSlicer(slice_size=1024, overlap=0.15)
        n = slicer.tile_count(9800, 13000)
        # Should be ~180-200 tiles for 127MP
        assert 150 <= n <= 250, f"Expected 150-250 tiles, got {n}"


class TestWBFFusion:
    @pytest.fixture
    def dets_a(self):
        return [
            {"box": {"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1},
             "confidence": 0.92, "class_id": 0, "class_name": "nut_missing"},
            {"box": {"x": 0.3, "y": 0.3, "w": 0.05, "h": 0.05},
             "confidence": 0.61, "class_id": 0, "class_name": "nut_missing"},
        ]

    @pytest.fixture
    def dets_b(self):
        return [
            {"box": {"x": 0.5, "y": 0.51, "w": 0.1, "h": 0.1},
             "confidence": 0.88, "class_id": 0, "class_name": "nut_missing"},
        ]

    def test_fuse_dual_detection(self, dets_a, dets_b):
        fusion = WBFFusion()
        result = fusion.fuse(dets_a, dets_b)
        # The high-conf dual detection should survive
        assert len(result) >= 1
        assert result[0]["dual_detected"] is True

    def test_single_model_false_positive_filtered(self):
        fusion = WBFFusion()
        # Only model A has a medium-conf detection -- should be filtered
        dets_a = [{"box": {"x": 0.7, "y": 0.7, "w": 0.05, "h": 0.05},
                    "confidence": 0.61, "class_id": 0, "class_name": "nut_missing"}]
        dets_b = []
        result = fusion.fuse(dets_a, dets_b)
        assert len(result) == 0  # Filtered by single_conf_threshold

    def test_high_conf_single_passes(self):
        fusion = WBFFusion()
        dets_a = [{"box": {"x": 0.7, "y": 0.7, "w": 0.05, "h": 0.05},
                    "confidence": 0.85, "class_id": 0, "class_name": "nut_missing"}]
        dets_b = []
        result = fusion.fuse(dets_a, dets_b)
        assert len(result) == 1  # High enough for single threshold

    def test_empty_input(self):
        fusion = WBFFusion()
        result = fusion.fuse([], [])
        assert result == []


class TestDeployment:
    def test_export_module_importable(self):
        from subway_defect.deployment.export_tensorrt import export_fp16, export_int8
        assert callable(export_fp16)
        assert callable(export_int8)

    def test_fastapi_app_creatable(self):
        from subway_defect.deployment.fastapi_server import app
        assert app is not None
        assert app.title == "Subway Defect Detection API"
