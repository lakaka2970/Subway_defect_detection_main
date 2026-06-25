# Vendored YOLO framework (subway_yolo) - AGPL-3.0 License

from subway_yolo.models.yolo import classify, detect

# Re-export segment from ultralytics for test compatibility
from ultralytics.models.yolo import segment  # noqa: F401

from .model import YOLO

__all__ = ["YOLO", "classify", "detect", "segment"]
