# Vendored YOLO framework (subway_yolo) - AGPL-3.0 License

from subway_yolo.models.yolo import classify, detect

from .model import YOLO

__all__ = ["YOLO", "classify", "detect"]
