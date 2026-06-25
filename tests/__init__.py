# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from subway_yolo.cfg import TASK2DATA, TASK2MODEL, TASKS
from subway_yolo.utils import ASSETS, WEIGHTS_DIR, checks

# Constants used in tests
MODEL = WEIGHTS_DIR / "path with spaces" / "yolo11n.pt"  # test spaces in path
CFG = "yolo11n.yaml"
SOURCE = ASSETS / "bus.jpg"
SOURCES_LIST = [ASSETS / "bus.jpg", ASSETS, ASSETS / "*", ASSETS / "**/*.jpg"]
CUDA_IS_AVAILABLE = checks.cuda_is_available()
CUDA_DEVICE_COUNT = checks.cuda_device_count()
TASK_MODEL_DATA = [(task, WEIGHTS_DIR / TASK2MODEL[task], TASK2DATA[task]) for task in TASKS]
MODELS = frozenset([*list(TASK2MODEL.values()), "yolo11n-grayscale.pt"])

__all__ = (
    "CFG",
    "CUDA_DEVICE_COUNT",
    "CUDA_IS_AVAILABLE",
    "MODEL",
    "SOURCE",
    "SOURCES_LIST",
)
