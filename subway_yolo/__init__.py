# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

__version__ = "8.4.33"

import importlib
import os
from typing import TYPE_CHECKING

# Set ENV variables (place before imports)
if not os.environ.get("OMP_NUM_THREADS"):
    os.environ["OMP_NUM_THREADS"] = "1"  # default for reduced CPU utilization during training

from subway_yolo.utils import ASSETS, SETTINGS
from subway_yolo.utils.checks import check_yolo as checks
from subway_yolo.utils.downloads import download

settings = SETTINGS

# Monkey-patch ultralytics BaseModel.predict to accept 'visualize' kwarg
# ultralytics >= 8.2 removed visualize support from BaseModel.predict()
# but subway_yolo's predictor may still pass it through internal code paths
try:
    import ultralytics.nn.tasks as _ult_tasks
    _orig_base_predict = _ult_tasks.BaseModel.predict
    def _patched_predict(self, x, *args, visualize=False, **kwargs):
        return _orig_base_predict(self, x, *args, **kwargs)
    _ult_tasks.BaseModel.predict = _patched_predict
except Exception:
    pass  # silent fallback — patch is non-critical

MODELS = ("YOLO", "RTDETR")

__all__ = (
    "__version__",
    "ASSETS",
    *MODELS,
    "checks",
    "download",
    "settings",
)

if TYPE_CHECKING:
    # Enable hints for type checkers
    from subway_yolo.models import YOLO  # noqa


def __getattr__(name: str):
    """Lazy-import model classes on first access."""
    if name in MODELS:
        return getattr(importlib.import_module("subway_yolo.models"), name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


def __dir__():
    """Extend dir() to include lazily available model names for IDE autocompletion."""
    return sorted(set(globals()) | set(MODELS))


if __name__ == "__main__":
    print(__version__)
