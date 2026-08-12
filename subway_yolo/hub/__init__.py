# Re-export from ultralytics.hub for subway_yolo compatibility
# ultralytics >= 8.2 removed the hub module; provide graceful fallback
try:
    from ultralytics.hub import *  # noqa: F403
except ModuleNotFoundError:
    # Stub classes/constants for local training (HUB integration not needed)
    PREFIX = ""
    HUB_WEB_ROOT = "https://hub.ultralytics.com"

    class HUBTrainingSession:
        """Stub for Ultralytics HUB training session — not available."""
        model_file = None
        train_args = None

        @staticmethod
        def create_session(*args, **kwargs):
            return None

        def upload_metrics(self):
            pass

        def upload_model(self, *args, **kwargs):
            pass
