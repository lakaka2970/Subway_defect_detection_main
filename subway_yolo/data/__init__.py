# Re-export from ultralytics.data for subway_yolo compatibility
from ultralytics.data import (
    BaseDataset,
    ClassificationDataset,
    GroundingDataset,
    PolygonSemanticDataset,
    SemanticDataset,
    YOLOConcatDataset,
    YOLODataset,
    YOLOMultiModalDataset,
    build_grounding,
    build_yolo_dataset,
    converter,
    load_inference_source,
)

# subway_yolo override: adds multiprocessing_context / persistent_workers for Windows-safe, reusable workers
from subway_yolo.data.build import build_dataloader
