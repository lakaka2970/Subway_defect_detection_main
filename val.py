from ultralytics import YOLO

# Load a model
model = YOLO(r"")

# Validate
# 验证模型在数据集上的表现
model.val(data=r'',
          imgsz=640,         # input image size
          batch=64,          # batch size per gpu (default 16)
          conf=0.001,        # object confidence threshold for detection (default 0.001 for val)
          iou=0.6,           # intersection over union (IoU) threshold for NMS
          device='0',        # device to run on, i.e. cuda device=0 or device=0,1,2,3 or device=cpu
          workers=0,         # number of worker threads for data loading (per RANK if DDP)
          plots=True,        # save plots during validation
          )