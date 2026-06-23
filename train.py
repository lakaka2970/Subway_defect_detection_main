from ultralytics import YOLO

# Load a model
model = YOLO(r"ultralytics\cfg\models\11\yolo11s-ECA.yaml")  # build a new model from scratch
# model.load = YOLO(r"D:\PycharmProjects\PythonProject1\ultralytics-graduate\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml")  # load a pretrained model 不使用预训练权重，就注释这一行即可
# train
model.train(data=r'',
                cache=True,
                imgsz=640,
                epochs=100,
                batch=32,
                close_mosaic=0,
                workers=0,
                device='0',
                optimizer='SGD',
                amp=True, # close amp
                )


