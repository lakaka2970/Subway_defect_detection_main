from ultralytics import YOLO

# Load a model
model = YOLO(r'')

# Predict
model.predict(source=r'', # source directory, image, or video
              save=True,          # save predicted images/videos
              imgsz=640,          # prediction image size
              conf=0.3,          # object confidence threshold (default 0.25)
              iou=0.7,            # intersection over union (IoU) threshold for NMS
              device='0',         # device to run on, i.e. cuda device=0 or device=cpu
              show=False,         # show results if possible (e.g. cv2.imshow)
              save_txt=False,     # save results as .txt file
              save_conf=False,    # save confidences in .txt file
              save_crop=False,    # save cropped prediction boxes
              classes=None,       # filter by class: .predict(classes=[0, 2, 3])
              line_width=None,    # bounding box thickness (pixels)
              visualize=False,    # visualize model features
              augment=False,      # apply image augmentation to prediction sources
              agnostic_nms=False, # class-agnostic NMS
              retina_masks=False, # use high-resolution segmentation masks
              )