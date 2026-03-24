from ultralytics import YOLO
model = YOLO('yolov8l-seg.pt')
model.train(
    data='/home/jovyan/work/Final Project/yolo_dataset/data.yaml',
    epochs=60,
    imgsz=512,
    batch=64,
    device=0,
    workers=0,
    amp=False
)
