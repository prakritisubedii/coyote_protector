from ultralytics import YOLO

# Load pre-trained YOLOv8 nano model (lightweight and fast; you can try yolov8s or yolov8m if you'd like)
model = YOLO('yolov8n.pt')

# Start training
model.train(
    data="/path/to/yolo_dataset/yolo_dataset.yaml",
    epochs=50,
    imgsz=640
)
