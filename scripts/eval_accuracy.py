from ultralytics import YOLO

# Load your TRAINED MODEL (not dataset.yaml)
model = YOLO('runs/detect/train/weights/best.pt')  # path to your trained model

# Use dataset.yaml for validation data
results = model.val(data='yolo_dataset/yolo_dataset.yaml') path/to/your/yolo_dataset.yaml

# Print metrics
print(f"mAP50: {results.box.map50:.3f}")
print(f"Precision: {results.box.mp:.3f}")  
print(f"Recall: {results.box.mr:.3f}")
