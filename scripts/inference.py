
"""
YOLOv8 Inference Script
-----------------------
Runs YOLOv8 object detection, draws bounding boxes, measures the longest side in px and μm,
and displays side-by-side original vs annotated images.

EDIT THESE PATHS BEFORE RUNNING:
 - weights_path: Path to your trained YOLOv8 weights (.pt file)
 - chip_pic_dir: Folder containing images to run inference on
 - px_to_um:     Pixel-to-micron conversion factor for your setup
"""

import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ---------- Paths (edit these) ----------
weights_path = 'path/to/your/best.pt'         # trained weights
chip_pic_dir = 'path/to/your/chip_pic'        # folder of images to test
px_to_um = 0.5                                # placeholder pixel->micron

# Load model
model = YOLO(weights_path)

# Pick first 10 images
selected_imgs = [f for f in os.listdir(chip_pic_dir) if f.lower().endswith(('.jpg', '.png'))][:10] #number of images to test

# Run prediction (YOLO will also save its own annotated copies if save=True)
results = model.predict(
    source=[os.path.join(chip_pic_dir, f) for f in selected_imgs],
    save=True
)

# Visualize + print measurements
for r in results:
    img = cv2.imread(r.path)
    if img is None:
        print(f"⚠️ Could not read image: {r.path}")
        continue

    for i, box in enumerate(r.boxes.xywh.cpu().numpy()):
        x_c, y_c, bw, bh = box
        x1 = int(x_c - bw / 2)
        y1 = int(y_c - bh / 2)
        x2 = int(x_c + bw / 2)
        y2 = int(y_c + bh / 2)

        # Draw YOLO bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Longest side as coarse size proxy
        longest_px = max(bw, bh)
        longest_um = longest_px * px_to_um
        label = f"{i+1}: {longest_px:.0f}px / {longest_um:.2f}μm"
        cv2.putText(img, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Side-by-side
    original = cv2.imread(r.path)
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title("Original"); plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("YOLO Annotated"); plt.axis('off')

    plt.suptitle(os.path.basename(r.path))
    plt.tight_layout()
    plt.show()

    # Print lengths to console
    for i, box in enumerate(r.boxes.xywh.cpu().numpy()):
        bw, bh = box[2], box[3]
        longest_px = max(bw, bh)
        longest_um = longest_px * px_to_um
        print(f"{os.path.basename(r.path)} — Crystal {i+1}: {longest_px:.0f}px ({longest_um:.2f} μm)")
