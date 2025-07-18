# detect_crystals.py

import os
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# === CONFIGURATION ===

MODEL_PATH = "models/best.pt"       # Path to the trained YOLO model
IMAGE_DIR = "images"                # Folder containing images to run inference on
PX_TO_UM = 0.5                      # Conversion factor: 1 pixel = 0.5 micron (placeholder)
NUM_IMAGES = 5                      # Number of images to process

# === LOAD YOLO MODEL ===

def load_model(model_path):
    """
    Loads a YOLO model from the specified path.
    """
    return YOLO(model_path)

# === LOAD IMAGE FILES ===

def load_images_from_directory(directory, max_images):
    """
    Returns a list of image filenames (with .jpg or .png extensions) from a directory.
    """
    return [
        f for f in os.listdir(directory)
        if f.lower().endswith(('.jpg', '.png'))
    ][:max_images]

# === RUN PREDICTION ===

def run_predictions(model, image_paths):
    """
    Runs object detection using the YOLO model on given image paths.
    """
    return model.predict(source=image_paths, save=False)

# === DRAW BOUNDING BOXES & SHOW ===

def annotate_and_display(results, px_to_um):
    """
    For each image, draws bounding boxes and labels with size in pixels and microns.
    Displays the image using matplotlib.
    """
    for result in results:
        img = cv2.imread(result.path)
        h, w = img.shape[:2]

        for i, box in enumerate(result.boxes.xywh.cpu().numpy()):
            x_c, y_c, bw, bh = box  # Center x, center y, box width, box height

            # Convert center-format to corner-format
            x1 = int(x_c - bw / 2)
            y1 = int(y_c - bh / 2)
            x2 = int(x_c + bw / 2)
            y2 = int(y_c + bh / 2)

            # Draw the rectangle (bounding box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Measure longest side and convert to microns
            longest_px = max(bw, bh)
            longest_um = longest_px * px_to_um

            # Label for the box
            label = f"{i+1}: {longest_px:.0f}px / {longest_um:.2f}μm"
            cv2.putText(img, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # Print measurement info
            print(f"{os.path.basename(result.path)} — Crystal {i+1}: {longest_px:.0f}px ({longest_um:.2f} μm)")

        # Display image with boxes
        plt.figure(figsize=(6, 6))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(os.path.basename(result.path))
        plt.axis('off')
        plt.show()

# === MAIN FUNCTION ===

def main():
    model = load_model(MODEL_PATH)

    image_names = load_images_from_directory(IMAGE_DIR, NUM_IMAGES)
    image_paths = [os.path.join(IMAGE_DIR, name) for name in image_names]

    results = run_predictions(model, image_paths)
    annotate_and_display(results, PX_TO_UM)

if __name__ == "__main__":
    main()

