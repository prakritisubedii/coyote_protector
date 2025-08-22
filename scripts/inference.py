"""
YOLOv8 Inference Script → CSV (sizes) + YOLO-rendered images
------------------------------------------------------------
- Uses YOLO's own annotated images (save=True). No matplotlib.
- Computes longest side per detection in px and μm.
- If size > alert_um (default 100 μm), prints "Stop the beam" and marks it in CSV.

EDIT THESE BEFORE RUNNING:
 - weights_path: Path to your trained YOLOv8 weights (.pt)
 - chip_pic_dir: Folder containing images to run inference on
 - px_to_um:     Pixel-to-micron conversion factor for your setup
"""

import os
import csv
from pathlib import Path
import numpy as np
from ultralytics import YOLO

# ---------- Paths (edit these) ----------
weights_path = 'runs/detect/train/weights/best.pt'         # trained weights
chip_pic_dir = 'chip_pic'        # folder of images to test
px_to_um = 0.5                                # pixel -> micron conversion
alert_um = 100.0                              # threshold for alert

# Where to write the CSV
out_dir = Path("runs/size_metrics")
out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir / "measurements.csv"

# Load model
model = YOLO(weights_path)

# Run prediction directly on the whole directory
results = model.predict(
    source=chip_pic_dir,   # directly pass the folder
    save=True,             # YOLO draws/saves annotated images
    verbose=True
)

# Write CSV
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "image",
        "det_idx",
        "class_id",
        "class_name",
        "confidence",
        "x_center_px",
        "y_center_px",
        "width_px",
        "height_px",
        "longest_px",
        "longest_um",
        "alert"
    ])

    for r in results:
        img_name = os.path.basename(r.path)

        if r.boxes is None or len(r.boxes) == 0:
            continue

        xywh = r.boxes.xywh.cpu().numpy()  # [x_c, y_c, w, h] in px
        confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else np.array([])
        clses = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else np.array([], dtype=int)

        for i, (x_c, y_c, bw, bh) in enumerate(xywh):
            longest_px = float(max(bw, bh))
            longest_um = longest_px * px_to_um

            cls_id = int(clses[i]) if clses.size > i else -1
            cls_name = model.names.get(cls_id, str(cls_id)) if hasattr(model, "names") else str(cls_id)
            conf = float(confs[i]) if confs.size > i else float("nan")

            alert_flag = "STOP" if longest_um > alert_um else ""
            if alert_flag:
                print(f"[STOP] {img_name} — det {i+1}: {longest_um:.2f} μm > {alert_um} μm → Stop the beam")

            writer.writerow([
                img_name,
                i + 1,
                cls_id,
                cls_name,
                f"{conf:.4f}",
                f"{x_c:.2f}",
                f"{y_c:.2f}",
                f"{bw:.2f}",
                f"{bh:.2f}",
                f"{longest_px:.2f}",
                f"{longest_um:.2f}",
                alert_flag
            ])

print(f"\nCSV saved to: {csv_path}")
print("YOLO's annotated images are under runs/detect/predict*/")
