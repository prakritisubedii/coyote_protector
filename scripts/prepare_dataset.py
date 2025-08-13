#!/usr/bin/env python3

import json
import os
from sklearn.model_selection import train_test_split
import cv2

# ---------- Paths (edit these) ----------
coco_json_path   = 'path/to/your/dataset_fixed.json'
chip_pic_dir     = 'path/to/your/chip_pic'
images_train_dir = 'path/to/your/yolo_dataset/images/train'
images_val_dir   = 'path/to/your/yolo_dataset/images/val'
labels_train_dir = 'path/to/your/yolo_dataset/labels/train'
labels_val_dir   = 'path/to/your/yolo_dataset/labels/val'
yaml_path        = 'path/to/your/yolo_dataset/yolo_dataset.yaml'

# Make sure dirs exist
for d in [images_train_dir, images_val_dir, labels_train_dir, labels_val_dir]:
    os.makedirs(d, exist_ok=True)

# ---------- Load COCO JSON ----------
with open(coco_json_path) as f:
    coco_data = json.load(f)

# Map image ID -> file name
image_id_to_filename = {
    img['id']: os.path.basename(img['file_name'])
    for img in coco_data['images']
}

# Group annotations by image ID
annotations = {}
for ann in coco_data['annotations']:
    img_id = ann['image_id']
    annotations.setdefault(img_id, []).append(ann)

# Category maps
cat_id_to_idx = {cat['id']: idx for idx, cat in enumerate(coco_data['categories'])}
cat_names = [cat['name'] for cat in coco_data['categories']]

# All image IDs
image_ids = list(image_id_to_filename.keys())

# ---------- Split (val images) ----------
train_ids, val_ids = train_test_split(image_ids, test_size=5, random_state=42) #5 for now
print(f"Training images: {len(train_ids)}")
print(f"Validation images: {len(val_ids)}")

def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    w /= img_w
    h /= img_h
    return x_center, y_center, w, h

def process_image(img_id, split):
    fname = image_id_to_filename[img_id]
    src_path = os.path.join(chip_pic_dir, fname)
    img = cv2.imread(src_path)
    if img is None:
        print(f"⚠️ Could not read image: {src_path}")
        return
    h, w = img.shape[:2]

    # Convert to grayscale, then stack to 3 channels
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_3c = cv2.merge([gray, gray, gray])

    if split == 'train':
        img_out_path = os.path.join(images_train_dir, fname)
        label_out_path = os.path.join(labels_train_dir, fname.rsplit('.',1)[0] + ".txt")
    else:
        img_out_path = os.path.join(images_val_dir, fname)
        label_out_path = os.path.join(labels_val_dir, fname.rsplit('.',1)[0] + ".txt")

    # Save grayscale image
    cv2.imwrite(img_out_path, gray_3c)

    # Write YOLO labels
    lines = []
    for ann in annotations.get(img_id, []):
        class_idx = cat_id_to_idx[ann['category_id']]
        yolo_box = coco_to_yolo_bbox(ann['bbox'], w, h)
        line = f"{class_idx} " + " ".join(f"{x:.6f}" for x in yolo_box)
        lines.append(line)

    with open(label_out_path, 'w') as f:
        f.write('\n'.join(lines))

# Process all
for img_id in train_ids:
    process_image(img_id, 'train')

for img_id in val_ids:
    process_image(img_id, 'val')

print("✅ All images processed and labels saved!")

# ---------- Write YAML ----------
data_yaml = f"""
train: path/to/your/yolo_dataset/images/train
val: path/to/your/yolo_dataset/images/val

nc: {len(cat_names)}
names: {cat_names}
"""

with open(yaml_path, 'w') as f:
    f.write(data_yaml)

print(f"✅ YOLO dataset YAML created at: {yaml_path}")
