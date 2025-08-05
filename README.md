# Crystal Detection with YOLOv8 

## Overview
This project uses a fine-tuned YOLOv8 model to detect crystals in real-time from microscopy images. It identifies crystals, calculates their size in microns, and can trigger actions if crystals exceed a certain threshold to prevent damage to the detector.

### Key Highlights:
- Manual and AI-assisted labeling using LabelMe and Segment Anything Model (SAM)
- Fine-tuned YOLOv8 model for accurate crystal detection
- Pixel-to-micron size conversion and threshold alert system
- Inference script that displays detection results with bounding boxes and size info

## Installation 
### 1. Create a conda environment on S3DF
``` bash
ssh yourusername@s3dflogin.slac.stanford.edu
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
```
Add to ```.bashrc ```:
 ``` bash
init_conda () {
    source ~/miniconda3/etc/profile.d/conda.sh
}
init_conda  # ← Run this command whenever using the environment
```
### 2. Set up the environment
``` bash 
conda create --name name_of_your_env python=3.10
conda activate your_env_name
conda install pytorch torchvision torchaudio cudatoolkit=12.2 -c pytorch

pip install ultralytics opencv-python matplotlib
 ```

### Clone or download the Repository:
``` bash
git clone https://github.com/prakritisubedii/coyote_protector.git
cd coyote_protector
```

### Dataset Preparation:
1. Label Images (if retraining the model)
   - Install Labelme locally:
     ``` bash
     pip install labelme
     ```
   - Launch Labelme and start labeling:
     ``` bash
     labelme path/to/images/
     ```  
      - Manually draw bounding boxes or use the AI Prompt or SAM tools in LabelMe
      - Adjust score and IoU thresholds for better results
      - Save the annotations in .json format
    - Transfer your labeled dataset to S3DF:
      ``` bash
      scp -r /path/to/labeled_dataset username@s3dflogin.slac.stanford.edu:/path/to/project
      ```

### Convert JSON to YOLO Format (on S3DF)
Use Ultralytics' CLI:
``` bash
   yolo convert model=labelme location=datasets/labels_json/ output=datasets/labels/ format=yolo
```

Before training, your dataset must be organized into a YOLO-compatible folder structure:
``` bash
├── images/
│   ├── train/        ← training images
│   └── val/          ← validation images
├── labels/
│   ├── train/        ← Corresponding .txt label files
│   └── val/          
```
Simple Manual Split:
- Move ~80% of your images and labels into train/
- Move the remaining ~20% into val/
  
### Training the Model
1. Create ``` dataset.yaml```
``` bash
path: ./datasets

train: images/train
val: images/val

nc: 1
names: ["crystals"]
```
2. Train the Model
```
pip install ultralytics
model = YOLO('yolov8n.pt')

model.train(
    data="path/to/yolo_dataset.yaml",
    epochs=50,
    imgsz=640
)
```
- Trained weights will be saved in :``` runs/detect/train/weight/best.pt ```
Move it to the ```models/``` folder:
``` mv runs/detect/train/weights/best.pt models/ ```


### Running Inference
- Run ```python detect_crystals.py``` to load the trained model and perform inference on your images.
  - This script will display the prediction made by the model with bounding box and crystal sizes in microns.
    

<img width="1201" height="614" alt="Screenshot 2025-07-21 at 3 08 19 PM" src="https://github.com/user-attachments/assets/f030985e-ce8f-454a-8050-8ff9f076d446" />

  

