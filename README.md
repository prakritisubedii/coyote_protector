# Crystal Detection with YOLOv8 

# Overview
This project uses a fine-tuned YOLOv8 model to detect crystals in real-time. I started by manually labeling ~50 images using LabelMe, drawing bounding boxes around visible crystal. Later, I used the Segment Anything Model (SAM) available in LabelMe to speed up the process. I fine-tuned the auto-generated boxes by adjusting the score and IoU thresholds to make sure the labels were accurate. After creating a clean dataset, I fine-tuned a pretrained YOLOv8 model on it to improve detection accuracy. During inference, the model detects crystals, calculates their size in pixels, and converts them to microns. If a crystal is larger than a set threshold, the system can trigger a action like stopping a beam to prevent damage to the detector.

# Installation 
## 1. Create a conda environment on S3DF
``` bash
1. ssh yourusername@s3dflogin.slac.stanford.edu
2. mkdir -p ~/miniconda3
3. wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
4. bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
```
Add to ```.bashrc ```:
 ``` bash
5. init_conda () {
    source ~/miniconda3/etc/profile.d/conda.sh
}
init_conda  # ← Run this command whenever using the environment
```
## 2. Set up the environment
``` bash 
7. conda create --name name_of_your_env python=3.10
8. conda activate your_env_name
9. conda install pytorch torchvision torchaudio cudatoolkit=12.2 -c pytorch

pip install ultralytics opencv-python matplotlib
 ```

## Clone or download the Repository:
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
      - Manually Label or Use the AI tools in LabelMe
      - Click AI Prompt or use SAM (Segment Anything Model)
      - Adjust score threshold and IOU threshold for better bounding boxes.
      - Save annotations in ```.json``` format
    - Transfer your labeled dataset to S3DF:
      ``` bash
      scp -r /path/to/labeled_dataset username@s3dflogin.slac.stanford.edu:/path/to/project
      ```

## Convert JSON to YOLO Format (on S3DF)
Use Ultralytics' CLI tool to convert annotations:
``` bash
   yolo convert model=labelme location=datasets/labels_json/ output=datasets/labels/ format=yolo
```
Simple Manual Split:
- Move ~80% of your images and labels into train/
- Move the remaining ~20% into val/

Before training, your dataset must be organized into a YOLO-compatible folder structure:
``` bash
├── images/
│   ├── train/        ← training images
│   └── val/          ← validation images
├── labels/
│   ├── train/        ← YOLO-format .txt files for train images
│   └── val/          ← YOLO-format .txt files for val images
```
## Training the Model
1. Create ``` dataset.yaml```

  - Make sure you have the YOLOv8 model installed:
    ``` pip install ultralytics ```
  - Place the trained model weights( ```best.pt``` in the ```models/``` directory. 

### Usage:
- Inference: Run ```detect_crystals.py``` to load the trained model and perform inference on your images.
  - This script will display the prediction made by the model with bounding box and crystal sizes in microns.
    

<img width="1201" height="614" alt="Screenshot 2025-07-21 at 3 08 19 PM" src="https://github.com/user-attachments/assets/f030985e-ce8f-454a-8050-8ff9f076d446" />

  

