# Crystal Detection with YOLOv8 

# Overview
This repository contains the code for the crystal detection project, which uses a trained YOLOv8 model to detect crystals in real-time using a webcam or video input. It measures the size of the detected crystals, converts pixels to microns, and triggers an action (e.g., stopping a beam) if a crystal exceeds a defined size threshold.

# Installation 
### Instructions to create environment and install dependencies on S3DF:
1. ssh yourusername@s3dflogin.slac.stanford.edu
2. mkdir -p ~/miniconda3
3. wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
4. bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
 ``` bash
5. init_conda () {
    source ~/miniconda3/etc/profile.d/conda.sh
}
init_conda  # ← To call it later, run 'init_conda'
```
7. conda create --name name_of_your_env python=3.10
8. conda activate your_env_name
9. conda install pytorch torchvision torchaudio cudatoolkit=12.2 -c pytorch

``` bash 
pip install ultralytics opencv-python matplotlib
 ```

### Clone or download the Repository:
 1. git clone https://github.com/prakritisubedii/coyote_protection.git
 2. cd coyote_protection


### Dataset:
- Image: Raw Images
- Annotation Format: Initially json format, then exported to YOLO format for model training
- Models: Contains the weight of the trained model

### Usage:
- Inference: Run ```detect_crystals.py``` to load the trained model and perform inference on images (currently empty).
  - This script will display the prediction made by the model with bounding box and crystal sizes in microns.
    

<img width="1201" height="614" alt="Screenshot 2025-07-21 at 3 08 19 PM" src="https://github.com/user-attachments/assets/f030985e-ce8f-454a-8050-8ff9f076d446" />

  
# Labeling 
Used Labelme with the AI Mask Model (SAM-based) for assisted labeling. 
### Steps:
- Installed label me with ``` pip install label me ```
- Used AI Prompt feature to segment crystals.
- Adjusted score threshold and IOU threshold as needed.
- Saved annotations in ```json format``` and later converted to yolo format for model training.

