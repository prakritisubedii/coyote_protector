# Crystal Detection with YOLOv8 

# Overview
This project uses a fine-tuned YOLOv8 model to detect crystals in real-time. I started by manually labeling ~50 images using LabelMe, drawing bounding boxes around visible crystal. Later, I used the Segment Anything Model (SAM) available in LabelMe to speed up the process. I fine-tuned the auto-generated boxes by adjusting the score and IoU thresholds to make sure the labels were accurate. After creating a clean dataset, I fine-tuned a pretrained YOLOv8 model on it to improve detection accuracy. During inference, the model detects crystals, calculates their size in pixels, and converts them to microns. If a crystal is larger than a set threshold, the system can trigger a action like stopping a beam to prevent damage to the detector.

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


### Dataset Preparation:
- Data: Load images in the images/ folder
- Labeling: (only necessary if retraining the model) 
   - Install labelme: ``` pip install label me ```
   - Open images in LabelMe and use the AI Prompt feature or SAM (Segment Anything Model) to segment crystals.
   - Adjust score threshold and IOU threshold as needed.
   - Review and manually correct bounding boxes if needed.
   - Save annotations in ```json format``` and later convert to ```yolo format``` for model training.
- Model Setup:
  - Make sure you have the YOLOv8 model installed:
    ``` pip install ultralytics ```
  - Place the trained model weights( ```best.pt``` in the ```models/``` directory. 

### Usage:
- Inference: Run ```detect_crystals.py``` to load the trained model and perform inference on your images.
  - This script will display the prediction made by the model with bounding box and crystal sizes in microns.
    

<img width="1201" height="614" alt="Screenshot 2025-07-21 at 3 08 19 PM" src="https://github.com/user-attachments/assets/f030985e-ce8f-454a-8050-8ff9f076d446" />

  

