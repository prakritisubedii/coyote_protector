# Crystal Detection with YOLOv8 

## Concept
Crystallography detectors are highly sensitive, and large crystals can cause serious damage if they are not identified in time. To address this, we developed an AI system using YOLOv8 that analyzes microscopy images and automatically detects crystals. The model generates bounding boxes around each crystal, and the dimensions of these boxes (in pixels) are converted to microns using a calibration factor (`px_to_um`) to estimate crystal size. By applying a threshold to these size estimates, the system can flag oversized crystals early and help protect detector systems.

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
Label Images (if retraining the model)
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

Before training, your dataset must be organized into a YOLO-compatible folder structure:
``` bash
├── images/
│   ├── train/        ← training images
│   └── val/          ← validation images
├── labels/
│   ├── train/        ← Corresponding .txt label files
│   └── val/          
```

Preparing Dataset:
- Open ```prepare_dataset.py``` and edit:
``` bash
coco_json_path   = 'path/to/your/dataset_fixed.json'  # COCO JSON annotations
chip_pic_dir     = 'path/to/your/chip_pic'            # Folder with original images
images_train_dir = 'path/to/your/yolo_dataset/images/train'
images_val_dir   = 'path/to/your/yolo_dataset/images/val'
labels_train_dir = 'path/to/your/yolo_dataset/labels/train'
labels_val_dir   = 'path/to/your/yolo_dataset/labels/val'
yaml_path        = 'path/to/your/yolo_dataset/yolo_dataset.yaml'
```
- Run the script:
 ``` python prepare_dataset.py ```

- The script will:
  - Convert COCO annotations to YOLO format
  - Convert images to grayscale (3-channel)
  - Split into train and val sets
  - Create a dataset.yaml file for YOLO training

### Run Training
 ``` bash
python training.py
```

Trained weights will be saved to:
``` runs/detect/train/weights/best.pt ```

On every run the weights will saved as (```train2/```, ```train3/```, etc.)

- You can also select different YOLOv8 models (n/s/m/l/x) and tune parameters like imgsz, epochs, batch, lr0, optimizer, and data augmentations to trade off speed vs accuracy.

### Running Inference

Once you have trained a model and have your `best.pt` weights:

1. Open `run_inference.py` and edit:
   - `weights_path` → path to your trained `.pt`
   - `chip_pic_dir` → folder containing test images
   - `px_to_um` → your pixel-to-micron conversion factor

2. Run:
   ```bash
   python run_inference.py
   ```

### Model Evaluation
After training your YOLOv8 model, you can evaluate its performance using standard object detection metrics:
- Open eval_accuracy.py and edit:
  - weights_path → path to your trained .pt file
  - data_yaml → path to your dataset YAML file
- Run:
``` bash
python eval_accuracy.py
```
The script will calculate and display:
- Precision: Percentage of correct positive predictions
- Recall: Percentage of actual positives correctly identified
- mAP@0.5: Mean Average Precision at IoU threshold of 0.5

### Example

<img width="1201" height="614" alt="Screenshot 2025-07-21 at 3 08 19 PM" src="https://github.com/user-attachments/assets/f030985e-ce8f-454a-8050-8ff9f076d446" />

  
## C++ Inference with ONNX Runtime  

This repository also includes a C++ implementation (`yolov8_cpp/`) that runs YOLOv8 with ONNX Runtime and OpenCV. See the [yolov8_cpp/README.md] for build and usage instructions.  

