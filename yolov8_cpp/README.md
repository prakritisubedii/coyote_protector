# YOLOv8 C++ Implementation with ONNX Runtime

This project implements YOLOv8 object detection using C++ with ONNX Runtime and OpenCV.

## Project Structure
``` bash
yolov8_cpp/
├── grayscale_images/         ← grayscale input images 
├── grayscale_labels/         ← labels for each image  (optional, for metrics)
├── CMakeLists.txt            ← CMake configuration file for building the project
├── best.onnx                 ← The YOLOv8 model in ONNX format
├── main.cpp                  ← Main C++ source code for the project
└── requirements.txt          ← List of required dependencies
```

### Labels Format 
- One ```.txt``` file per image, same stem (e.g. ```img001.png``` → ```img001.txt```)
- YOLO format:
``` bash
class_id cx cy w h
```
### Model Export 
Export your YOLO PyTorch weights to ONNX with Ultralytics:
``` bash
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

# Export to ONNX
model.export(
    format='onnx',
    imgsz=640,          # Input image size
    optimize=True,      # Optimize for inference
    half=False,         # Use FP32 (set True for FP16 if supported)
    dynamic=False,      # Static input shapes (faster)
    simplify=True       # Simplify the model
)
```
By default, the exported model will be saved under:
``` bash
runs/export/weights/best.onnx
```
Move or copy it into your C++ project folder:
``` bash
cp runs/export/weights/best.onnx /path/to/yolov8_cpp/best.onnx
```

### Prerequisites
Make sure your system has:
- g++ ≥ 7.0 (C++17)
- cmake ≥ 3.10
  
## Installation

### 1. Install ONNX Runtime
``` bash 
cd /path/to/yolov8_cpp 
wget https://github.com/microsoft/onnxruntime/releases/download/v1.17.0/onnxruntime-linux-x64-1.17.0.tgz
tar xzf onnxruntime-linux-x64-1.17.0.tgz
```
  
### 2. Build OpenCV from Source
``` bash
# Create build + install dirs
mkdir -p /path/to/yolov8_cpp/opencv_build
mkdir -p /path/to/yolov8_cpp/opencv_install

# Download OpenCV 4.10.0
mkdir -p /path/to/yolov8_cpp/opencv_source
cd /path/to/yolov8_cpp/opencv_source
wget https://github.com/opencv/opencv/archive/refs/tags/4.10.0.tar.gz -O opencv-4.10.0.tar.gz
tar xzf opencv-4.10.0.tar.gz

# Build + install
cd /path/to/yolov8_cpp/opencv_build
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/path/to/yolov8_cpp/opencv_install \
      /path/to/yolov8_cpp/opencv_source/opencv-4.10.0

make -j8
make install
```
### Project Folder Layout (after setup)
After installing ONNX Runtime and building OpenCV, your project directory should look like this:
``` bash
yolov8_cpp/
├── build/                       ← build output (created after cmake + make)
├── grayscale_images/            ← input images
├── grayscale_labels/            ← labels (optional)
├── onnxruntime-linux-x64-1.17.0/← ONNX Runtime package
├── opencv_build/                ← OpenCV build folder
├── opencv_install/              ← OpenCV install folder
├── opencv_source/               ← OpenCV source folder
├── best.onnx                    ← YOLOv8 model (ONNX)
├── CMakeLists.txt               ← CMake config
├── main.cpp                     ← C++ source code                  
```

### 3. Update ``` CMakeLists.txt ```
Open ```CMakeLists.txt``` and make sure to update these two lines with your paths:

``` bash
# ONNX Runtime (change if version/folder is different)
set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.17.0)

# OpenCV (change to where OpenCVConfig.cmake was installed)
set(OpenCV_DIR "/path/to/yolov8_cpp/opencv_install/lib/cmake/opencv4")
```
Note: on some systems it’s ``` .../lib64/cmake/opencv4``` instead of ```.../lib/cmake/opencv4```.

### 4. Build the Project
From inside your project folder:
``` bash
cd /path/to/yolov8_cpp
mkdir -p build && cd build
rm -r *   # optional, clear old builds
cmake \
  -DOpenCV_DIR=/path/to/yolov8_cpp/opencv_install/lib64/cmake/opencv4 \
  -DONNXRUNTIME_DIR=${PWD}/../onnxruntime-linux-x64-1.17.0 \
  ..

make -j
```
- You only need to run this when setting up the project for the first time, or whenever you change the code or ```CMakeLists.txt```.
- If you only want to re-run detection on new images, skip these steps and go straight to running the binary.

### 5. Run inference
After building, run inference like this:
``` bash
./yolov8_cpp \
  /path/to/yolov8_cpp/best.onnx \
  /path/to/yolov8_cpp/grayscale_images \
  /path/to/yolov8_cpp/grayscale_labels \
  /path/to/yolov8_cpp/output
```
- Argument 1 = ONNX model path
- Argument 2 = input images folder
- Argument 3 = labels folder (optional, used for metrics)
- Argument 4 = output folder (results will be written here)

### How ```main.cpp``` works:

The program does the following:
1. **Preprocessing** – converts each image to RGB, resizes to 640×640 with letterbox, normalizes to `[0,1]`.  
2. **Inference** – runs YOLOv8 ONNX model with ONNX Runtime.  
3. **Postprocessing** – filters predictions by confidence/IoU, rescales boxes back to the original image, applies NMS if needed.  
4. **Saving Results** – always writes:  
   - `summary.csv` → detections per image (boxes, scores, size estimates, inference time,etc.)  
   - `metrics.csv` → precision, recall, F1, average IoU, AP@0.50, average inference time 
5. **Visualization (optional)** – set `SAVE_VIS = true` in `main.cpp` to save `<image>_vis.jpg` with bounding boxes.  
6. **Accuracy (optional)** – if labels are present, metrics are computed against them; if not, detections are still saved.  

All thresholds, pixel-to-micron factor, and flags are defined at the top of `main.cpp`.

## Output Files
- `summary.csv` – per-image detections  
- `metrics.csv` – overall run metrics  
- `<image>_vis.jpg` – bounding box visualizations 

## Examples
<img width="1301" height="484" alt="Screenshot 2025-08-17 at 6 33 47 PM" src="https://github.com/user-attachments/assets/b033b083-db51-48b7-acda-2afd8dd44227" />




<img width="1071" height="395" alt="Screenshot 2025-08-17 at 6 39 42 PM" src="https://github.com/user-attachments/assets/c0208d70-71bc-4ba4-8db5-dd3fccc7509b" />

