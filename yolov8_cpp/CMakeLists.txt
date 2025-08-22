cmake_minimum_required(VERSION 3.10)
project(yolov8_cpp)

# Force C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# If your OpenCV was built with the old ABI, flip this to 0. Otherwise try 1 first.
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# ONNX Runtime
set(ONNXRUNTIME_DIR ${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.17.0)
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# OpenCV (your custom install)
set(OpenCV_DIR "/sdf/home/p/prakriti/prjlumine22/results/prakriti/yolov8_cpp/opencv_install/lib64/cmake/opencv4")
find_package(OpenCV REQUIRED)

add_executable(yolov8_cpp main.cpp)

target_include_directories(yolov8_cpp PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(yolov8_cpp PRIVATE ${OpenCV_LIBS} onnxruntime)

find_package(Threads REQUIRED)
target_link_libraries(yolov8_cpp PRIVATE ${OpenCV_LIBS} onnxruntime Threads::Threads)
