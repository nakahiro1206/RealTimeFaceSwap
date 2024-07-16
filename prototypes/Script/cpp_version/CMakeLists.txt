cmake_minimum_required(VERSION 3.10)
project(FaceDetection)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)
target_include_directories(${OpenCV_INCLUDE_DIRS})

# Find ONNX Runtime
find_package(onnxruntime REQUIRED)
target_include_directories(${ONNXRuntime_INCLUDE_DIRS})

# Add executable
add_executable(FaceDetection main.cpp)

# Link libraries
target_link_libraries(FaceDetection ${OpenCV_LIBS} ${ONNXRuntime_LIBS})
