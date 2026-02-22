cmake_minimum_required(VERSION 3.20)
project(HyperMotion VERSION 1.0.0 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build options
option(HM_BUILD_TOOLS "Build CLI tools" ON)
option(HM_BUILD_GUI "Build GUI application (requires GLFW + Dear ImGui)" OFF)
option(HM_BUILD_TESTS "Build tests" OFF)
option(HM_USE_TENSORRT "Enable TensorRT backend" ON)
option(HM_USE_CUDA "Enable CUDA support" ON)

# -------------------------------------------------------------------
# Dependencies
# -------------------------------------------------------------------

# OpenCV (4.8+)
find_package(OpenCV 4.8 REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn highgui)

# Eigen (3.4+)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# LibTorch (C++ PyTorch) — only needed by the pose module (DepthLifter)
find_package(Torch QUIET)
if(Torch_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    set(HM_HAS_TORCH TRUE)
else()
    message(STATUS "LibTorch not found — DepthLifter will use geometric fallback only")
    set(HM_HAS_TORCH FALSE)
endif()

# ONNX Runtime
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
          "/usr/local/include/onnxruntime"
          "/usr/include/onnxruntime")
find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
          "/usr/local/lib"
          "/usr/lib")

# nlohmann/json
find_package(nlohmann_json 3.11 REQUIRED)

# CUDA
if(HM_USE_CUDA)
    find_package(CUDAToolkit 12.0 REQUIRED)
endif()

# TensorRT
if(HM_USE_TENSORRT)
    find_path(TENSORRT_INCLUDE_DIR NvInfer.h
        PATHS "${TENSORRT_ROOT}/include"
              "/usr/local/include"
              "/usr/include/x86_64-linux-gnu")
    find_library(TENSORRT_LIB nvinfer
        PATHS "${TENSORRT_ROOT}/lib"
              "/usr/local/lib"
              "/usr/lib/x86_64-linux-gnu")
    find_library(TENSORRT_ONNXPARSER_LIB nvonnxparser
        PATHS "${TENSORRT_ROOT}/lib"
              "/usr/local/lib"
              "/usr/lib/x86_64-linux-gnu")
endif()

# -------------------------------------------------------------------
# Library: hypermotion
# -------------------------------------------------------------------

set(HM_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

set(HM_CORE_SOURCES
    src/core/MathUtils.cpp
    src/core/Logger.cpp
)

set(HM_POSE_SOURCES
    src/pose/PlayerDetector.cpp
    src/pose/SinglePoseEstimator.cpp
    src/pose/PoseTracker.cpp
    src/pose/DepthLifter.cpp
    src/pose/MultiPersonPoseEstimator.cpp
)

set(HM_SKELETON_SOURCES
    src/skeleton/SkeletonMapper.cpp
    src/skeleton/RotationSolver.cpp
    src/skeleton/SkeletonRetargeter.cpp
)

set(HM_SIGNAL_SOURCES
    src/signal/OutlierFilter.cpp
    src/signal/SavitzkyGolay.cpp
    src/signal/ButterworthFilter.cpp
    src/signal/QuaternionSmoother.cpp
    src/signal/FootContactFilter.cpp
    src/signal/SignalPipeline.cpp
)

set(HM_SEGMENTER_SOURCES
    src/segmenter/MotionFeatureExtractor.cpp
    src/segmenter/TemporalConvNet.cpp
    src/segmenter/MotionSegmenter.cpp
)

set(HM_ML_SOURCES
    src/ml/OnnxInference.cpp
    src/ml/NoiseScheduler.cpp
    src/ml/ConditionEncoder.cpp
    src/ml/MotionTransformer.cpp
    src/ml/MotionDiffusionModel.cpp
    src/ml/MotionGenerator.cpp
)

set(HM_STYLE_SOURCES
    src/style/StyleEncoder.cpp
    src/style/ContrastiveLoss.cpp
    src/style/StyleTrainer.cpp
    src/style/StyleLibrary.cpp
)

set(HM_EXPORT_SOURCES
    src/export/BVHExporter.cpp
    src/export/JSONExporter.cpp
    src/export/AnimClipUtils.cpp
)

set(HM_PIPELINE_SOURCES
    src/Pipeline.cpp
)

add_library(hypermotion STATIC
    ${HM_CORE_SOURCES}
    ${HM_POSE_SOURCES}
    ${HM_SKELETON_SOURCES}
    ${HM_SIGNAL_SOURCES}
    ${HM_SEGMENTER_SOURCES}
    ${HM_ML_SOURCES}
    ${HM_STYLE_SOURCES}
    ${HM_EXPORT_SOURCES}
    ${HM_PIPELINE_SOURCES}
)

target_include_directories(hypermotion PUBLIC
    ${HM_INCLUDE_DIR}
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_link_libraries(hypermotion PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    ${ONNXRUNTIME_LIB}
    nlohmann_json::nlohmann_json
)

# LibTorch is only needed by the pose module (DepthLifter learned network)
if(HM_HAS_TORCH)
    target_link_libraries(hypermotion PUBLIC ${TORCH_LIBRARIES})
    target_compile_definitions(hypermotion PUBLIC HM_HAS_TORCH=1)
endif()

if(HM_USE_CUDA)
    target_link_libraries(hypermotion PUBLIC CUDA::cudart)
    target_compile_definitions(hypermotion PUBLIC HM_USE_CUDA=1)
endif()

if(HM_USE_TENSORRT)
    target_include_directories(hypermotion PUBLIC ${TENSORRT_INCLUDE_DIR})
    target_link_libraries(hypermotion PUBLIC ${TENSORRT_LIB} ${TENSORRT_ONNXPARSER_LIB})
    target_compile_definitions(hypermotion PUBLIC HM_USE_TENSORRT=1)
endif()

target_compile_definitions(hypermotion PUBLIC
    HM_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    HM_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    HM_VERSION_PATCH=${PROJECT_VERSION_PATCH}
)

# -------------------------------------------------------------------
# CLI Tools
# -------------------------------------------------------------------

if(HM_BUILD_TOOLS)
    add_executable(hm_extract tools/hm_extract.cpp)
    target_link_libraries(hm_extract PRIVATE hypermotion)

    add_executable(hm_train tools/hm_train.cpp)
    target_link_libraries(hm_train PRIVATE hypermotion)

    add_executable(hm_style tools/hm_style.cpp)
    target_link_libraries(hm_style PRIVATE hypermotion)

    add_executable(hm_batch tools/hm_batch.cpp)
    target_link_libraries(hm_batch PRIVATE hypermotion)
endif()

# -------------------------------------------------------------------
# GUI Application (Dear ImGui + GLFW + OpenGL)
# Self-contained in gui/ — has its own CMakeLists.txt, resources,
# build scripts, and CPack packaging config.
# -------------------------------------------------------------------

if(HM_BUILD_GUI)
    add_subdirectory(gui)
endif()

# -------------------------------------------------------------------
# Install
# -------------------------------------------------------------------

install(TARGETS hypermotion
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY ${HM_INCLUDE_DIR}/HyperMotion DESTINATION include)

if(HM_BUILD_TOOLS)
    install(TARGETS hm_extract hm_train hm_style hm_batch
        RUNTIME DESTINATION bin)
endif()
