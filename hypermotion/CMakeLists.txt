cmake_minimum_required(VERSION 3.20)
project(HyperMotion VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(FetchContent)

# Build options
option(HM_BUILD_TOOLS "Build CLI tools" ON)
option(HM_BUILD_GUI "Build GUI application (requires GLFW + Dear ImGui)" OFF)
option(HM_BUILD_TESTS "Build tests" OFF)
option(HM_USE_TENSORRT "Enable TensorRT backend" OFF)
option(HM_USE_CUDA "Enable CUDA support" OFF)

# -------------------------------------------------------------------
# Dependencies — try system packages first, fall back to FetchContent
# -------------------------------------------------------------------

# OpenCV (4.2+)
find_package(OpenCV 4.2 QUIET COMPONENTS core imgproc imgcodecs videoio dnn highgui)
if(NOT OpenCV_FOUND)
    message(STATUS "OpenCV not found on system — fetching from source (minimal modules)...")
    set(BUILD_LIST "core,imgproc,imgcodecs,videoio,dnn,highgui" CACHE STRING "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_python2 OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_java OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_js OFF CACHE BOOL "" FORCE)
    set(WITH_FFMPEG OFF CACHE BOOL "" FORCE)
    set(WITH_GSTREAMER OFF CACHE BOOL "" FORCE)
    set(WITH_GTK OFF CACHE BOOL "" FORCE)
    set(WITH_QT OFF CACHE BOOL "" FORCE)
    set(WITH_V4L OFF CACHE BOOL "" FORCE)
    set(WITH_PROTOBUF ON CACHE BOOL "" FORCE)
    set(BUILD_PROTOBUF ON CACHE BOOL "" FORCE)
    FetchContent_Declare(
        opencv
        GIT_REPOSITORY https://github.com/opencv/opencv.git
        GIT_TAG        4.10.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(opencv)
    set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_videoio opencv_dnn opencv_highgui)
    # When built from source, include dirs may not propagate via targets.
    # Generated headers (opencv_modules.hpp, cvconfig.h) go to CMAKE_BINARY_DIR.
    set(OpenCV_INCLUDE_DIRS
        "${opencv_SOURCE_DIR}/modules/core/include"
        "${opencv_SOURCE_DIR}/modules/imgproc/include"
        "${opencv_SOURCE_DIR}/modules/imgcodecs/include"
        "${opencv_SOURCE_DIR}/modules/videoio/include"
        "${opencv_SOURCE_DIR}/modules/dnn/include"
        "${opencv_SOURCE_DIR}/modules/highgui/include"
        "${opencv_SOURCE_DIR}/modules/calib3d/include"
        "${opencv_SOURCE_DIR}/modules/features2d/include"
        "${opencv_SOURCE_DIR}/modules/flann/include"
        "${CMAKE_BINARY_DIR}"
    )
    set(HM_OPENCV_FROM_SOURCE TRUE)
    message(STATUS "OpenCV fetched and will be built from source")
else()
    message(STATUS "OpenCV ${OpenCV_VERSION} found on system")
endif()

# Eigen (3.4+) — header-only
find_package(Eigen3 3.4 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    find_package(Eigen3 QUIET)
endif()
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found on system — fetching (header-only)...")
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_Populate(eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}"
    )
    message(STATUS "Eigen3 fetched (header-only from ${eigen_SOURCE_DIR})")
else()
    message(STATUS "Eigen3 ${Eigen3_VERSION} found on system")
endif()

# nlohmann/json — header-only
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann/json not found on system — fetching...")
    set(JSON_BuildTests OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)
    message(STATUS "nlohmann/json fetched")
else()
    message(STATUS "nlohmann/json ${nlohmann_json_VERSION} found on system")
endif()

# LibTorch (C++ PyTorch) — required for training (hm_train, hm_style),
# optional for inference-only builds.
find_package(Torch QUIET)
if(Torch_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
    set(HM_HAS_TORCH TRUE)
    message(STATUS "LibTorch found — training modules enabled")
else()
    message(STATUS "LibTorch not found — training tools (hm_train, hm_style) will not have ML support")
    set(HM_HAS_TORCH FALSE)
endif()

# ONNX Runtime — optional
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS "${ONNXRUNTIME_ROOT}/include"
          "/usr/local/include/onnxruntime"
          "/usr/include/onnxruntime")
find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
          "/usr/local/lib"
          "/usr/lib")
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
    message(STATUS "ONNX Runtime found")
else()
    message(STATUS "ONNX Runtime not found — inference features will use stubs")
    set(ONNXRUNTIME_INCLUDE_DIR "")
    set(ONNXRUNTIME_LIB "")
endif()

# CUDA (enable CUDA language only when requested and available)
if(HM_USE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        find_package(CUDAToolkit 12.0 REQUIRED)
        message(STATUS "CUDA enabled — GPU acceleration available")
    else()
        message(WARNING "CUDA requested but no CUDA compiler found — disabling CUDA")
        set(HM_USE_CUDA OFF)
    endif()
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
)

if(HM_OPENCV_FROM_SOURCE)
    target_include_directories(hypermotion PUBLIC ${OpenCV_INCLUDE_DIRS})
endif()

target_link_libraries(hypermotion PUBLIC
    ${OpenCV_LIBS}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
)

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIB)
    target_include_directories(hypermotion PUBLIC ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(hypermotion PUBLIC ${ONNXRUNTIME_LIB})
    target_compile_definitions(hypermotion PUBLIC HM_HAS_ONNXRUNTIME=1)
endif()

# LibTorch — enables training modules (MotionTransformer, ConditionEncoder,
# StyleEncoder, TemporalConvNet, StyleTrainer, ContrastiveLoss, etc.)
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

# -------------------------------------------------------------------
# Tests (GoogleTest)
# -------------------------------------------------------------------

if(HM_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(hm_tests
        tests/test_types.cpp
        tests/test_mathutils.cpp
        tests/test_signal.cpp
        tests/test_export.cpp
        tests/test_noise_scheduler.cpp
        tests/test_skeleton.cpp
    )
    target_link_libraries(hm_tests PRIVATE hypermotion GTest::gtest_main)
    target_include_directories(hm_tests PRIVATE ${HM_INCLUDE_DIR})

    include(GoogleTest)
    gtest_discover_tests(hm_tests)
endif()
