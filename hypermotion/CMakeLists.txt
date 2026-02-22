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

# LibTorch (C++ PyTorch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

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
    src/ml/ConditionEncoder.cpp
    src/ml/NoiseScheduler.cpp
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
    ${TORCH_LIBRARIES}
    ${ONNXRUNTIME_LIB}
    nlohmann_json::nlohmann_json
)

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
# -------------------------------------------------------------------

if(HM_BUILD_GUI)
    find_package(glfw3 3.3 REQUIRED)
    find_package(OpenGL REQUIRED)

    # Dear ImGui sources — expects IMGUI_DIR to point at the imgui repo
    if(NOT DEFINED IMGUI_DIR)
        set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/imgui")
    endif()

    set(IMGUI_SOURCES
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_demo.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
        ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
    )

    set(HM_GUI_SOURCES
        gui/main.cpp
        gui/HyperMotionApp.cpp
    )

    # Windows: embed icon, version info, and manifest into the .exe
    set(HM_GUI_RESOURCES "")
    if(WIN32)
        set(HM_GUI_RESOURCES gui/resources/hm_studio.rc)
    endif()

    # WIN32 flag: produces a Windows GUI app (no console window on launch)
    # On non-Windows platforms this flag is silently ignored
    add_executable(hm_studio WIN32
        ${HM_GUI_SOURCES}
        ${IMGUI_SOURCES}
        ${HM_GUI_RESOURCES}
    )

    # On Windows with WIN32 subsystem, entry point is WinMain.
    # We redirect back to standard main() via SDL-style approach.
    if(WIN32)
        target_link_options(hm_studio PRIVATE "/ENTRY:mainCRTStartup")
    endif()

    target_include_directories(hm_studio PRIVATE
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
        ${CMAKE_CURRENT_SOURCE_DIR}/gui
        ${CMAKE_CURRENT_SOURCE_DIR}/gui/resources
    )

    target_link_libraries(hm_studio PRIVATE
        hypermotion
        glfw
        OpenGL::GL
    )

    if(WIN32)
        target_link_libraries(hm_studio PRIVATE dwmapi)
    endif()

    target_compile_definitions(hm_studio PRIVATE
        IMGUI_IMPL_OPENGL_LOADER_CUSTOM
    )

    # Set output name and properties
    set_target_properties(hm_studio PROPERTIES
        OUTPUT_NAME "HyperMotion Studio"
        VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )

    # Copy required DLLs next to the .exe on Windows after build
    if(WIN32)
        add_custom_command(TARGET hm_studio POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_RUNTIME_DLLS:hm_studio>
                $<TARGET_FILE_DIR:hm_studio>
            COMMAND_EXPAND_LISTS
        )
    endif()

    install(TARGETS hm_studio RUNTIME DESTINATION bin)
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
# CPack — distributable package (ZIP portable or NSIS installer)
# -------------------------------------------------------------------

set(CPACK_PACKAGE_NAME "HyperMotion-Studio")
set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
set(CPACK_PACKAGE_VENDOR "HyperMotion")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Motion Capture & Animation System for UE5 Football Games")
set(CPACK_PACKAGE_INSTALL_DIRECTORY "HyperMotion Studio")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")

if(WIN32)
    set(CPACK_GENERATOR "ZIP;NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "HyperMotion Studio")
    set(CPACK_NSIS_PACKAGE_NAME "HyperMotion Studio")
    set(CPACK_NSIS_ENABLE_UNINSTALL_BEFORE_INSTALL ON)
    set(CPACK_NSIS_CREATE_ICONS_EXTRA
        "CreateShortCut '$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\HyperMotion Studio.lnk' '$INSTDIR\\\\bin\\\\HyperMotion Studio.exe'")
    set(CPACK_NSIS_DELETE_ICONS_EXTRA
        "Delete '$SMPROGRAMS\\\\$START_MENU\\\\HyperMotion Studio.lnk'")
else()
    set(CPACK_GENERATOR "TGZ")
endif()

include(CPack)
