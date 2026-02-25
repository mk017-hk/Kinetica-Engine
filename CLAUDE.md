# HyperMotion — AI Video-to-Animation Pipeline

## Project Overview
- **Purpose**: Convert broadcast sports footage into production-ready skeletal animation
- **Core language**: C++20 for library, inference, CLI tools, and GUI
- **Training language**: Python (PyTorch) for ML model training, exported to ONNX for C++ inference
- **Build**: CMake 3.20+
- **Dependencies**: OpenCV 4.2+, Eigen 3.4+, nlohmann/json 3.11+
- **Optional**: LibTorch (training), ONNX Runtime (inference), CUDA/TensorRT (GPU)

## Development Phases

### Phase 1: Core Pipeline (current)
- All data types, math utilities, and skeleton hierarchy
- Signal processing pipeline (outlier, Savitzky-Golay, Butterworth, quaternion smoothing, foot contact)
- Motion segmentation with heuristic fallback
- BVH and JSON export with schema versioning
- Demo mode (HM_DEMO_MODE) that runs without ML models
- Unit test suite with GoogleTest
- CI pipeline (GitHub Actions)

### Phase 2: ML Integration
- Train pose estimation models (YOLOv8 + HRNet) on sports data
- Train motion classifier (Temporal Conv Net) for 16 motion types
- Train diffusion model for motion generation (offline batch processing)
- ONNX export pipeline from Python to C++ inference
- Style encoder training with contrastive learning

### Phase 3: Production
- Studio GUI with timeline, 3D viewport, and batch processing
- Real-time streaming mode for live capture
- Style transfer and motion blending
- Performance optimization (TensorRT, CUDA)

## Directory Structure
```
hypermotion/
├── CMakeLists.txt
├── include/HyperMotion/
│   ├── core/          # Types, MathUtils, Logger
│   ├── pose/          # Module 1: Multi-Person Pose Estimation
│   ├── skeleton/      # Module 2: Skeleton Mapping
│   ├── signal/        # Module 3: Signal Processing
│   ├── segmenter/     # Module 4: Motion Segmentation
│   ├── ml/            # Module 5: ML Animation Generation
│   ├── style/         # Module 6: Player Style Fingerprinting
│   └── export/        # Module 7: Export Pipeline
├── src/               # Implementations matching include/
├── tests/             # GoogleTest suite
├── tools/             # CLI executables
├── training/          # Python ML training (PyTorch → ONNX)
└── gui/               # Studio GUI (Dear ImGui, optional build)
```

## Namespace Convention
- `hm::` — core types and utilities
- `hm::pose::` — Module 1: Pose Estimation
- `hm::skeleton::` — Module 2: Skeleton Mapping
- `hm::signal::` — Module 3: Signal Processing
- `hm::segmenter::` — Module 4: Motion Segmentation
- `hm::ml::` — Module 5: ML Generation
- `hm::style::` — Module 6: Style Fingerprinting
- `hm::xport::` — Module 7: Export Pipeline

## Core Types (include/HyperMotion/core/Types.h)

### Joint Enum — 22-joint skeleton
```
Hips(0), Spine(1), Spine1(2), Spine2(3), Neck(4), Head(5),
LeftShoulder(6), LeftArm(7), LeftForeArm(8), LeftHand(9),
RightShoulder(10), RightArm(11), RightForeArm(12), RightHand(13),
LeftUpLeg(14), LeftLeg(15), LeftFoot(16), LeftToeBase(17),
RightUpLeg(18), RightLeg(19), RightFoot(20), RightToeBase(21)
```

### Constants
- JOINT_COUNT = 22
- ROTATION_DIM = 6 (6D rotation per joint, Zhou et al.)
- FRAME_DIM = 132 (22 x 6)
- STYLE_DIM = 64
- MOTION_TYPE_COUNT = 16

### Motion Types
Idle, Walk, Jog, Sprint, TurnLeft, TurnRight, Decelerate, Jump,
Slide, Kick, Tackle, Shield, Receive, Celebrate, Goalkeeper, Unknown

## Build Options
| CMake Option | Default | Description |
|-------------|---------|-------------|
| HM_BUILD_TOOLS | ON | Build CLI tools |
| HM_BUILD_TESTS | OFF | Build GoogleTest suite |
| HM_BUILD_GUI | OFF | Build Studio GUI |
| HM_DEMO_MODE | OFF | Enable synthetic demo pipeline |
| HM_ENABLE_CUDA | OFF | Enable CUDA acceleration |
| HM_ENABLE_TENSORRT | OFF | Enable TensorRT backend |
| HM_ENABLE_LIBTORCH | OFF | Enable LibTorch for training |

## Code Style
- C++20 with `std::` containers and algorithms
- Pimpl idiom for classes with heavy dependencies
- `HM_LOG_INFO/WARN/ERROR(tag, message)` for logging
- `hm::` namespace for all public API
- Header + .cpp for every class, no header-only implementations except Types.h
- Use `#ifdef HM_HAS_TORCH` / `#ifdef HM_HAS_ONNXRUNTIME` to guard optional deps

## Schema Versioning
- JSON export includes `"schemaVersion": "1.0.0"` in metadata
- BVH follows standard BVH format with configurable rotation order
- Schema version constant defined in `core/Types.h`
