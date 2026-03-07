# HyperMotion

**AI-powered video-to-animation pipeline for sports motion capture**

HyperMotion converts broadcast football (soccer) footage into production-ready skeletal animation data. It detects players, estimates 2D/3D poses, maps them to a 22-joint game skeleton, applies signal processing, classifies motion types, and exports industry-standard BVH and JSON animation files.

## Architecture

```
Video Input ──► Player Detection (YOLOv8) ──► Pose Estimation (HRNet)
                                                      │
                                               Pose Tracking (Hungarian)
                                                      │
                                               2D→3D Lifting
                                                      │
                                             Skeleton Mapping (22-joint)
                                                      │
                                            Signal Processing (5-stage)
                                                      │
                                           Motion Segmentation (TCN)
                                                      │
                                          ┌────────────┴────────────┐
                                     BVH Export              JSON Export
```

### Modules

| Module | Namespace | Status | Description |
|--------|-----------|--------|-------------|
| Core | `hm::` | Implemented | Types, MathUtils (FK/IK, 6D rotations), Logger |
| Pose Estimation | `hm::pose::` | Implemented | Multi-person detection, pose, tracking, depth lifting |
| Skeleton Mapping | `hm::skeleton::` | Implemented | COCO→22-joint mapping, rotation solver, retargeting |
| Signal Processing | `hm::signal::` | Implemented | Outlier, Savitzky-Golay, Butterworth, quaternion smoothing, foot contact |
| Motion Analysis | `hm::motion::` | Implemented | Foot contact detection, trajectory extraction and prediction |
| Motion Segmentation | `hm::segmenter::` | Implemented | Feature extraction, temporal classification, segment merging |
| Motion Clustering | `hm::dataset::` | Implemented | K-means clustering of clips by motion features |
| ML Generation | `hm::ml::` | In Progress | Diffusion-based motion synthesis (offline, requires trained models) |
| Player Style | `hm::style::` | In Progress | Style fingerprinting and library (requires trained models) |
| Export | `hm::xport::` | Implemented | BVH, JSON, clip utilities (sub-clip, resample, mirror, concatenate) |

### Key Design Decisions

- **C++20** for the core library and all inference paths
- **Python** for ML model training only (PyTorch → ONNX export → C++ inference)
- **ONNX Runtime** for inference, **LibTorch** optional for training
- **Demo mode** (`HM_DEMO_MODE`) for testing without ML models — generates synthetic skeleton data through the full pipeline

## Build

### Requirements

- CMake 3.20+
- C++20 compiler (GCC 11+, Clang 14+, MSVC 2022+)
- OpenCV 4.2+ (auto-fetched if not found)
- Eigen 3.4+ (auto-fetched if not found)
- nlohmann/json 3.11+ (auto-fetched if not found)

Optional:
- LibTorch 2.1+ (for training modules)
- ONNX Runtime 1.16+ (for ML inference)
- CUDA 12.0+ / TensorRT 8.6+ (for GPU acceleration)

### Linux

```bash
cd hypermotion
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DHM_BUILD_TOOLS=ON -DHM_BUILD_TESTS=ON
make -j$(nproc)
```

### Windows

```cmd
cd hypermotion
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DHM_BUILD_TOOLS=ON -DHM_BUILD_TESTS=ON
cmake --build . --config Release
```

## Run

### Demo (no ML models required)

```bash
./build/hm_demo
```

Generates a synthetic 90-frame walking animation and exports `demo_clip.json` and `demo_clip.bvh`. This exercises the full pipeline: skeleton construction, signal processing, motion segmentation, and export.

### Full Pipeline (requires ONNX models)

```bash
./build/hm_extract \
  --detector yolov8l.onnx \
  --pose hrnet_w48.onnx \
  --input match.mp4 \
  --output anims/ \
  --format both
```

### Tests

```bash
cd build && ctest --output-on-failure
```

## Motion Analysis Systems

### Foot Contact Detection (`hm::motion::FootContactDetector`)

Detects per-frame foot plant events using three signals:

1. **Foot velocity** — computed via finite difference of the foot joint's world position. A foot moving below the velocity threshold (default 2 cm/s) is a candidate for contact.
2. **Foot height** — the Y-coordinate of the foot joint relative to the ground plane. Must be below the height threshold (default 5 cm).
3. **Temporal stability** — raw contact detections are filtered through a sliding window (default 3 frames). Contact is confirmed only when all frames in the window agree, eliminating single-frame noise.

Output per frame: `left_foot_contact`, `right_foot_contact`, plus smooth blend values (0..1) for transition animation. Stored in `AnimClip::footContacts`.

This improves animation quality by enabling foot-locking (IK correction to prevent foot sliding) and providing ground truth for locomotion state machines.

### Trajectory Extraction (`hm::motion::TrajectoryExtractor`)

For each frame of an animation clip, computes:

- **Root velocity** — smoothed over a configurable window (default 5 frames) for noise reduction
- **Direction of movement** — derived from the root rotation's forward vector
- **Predicted future trajectory** — at t+0.5s, t+1.0s, and t+1.5s using constant-velocity extrapolation with turn rate

Each trajectory point contains: predicted position, velocity, and facing angle. Stored in `AnimClip::trajectories` (one vector of `TrajectoryPoint` per frame).

This data is essential for motion matching, where the runtime compares the desired future trajectory against the trajectory stored in each animation clip to find the best match.

### Motion Clustering (`hm::dataset::MotionClusterer`)

Automatically discovers animation categories by clustering clips based on motion features:

- **Velocity** (average and peak)
- **Turn rate** (degrees/second)
- **Stride frequency** (estimated from foot vertical oscillation)
- **Joint angles** (knee bend, hip rotation, arm swing averages)
- **Vertical range** (root Y extent, distinguishes jumps from ground motion)

Uses k-means with k-means++ initialization and feature normalisation. Each clip is assigned a cluster label (`cluster_01`, `cluster_02`, ...) stored in `AnimClip::clusterID`.

Clustering enables automatic dataset organisation without manual labelling, and the discovered categories can be used to build motion matching databases, balance training data, or identify rare motion types.

## Demo Output

The `hm_demo` tool produces:
- `demo_clip.json` — Full skeleton data with schema version, metadata, 22-joint rotations, positions, and motion segments
- `demo_clip.bvh` — Standard BVH file importable into Blender, Maya, MotionBuilder, etc.

JSON schema version: `1.0.0`

## Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed milestones.

- **Phase 1** (current): Core pipeline, demo mode, BVH/JSON export, unit tests
- **Phase 2**: ML model training, ONNX inference integration, multi-person tracking
- **Phase 3**: Studio GUI, real-time streaming, style transfer at scale

## Status

See [STATUS.md](STATUS.md) for per-module implementation status.

## Project Structure

```
hypermotion/
├── CMakeLists.txt          # Build configuration
├── include/HyperMotion/    # Public headers
├── src/                    # Implementation
├── tests/                  # GoogleTest suite
├── tools/                  # CLI executables (hm_extract, hm_demo, etc.)
├── training/               # Python ML training scripts (PyTorch)
└── gui/                    # Studio GUI (Dear ImGui, optional)
```

## Licence

MIT — see [LICENCE](LICENCE)
