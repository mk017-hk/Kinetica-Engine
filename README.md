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
                                        Multi-Player Tracking (ReID)
                                                      │
                                         Skeleton Mapping (22-joint)
                                                      │
                                        Signal Processing (5-stage)
                                                      │
                                     Canonical Motion Builder
                                   (limb stabilisation, root extraction)
                                                      │
                                       Motion Segmentation (TCN)
                                                      │
                                ┌───────────┬─────────┴──────────┐
                          Foot Contact   Trajectory         Motion
                          Detection      Extraction      Fingerprinting
                                └───────────┴─────────┬──────────┘
                                                      │
                                          Clip Extraction + QA Filter
                                                      │
                                           Motion Clustering (k-means)
                                                      │
                                  ┌───────────────────┴───────────────────┐
                             BVH Export                            JSON Export
                                  └───────────────────┬───────────────────┘
                                                      │
                                           Animation Database
```

### Modules

| Module | Namespace | Description |
|--------|-----------|-------------|
| Core | `hm::` | Types, MathUtils (FK/IK, 6D rotations), Logger, Config I/O |
| Pose Estimation | `hm::pose::` | Player detection, single-person pose, batch processor, depth lifting |
| Multi-Player Tracking | `hm::tracking::` | Persistent player IDs across frames, ReID-based re-identification |
| Skeleton Mapping | `hm::skeleton::` | COCO→22-joint mapping, rotation solver, retargeting |
| Signal Processing | `hm::signal::` | Outlier, Savitzky-Golay, Butterworth, quaternion smoothing, foot contact |
| Canonical Motion | `hm::motion::` | Limb length stabilisation, root orientation solve, local-space conversion |
| Motion Segmentation | `hm::segmenter::` | Feature extraction, temporal classification, segment merging |
| Motion Analysis | `hm::motion::` | Foot contact detection, trajectory extraction and prediction |
| Motion Fingerprinting | `hm::analysis::` | Feature vectors per clip (velocity, stride, joint stats, duration) |
| Motion Intelligence | `hm::analysis::` | Motion embeddings (128D), similarity search, interpolation |
| Motion Clustering | `hm::dataset::` | K-means clustering of clips by motion features |
| Dataset | `hm::dataset::` | Clip extraction, quality filter, classification, animation database |
| ML Generation | `hm::ml::` | Diffusion-based motion synthesis (offline, requires trained models) |
| Player Style | `hm::style::` | Style fingerprinting and library (requires trained models) |
| Export | `hm::xport::` | BVH, JSON, clip utilities (sub-clip, resample, mirror, concatenate) |
| Streaming | `hm::streaming::` | Async pipeline: concurrent decode, inference, analysis threads |

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

## Analysing a Football Match

The `kinetica_analyse_match` CLI tool processes a full match video and produces a structured animation dataset.

### Quick Start

```bash
# With ML models (full pipeline)
./build/bin/kinetica_analyse_match match.mp4 output/ \
  --detector models/yolov8.onnx \
  --pose models/hrnet.onnx

# Async streaming mode (overlaps decode/inference/analysis)
./build/bin/kinetica_analyse_match match.mp4 output/ --streaming \
  --detector models/yolov8.onnx \
  --pose models/hrnet.onnx
```

### What Happens

1. **Decode** — Video frames are read (or streamed in async mode)
2. **Detect** — YOLOv8 finds player bounding boxes in each frame
3. **Pose** — HRNet estimates 17 COCO keypoints per person, lifted to 3D
4. **Track** — Hungarian algorithm + ReID features assign persistent player IDs (up to 22 players)
5. **Map** — COCO keypoints are mapped to the 22-joint HyperMotion skeleton
6. **Filter** — 5-stage signal processing: outlier removal, Savitzky-Golay smoothing, Butterworth low-pass, quaternion smoothing, foot contact filtering
7. **Canonicalise** — Limb lengths are stabilised, root orientation is solved from hip/spine vectors, joints are converted to local space with root motion separated
8. **Segment** — Motion boundaries are detected from velocity changes, direction changes, foot contact transitions, and pauses. Clips are 0.5-5 seconds.
9. **Fingerprint** — Each clip gets a feature vector: average velocity, turn rate, stride length, joint angle statistics, duration
10. **Cluster** — K-means groups similar clips into motion categories
11. **Export** — BVH + JSON files are written to a structured directory

### Output Structure

```
output/
├── walk/
│   ├── clip_0001.bvh
│   ├── clip_0001.json
│   └── clip_0001.meta.json
├── jog/
│   └── ...
├── sprint/
│   └── ...
└── database_summary.json
```

### CLI Options

```
kinetica_analyse_match <input_video> <output_dir> [options]

Options:
  --config <path>         Pipeline config JSON file
  --detector <path>       YOLOv8 detection model (.onnx)
  --pose <path>           HRNet pose estimation model (.onnx)
  --depth <path>          Depth lifting model (.onnx)
  --segmenter <path>      Motion segmenter TCN model (.onnx)
  --classifier <path>     Motion classifier model (.onnx)
  --fps <value>           Target FPS (default: 30)
  --no-bvh               Skip BVH export
  --no-json              Skip JSON export
  --streaming            Use async streaming pipeline
  --stats <path>          Write timing stats to JSON file
  --quiet                 Suppress progress output
```

## Run

### Demo (no ML models required)

```bash
./build/bin/hm_demo
```

Generates a synthetic 90-frame walking animation and exports `demo_clip.json` and `demo_clip.bvh`. This exercises the full pipeline: skeleton construction, signal processing, motion segmentation, and export.

### Full Pipeline (requires ONNX models)

```bash
./build/bin/hm_extract \
  --detector yolov8l.onnx \
  --pose hrnet_w48.onnx \
  --input match.mp4 \
  --output anims/ \
  --format both
```

### Tests

```bash
cd hypermotion/build && ctest --output-on-failure
```

Test suite covers: core types, math utilities, signal processing, skeleton mapping, canonical motion (round-trip stability), motion segmentation, tracking persistence, clip quality filtering, motion fingerprinting, BVH/JSON export, configuration I/O, and a full synthetic end-to-end integration test that exercises the complete pipeline path from skeleton frames through to database export.

## New Modules

### Pose Estimator (`hm::pose::PoseEstimator`)

Unified stateless pose estimator that wraps detection, 2D pose, and 3D lifting into a single `estimateFrame()` call. Suitable for use inside batch or streaming pipelines where tracking is handled externally.

### Pose Batch Processor (`hm::pose::PoseBatchProcessor`)

Accumulates video frames into GPU-friendly batches (default 8 frames). Detection runs per-frame; person crops are batched for efficient 2D pose estimation. Provides `processVideo()` for end-to-end batch processing or `addFrame()`/`flush()` for streaming integration.

### Canonical Motion Builder (`hm::motion::CanonicalMotionBuilder`)

Converts raw skeleton sequences into a canonical representation:

- **Limb length stabilisation** — measures per-joint limb lengths via median, then applies EMA smoothing to eliminate jitter
- **Root orientation solving** — derives root facing direction from hip-to-spine and left-hip-to-right-hip vectors
- **Local-space conversion** — transforms world-space joint positions into parent-relative local rotations
- **Root motion extraction** — separates root position and rotation from joint animation, enabling root motion curves

This canonical form is the standard motion format consumed by segmentation, fingerprinting, export, and ML training.

### Motion Fingerprinting (`hm::analysis::MotionFingerprint`)

Computes an 18D feature vector for each animation clip:

- **Locomotion**: average velocity, peak velocity, average acceleration
- **Turning**: average turn rate, peak turn rate
- **Stride**: stride length (from foot separation peaks), stride frequency
- **Joint statistics**: knee bend, hip rotation, arm swing, spine flexion, head stability
- **Temporal**: clip duration, frame count
- **Foot contacts**: left/right contact ratios

Includes `findSimilar()` for Euclidean nearest-neighbour search over fingerprint databases.

### Streaming Pipeline (`hm::streaming::StreamingPipeline`)

Three-stage asynchronous pipeline:

```
Decode Thread ──► [Frame Queue] ──► Inference Thread ──► [Pose Queue]
    ──► Analysis Thread ──► [Clip Output]
```

- Thread-safe bounded queues with back-pressure
- Optional frame dropping when queues are full
- Clips delivered via callback as they complete
- Real-time statistics monitoring

### Motion Analysis Systems

#### Foot Contact Detection (`hm::motion::FootContactDetector`)

Detects per-frame foot plant events using three signals:

1. **Foot velocity** — finite difference, threshold 2 cm/s
2. **Foot height** — relative to ground plane, threshold 5 cm
3. **Temporal stability** — sliding window (3 frames) for noise elimination

#### Trajectory Extraction (`hm::motion::TrajectoryExtractor`)

Per-frame root motion trajectory with predicted future positions at t+0.5s, t+1.0s, t+1.5s using constant-velocity extrapolation with turn rate.

#### Motion Clustering (`hm::dataset::MotionClusterer`)

K-means clustering by velocity, turn rate, stride frequency, joint angles, and vertical range. K-means++ initialization with feature normalisation.

#### Motion Embeddings (`hm::analysis::MotionEmbedder`)

128D L2-normalized embeddings via Temporal CNN encoder, with feature-based fallback when no model is available.

## Project Structure

```
hypermotion/
├── CMakeLists.txt          # Build configuration
├── include/HyperMotion/
│   ├── core/               # Types, MathUtils, Logger, Config
│   ├── pose/               # PoseEstimator, PoseBatchProcessor, detection, tracking
│   ├── tracking/           # MultiPlayerTracker, PlayerIDManager
│   ├── skeleton/           # SkeletonMapper, RotationSolver, Retargeter
│   ├── signal/             # 5-stage signal processing pipeline
│   ├── motion/             # CanonicalMotionBuilder, FootContact, Trajectory
│   ├── segmenter/          # MotionSegmenter, FeatureExtractor, TemporalConvNet
│   ├── analysis/           # MotionFingerprint, MotionEmbedder, Search, Interpolator
│   ├── dataset/            # ClipExtractor, AnimationDatabase, MotionClusterer
│   ├── ml/                 # DiffusionModel, MotionGenerator, OnnxInference
│   ├── style/              # StyleEncoder, StyleLibrary
│   ├── export/             # BVHExporter, JSONExporter, AnimClipUtils
│   └── streaming/          # StreamingPipeline (async decode/infer/analyse)
├── src/                    # Implementations matching include/
├── tests/                  # GoogleTest suite
├── tools/                  # CLI executables
│   ├── hm_extract          # Video → animation clips
│   ├── hm_demo             # Synthetic demo (no models needed)
│   ├── hm_batch            # Batch process multiple videos
│   ├── hm_config           # Config management
│   ├── hm_train            # ML model training
│   ├── hm_style            # Style encoder training
│   └── kinetica_analyse_match  # Full match analysis
├── training/               # Python ML training scripts (PyTorch → ONNX)
└── gui/                    # Studio GUI (Dear ImGui, optional)
```

## Why Canonical Motion Is the Source of Truth

Every downstream module — segmentation, fingerprinting, dataset export, ML training, BVH output — consumes **canonical motion clips** rather than raw pose data. This design decision has several benefits:

1. **Consistent skeleton**: All clips share the same 22-joint hierarchy with stabilised limb lengths, regardless of player body proportions in the video
2. **Root motion separation**: Root position and orientation are stored independently from local joint rotations, enabling root motion curves for game engines
3. **Noise isolation**: Limb length jitter, joint confidence dropouts, and tracking noise are filtered before data enters any analysis module
4. **Retargeting-ready**: Local-space joint rotations transfer directly to any target skeleton that shares the same hierarchy
5. **Reproducibility**: The canonical form is deterministic given the same input, making dataset versioning and comparison straightforward

The `CanonicalMotionBuilder` sits at the centre of the pipeline: tracked pose sequences enter, and all subsequent processing (segmentation, quality filtering, fingerprinting, export) operates exclusively on the canonical representation.

## Module Implementation Status

| Module | Status | Notes |
|--------|--------|-------|
| Core Types & MathUtils | **Complete** | 22-joint skeleton, FK/IK, 6D rotations, quaternion ops |
| Signal Processing | **Complete** | 5-stage pipeline: outlier, Savitzky-Golay, Butterworth, quat smoothing, foot contact |
| Skeleton Mapper | **Complete** | COCO→22-joint with confidence propagation and timestamp-based velocity |
| Canonical Motion Builder | **Complete** | Limb stabilisation, root orientation solve, local-space conversion |
| Motion Segmenter | **Complete** | Heuristic fallback when TCN model unavailable; ONNX TCN inference when model loaded |
| Clip Extraction & QA | **Complete** | Quality scoring with confidence, smoothness, velocity, temporal coverage |
| Motion Fingerprinting | **Complete** | 18D feature vectors per clip |
| Animation Database | **Complete** | Structured export with BVH + JSON + metadata per clip |
| BVH / JSON Export | **Complete** | Schema-versioned output, standard BVH format |
| Multi-Player Tracking | **Complete** | Hungarian assignment + ReID, persistent player IDs |
| Motion Clustering | **Complete** | K-means++ with feature normalisation |
| Streaming Pipeline | **Complete** | Async 3-stage pipeline (decode→inference→analysis) |
| Pose Estimation | **Requires models** | ONNX inference stubs clean; needs YOLOv8 + HRNet weights |
| ML Motion Generation | **Requires models** | Diffusion model infrastructure present; needs trained weights |
| Style Fingerprinting | **Requires models** | Style encoder + contrastive loss implemented; needs training |
| Studio GUI | **Scaffolded** | Dear ImGui integration exists but incomplete |

## Current Limitations

- **No trained models shipped**: The repository includes ML infrastructure (ONNX Runtime inference, TCN classifiers, diffusion model) but does not bundle pre-trained weights. Without models, the pipeline uses heuristic fallbacks for detection, pose estimation, motion classification, and style encoding
- **Heuristic pose estimation**: Without YOLO/HRNet ONNX models, the pose pipeline cannot process real video. Demo mode and integration tests generate synthetic data to exercise the full pipeline path
- **Single-camera assumption**: The 2D→3D lifting uses monocular depth estimation. Multi-camera triangulation is not yet supported
- **No real-time GUI**: The Studio GUI (`HM_BUILD_GUI`) is scaffolded but incomplete
- **CPU-only default**: GPU acceleration requires CUDA 12.0+ and TensorRT, which are off by default
- **Optional dependency stubs**: When ONNX Runtime or LibTorch are not available, all ML modules compile with clean stubs that return empty results and log warnings

## Roadmap

- **Phase 1** (complete): Core pipeline, demo mode, BVH/JSON export, unit tests, all foundational algorithms, pipeline integration and hardening
- **Phase 2** (next): Train and integrate ML models (YOLOv8 + HRNet pose, TCN motion classifier, diffusion generator), ONNX export from Python training scripts
- **Phase 3**: Studio GUI with timeline and 3D viewport, real-time streaming, style transfer, TensorRT optimisation

### Immediate Next Steps

1. Train YOLOv8 player detector on football broadcast data and export to ONNX
2. Train HRNet pose estimator fine-tuned on sports poses and export to ONNX
3. Train TCN motion classifier on labelled football motion segments
4. Validate end-to-end pipeline on real match footage with trained models
5. Complete Studio GUI with 3D viewport and timeline editing

## Licence

MIT — see [LICENCE](LICENCE)
