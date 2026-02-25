# HyperMotion Roadmap

## Phase 1: Core Pipeline (current)

**Goal**: A buildable library with a working demo that produces real animation artifacts.

| Milestone | Acceptance Test | Status |
|-----------|----------------|--------|
| Core types and math | Unit tests pass for Euler/Quat/6D round-trips, FK/IK | Done |
| Signal processing | 5-stage pipeline runs on synthetic data, smoothing reduces jitter | Done |
| Motion segmentation | Heuristic classifier produces segments from velocity data | Done |
| BVH export | Output file loads in Blender with correct hierarchy | Done |
| JSON export | Output file passes schema validation (version, frames, joints) | Done |
| Demo mode | `hm_demo` produces demo_clip.json and demo_clip.bvh in <10s | Done |
| Unit tests | 60+ tests pass via `ctest` | Done |
| CI pipeline | GitHub Actions builds on Ubuntu, runs tests and demo | Done |

## Phase 2: ML Integration

**Goal**: Replace synthetic data with real ML inference on video input.

| Milestone | Acceptance Test | Status |
|-----------|----------------|--------|
| YOLOv8 detection | Detect players in 1080p broadcast frame at >15 FPS | Planned |
| HRNet pose estimation | 17-keypoint pose with >0.7 mAP on sports footage | Planned |
| Multi-person tracking | Track 22 players across 10s clip with <5% ID switches | Planned |
| 2D→3D lifting | Depth estimation with geometric fallback | Planned |
| TCN motion classifier | >85% accuracy on 16 motion types (train in Python, infer via ONNX) | Planned |
| Diffusion motion gen | Generate 64-frame motion clips from condition vectors (offline) | Planned |
| Style encoder | Contrastive-trained embeddings that cluster by player | Planned |
| ONNX export pipeline | Python training → ONNX → C++ inference verified end-to-end | Planned |

## Phase 3: Production

**Goal**: Production-quality tools for animation studios and game developers.

| Milestone | Acceptance Test | Status |
|-----------|----------------|--------|
| Studio GUI | Timeline, 3D skeleton viewer, batch processing | Planned |
| Real-time streaming | Process live camera feed at 30 FPS | Planned |
| TensorRT optimization | 2x inference speedup on supported GPUs | Planned |
| Style transfer | Apply player style to generated motion with visible difference | Planned |
| Multi-sport support | Generalize skeleton and motion types beyond football | Planned |
| SDK and API | Documented C++ API with versioned headers | Planned |
