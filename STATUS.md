# HyperMotion Module Status

Last updated: 2026-03-07

## Module Status

| Module | Namespace | Headers | Implementation | Tests | Status |
|--------|-----------|---------|---------------|-------|--------|
| **Core Types** | `hm::` | Types.h | (header-only) | test_types.cpp | Complete |
| **MathUtils** | `hm::` | MathUtils.h | MathUtils.cpp | test_mathutils.cpp | Complete |
| **Logger** | `hm::` | Logger.h | Logger.cpp | — | Complete |
| **Config I/O** | `hm::` | PipelineConfigIO.h | PipelineConfigIO.cpp | test_config.cpp | Complete |
| **Pipeline** | `hm::` | Pipeline.h | Pipeline.cpp | test_integration.cpp | Complete |
| **M1: Pose Estimation** | `hm::pose::` | 8 headers | 8 .cpp files | — | Requires ONNX models |
| **M2: Skeleton Mapping** | `hm::skeleton::` | 3 headers | 3 .cpp files | test_skeleton.cpp | Complete |
| **M3: Signal Processing** | `hm::signal::` | 6 headers | 6 .cpp files | test_signal.cpp | Complete |
| **M4: Segmentation** | `hm::segmenter::` | 3 headers | 3 .cpp files | test_segmentation.cpp | Complete (heuristic fallback) |
| **M5: ML Generation** | `hm::ml::` | 6 headers | 6 .cpp files | test_noise_scheduler.cpp | Requires trained models |
| **M6: Style** | `hm::style::` | 4 headers | 4 .cpp files | — | Requires trained models |
| **M7: Export** | `hm::xport::` | 3 headers | 3 .cpp files | test_export.cpp | Complete |
| **Tracking** | `hm::tracking::` | 2 headers | 2 .cpp files | test_tracking.cpp, test_hardening.cpp | Complete |
| **Canonical Motion** | `hm::motion::` | 3 headers | 3 .cpp files | test_canonical_motion.cpp, test_hardening.cpp | Complete |
| **Analysis** | `hm::analysis::` | 4 headers | 4 .cpp files | test_fingerprint.cpp, test_hardening.cpp | Complete |
| **Dataset** | `hm::dataset::` | 6 headers | 6 .cpp files | test_clip_quality.cpp, test_integration.cpp, test_hardening.cpp | Complete |
| **Streaming** | `hm::streaming::` | 1 header | 1 .cpp file | — | Complete (limited pipeline) |
| **CLI Tools** | — | — | 7 tools + hm_demo | — | Complete |
| **Training (Python)** | — | — | 13 .py files | — | In Progress |
| **Studio GUI** | — | 1 header | 2 .cpp files | — | Scaffolded |

## What "Complete" means
- Header and .cpp files exist with real logic (not stubs)
- Compiles without optional dependencies (ONNX, LibTorch, CUDA)
- Has graceful fallback when optional dependencies are missing
- Used by at least one CLI tool or test
- Tested via unit tests or integration tests

## What "Requires trained models" means
- Code exists, compiles, and has clean stubs
- Full functionality requires ONNX/LibTorch models that are not shipped
- Has compile-time guards for optional deps (`#ifdef HM_HAS_TORCH`, `#ifdef HM_HAS_ONNXRUNTIME`)

## What "Scaffolded" means
- Architecture is designed and initial implementation exists
- Not feature-complete or production-ready

## Streaming Pipeline Limitations
The streaming pipeline (`--streaming`) runs decode, inference, and analysis concurrently
but produces per-player clips without the full dataset pipeline. It does NOT currently run:
- Clip extraction from segments
- Clip quality filtering
- Motion classification
- Motion clustering
- Database export with structured folders

Use standard (non-streaming) mode via `kinetica_analyse_match` for the complete pipeline.
