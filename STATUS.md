# HyperMotion Module Status

Last updated: 2026-02-25

## Module Status

| Module | Namespace | Headers | Implementation | Tests | Status |
|--------|-----------|---------|---------------|-------|--------|
| **Core Types** | `hm::` | Types.h | (header-only) | test_types.cpp | Implemented |
| **MathUtils** | `hm::` | MathUtils.h | MathUtils.cpp | test_mathutils.cpp | Implemented |
| **Logger** | `hm::` | Logger.h | Logger.cpp | — | Implemented |
| **M1: Pose Estimation** | `hm::pose::` | 5 headers | 5 .cpp files | — | Implemented (needs ONNX models) |
| **M2: Skeleton Mapping** | `hm::skeleton::` | 3 headers | 3 .cpp files | test_skeleton.cpp | Implemented |
| **M3: Signal Processing** | `hm::signal::` | 6 headers | 6 .cpp files | test_signal.cpp | Implemented |
| **M4: Segmentation** | `hm::segmenter::` | 3 headers | 3 .cpp files | — | Implemented (heuristic fallback) |
| **M5: ML Generation** | `hm::ml::` | 6 headers | 6 .cpp files | test_noise_scheduler.cpp | In Progress (needs trained models) |
| **M6: Style** | `hm::style::` | 4 headers | 4 .cpp files | — | In Progress (needs trained models) |
| **M7: Export** | `hm::xport::` | 3 headers | 3 .cpp files | test_export.cpp | Implemented |
| **Pipeline** | `hm::` | Pipeline.h | Pipeline.cpp | — | Implemented |
| **CLI Tools** | — | — | 4 tools + hm_demo | — | Implemented |
| **Training (Python)** | — | — | 13 .py files | — | In Progress |
| **Studio GUI** | — | 1 header | 2 .cpp files | — | Planned |

## What "Implemented" means
- Header and .cpp files exist with real logic (not stubs)
- Compiles without optional dependencies (ONNX, LibTorch, CUDA)
- Has graceful fallback when optional dependencies are missing
- Used by at least one CLI tool or test

## What "In Progress" means
- Code exists and compiles
- Requires trained ML models or optional dependencies for full functionality
- Has compile-time guards for optional deps (`#ifdef HM_HAS_TORCH`)

## What "Planned" means
- Architecture is designed but implementation is not yet started
- May have placeholder headers
