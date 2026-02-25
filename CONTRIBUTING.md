# Contributing to HyperMotion

## Build Requirements

- CMake 3.20+
- C++20 compiler: GCC 11+, Clang 14+, or MSVC 2022+
- Git

Dependencies (auto-fetched by CMake if not installed):
- OpenCV 4.2+
- Eigen 3.4+
- nlohmann/json 3.11+

## Building

```bash
cd hypermotion
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DHM_BUILD_TESTS=ON -DHM_BUILD_TOOLS=ON
make -j$(nproc)
```

## Running Tests

```bash
cd hypermotion/build
ctest --output-on-failure
```

Or run the test binary directly:
```bash
./hm_tests
```

## Running the Demo

```bash
./hm_demo
```

This generates `demo_clip.json` and `demo_clip.bvh` in the current directory.

## Code Style

- **Language**: C++20
- **Naming**: PascalCase for types, camelCase for functions and variables, UPPER_CASE for constants
- **Namespaces**: All code under `hm::` with sub-namespaces per module
- **Headers**: One header per class, with matching .cpp file
- **Includes**: Use `#include "HyperMotion/module/File.h"` for project headers
- **Logging**: Use `HM_LOG_INFO(tag, message)` macros, not `std::cout`
- **Error handling**: Return bool or optional for fallible operations, log errors
- **Optional deps**: Guard with `#ifdef HM_HAS_TORCH` / `#ifdef HM_HAS_ONNXRUNTIME`

## Adding a New Module

1. Create headers in `include/HyperMotion/newmodule/`
2. Create implementations in `src/newmodule/`
3. Add source files to the appropriate CMake source list in `CMakeLists.txt`
4. Add tests in `tests/test_newmodule.cpp`
5. Update `STATUS.md`

## Commit Messages

Use clear, imperative messages:
- `Add signal processing pipeline with 5-stage filtering`
- `Fix quaternion double-cover in slerp`
- `Update BVH exporter to support ZXY rotation order`

## Pull Requests

- One feature or fix per PR
- Must compile with default CMake options
- Must pass all existing tests
- Include a test for new functionality where practical
