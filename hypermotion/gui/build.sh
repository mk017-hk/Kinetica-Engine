#!/usr/bin/env bash
# ================================================================
#  HyperMotion Studio — Linux / macOS Build Script
#  Run from anywhere; the script locates the hypermotion/ root.
#  Produces: hypermotion/build/HyperMotion Studio  (or hm_studio)
#
#  Dependencies are fetched automatically via CMake FetchContent
#  when not found on the system.
# ================================================================

set -e

echo ""
echo "  ========================================"
echo "   HyperMotion Studio — Build"
echo "  ========================================"
echo ""

# --- Locate the hypermotion project root (one level up from gui/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Check dependencies ---
command -v cmake >/dev/null 2>&1 || { echo "[ERROR] CMake not found. Install cmake (3.20+)."; exit 1; }
command -v make  >/dev/null 2>&1 || command -v ninja >/dev/null 2>&1 || { echo "[ERROR] make or ninja not found."; exit 1; }
command -v git   >/dev/null 2>&1 || { echo "[ERROR] git not found (needed for FetchContent)."; exit 1; }

# Check CMake version
CMAKE_VERSION=$(cmake --version | head -1 | grep -oP '\d+\.\d+')
CMAKE_MAJOR=$(echo "$CMAKE_VERSION" | cut -d. -f1)
CMAKE_MINOR=$(echo "$CMAKE_VERSION" | cut -d. -f2)
if [ "$CMAKE_MAJOR" -lt 3 ] || { [ "$CMAKE_MAJOR" -eq 3 ] && [ "$CMAKE_MINOR" -lt 20 ]; }; then
    echo "[ERROR] CMake 3.20+ required, found $(cmake --version | head -1)"
    echo "        Install via: pip3 install cmake --upgrade"
    exit 1
fi

BUILD_DIR="${PROJECT_ROOT}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# Use Ninja if available, otherwise Unix Makefiles
GENERATOR="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
fi

CMAKE_ARGS="-DHM_BUILD_GUI=ON -DHM_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"

# CUDA and TensorRT are OFF by default — enable with: HM_ENABLE_CUDA=ON ./build.sh
CMAKE_ARGS="${CMAKE_ARGS} -DHM_ENABLE_CUDA=${HM_ENABLE_CUDA:-OFF} -DHM_ENABLE_TENSORRT=${HM_ENABLE_TENSORRT:-OFF}"

# Optional overrides via environment variables
[ -n "$IMGUI_DIR" ]        && CMAKE_ARGS="${CMAKE_ARGS} -DIMGUI_DIR=${IMGUI_DIR}"
[ -n "$TORCH_DIR" ]        && CMAKE_ARGS="${CMAKE_ARGS} -DTorch_DIR=${TORCH_DIR}"
[ -n "$ONNXRUNTIME_ROOT" ] && CMAKE_ARGS="${CMAKE_ARGS} -DONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}"
[ -n "$OpenCV_DIR" ]       && CMAKE_ARGS="${CMAKE_ARGS} -DOpenCV_DIR=${OpenCV_DIR}"

# FetchContent downloads go here (persisted across rebuilds)
export FETCHCONTENT_BASE_DIR="${PROJECT_ROOT}/build/_deps"

echo "[1/3] Configuring CMake (${GENERATOR})..."
echo "  Source:  ${PROJECT_ROOT}"
echo "  Build:   ${BUILD_DIR}"
echo "  Missing dependencies will be fetched automatically."
echo ""

cmake -G "${GENERATOR}" -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" ${CMAKE_ARGS}

echo ""
echo "[2/3] Building with ${JOBS} threads..."
echo ""

cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" --target hm_studio -j "${JOBS}"

echo ""
echo "[3/3] Done!"
echo ""

EXE="${BUILD_DIR}/HyperMotion Studio"
if [ ! -f "${EXE}" ]; then
    EXE="${BUILD_DIR}/hm_studio"
fi

echo "  Executable: ${EXE}"
echo ""
echo "  To create a distributable package:"
echo "    cd ${BUILD_DIR} && cpack"
echo ""
echo "  To run:"
echo "    \"${EXE}\""
echo ""
