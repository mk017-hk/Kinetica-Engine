#!/usr/bin/env bash
# ================================================================
#  HyperMotion Studio — Linux / macOS Build Script
#  Run from anywhere; the script locates the hypermotion/ root.
#  Produces: hypermotion/build/HyperMotion Studio  (or hm_studio)
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
command -v cmake >/dev/null 2>&1 || { echo "[ERROR] CMake not found."; exit 1; }
command -v make  >/dev/null 2>&1 || command -v ninja >/dev/null 2>&1 || { echo "[ERROR] make or ninja not found."; exit 1; }

BUILD_DIR="${PROJECT_ROOT}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

# Use Ninja if available, otherwise Unix Makefiles
GENERATOR="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
fi

CMAKE_ARGS="-DHM_BUILD_GUI=ON -DHM_BUILD_TOOLS=ON -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"

# CUDA and TensorRT are OFF by default — enable with: HM_USE_CUDA=ON ./build.sh
CMAKE_ARGS="${CMAKE_ARGS} -DHM_USE_CUDA=${HM_USE_CUDA:-OFF} -DHM_USE_TENSORRT=${HM_USE_TENSORRT:-OFF}"

# Optional overrides via environment variables
[ -n "$IMGUI_DIR" ]        && CMAKE_ARGS="${CMAKE_ARGS} -DIMGUI_DIR=${IMGUI_DIR}"
[ -n "$TORCH_DIR" ]        && CMAKE_ARGS="${CMAKE_ARGS} -DTorch_DIR=${TORCH_DIR}"
[ -n "$ONNXRUNTIME_ROOT" ] && CMAKE_ARGS="${CMAKE_ARGS} -DONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}"
[ -n "$OpenCV_DIR" ]       && CMAKE_ARGS="${CMAKE_ARGS} -DOpenCV_DIR=${OpenCV_DIR}"

echo "[1/3] Configuring CMake (${GENERATOR})..."
echo "  Source:  ${PROJECT_ROOT}"
echo "  Build:   ${BUILD_DIR}"
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
