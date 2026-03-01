#!/usr/bin/env bash
# ================================================================
#  HyperMotion — Build Library + CLI Tools
#  Run from anywhere; the script locates the hypermotion/ root.
#  Produces: hypermotion/build/ with hm_extract, hm_train, etc.
#
#  Dependencies are fetched automatically via CMake FetchContent
#  when not found on the system.
#
#  Usage:
#    ./build.sh                     # Build library + CLI tools
#    ./build.sh --gui               # Also build GUI (HyperMotion Studio)
#    ./build.sh --tests             # Also build and run tests
#    HM_ENABLE_CUDA=ON ./build.sh    # Enable CUDA
# ================================================================

set -e

echo ""
echo "  ========================================"
echo "   HyperMotion — Build"
echo "  ========================================"
echo ""

# --- Locate the hypermotion project root ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# --- Parse arguments ---
BUILD_GUI=OFF
BUILD_TESTS=OFF
for arg in "$@"; do
    case "$arg" in
        --gui)   BUILD_GUI=ON ;;
        --tests) BUILD_TESTS=ON ;;
        *)       echo "[WARN] Unknown argument: $arg" ;;
    esac
done

# --- Check dependencies ---
command -v cmake >/dev/null 2>&1 || { echo "[ERROR] CMake not found. Install cmake (3.20+)."; exit 1; }
command -v make  >/dev/null 2>&1 || command -v ninja >/dev/null 2>&1 || { echo "[ERROR] make or ninja not found."; exit 1; }
command -v git   >/dev/null 2>&1 || { echo "[ERROR] git not found (needed for FetchContent)."; exit 1; }

BUILD_DIR="${PROJECT_ROOT}/build"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"

GENERATOR="Unix Makefiles"
if command -v ninja >/dev/null 2>&1; then
    GENERATOR="Ninja"
fi

CMAKE_ARGS="-DHM_BUILD_TOOLS=ON -DHM_BUILD_GUI=${BUILD_GUI} -DHM_BUILD_TESTS=${BUILD_TESTS}"
CMAKE_ARGS="${CMAKE_ARGS} -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
CMAKE_ARGS="${CMAKE_ARGS} -DHM_ENABLE_CUDA=${HM_ENABLE_CUDA:-OFF} -DHM_ENABLE_TENSORRT=${HM_ENABLE_TENSORRT:-OFF}"

[ -n "$TORCH_DIR" ]        && CMAKE_ARGS="${CMAKE_ARGS} -DTorch_DIR=${TORCH_DIR}"
[ -n "$ONNXRUNTIME_ROOT" ] && CMAKE_ARGS="${CMAKE_ARGS} -DONNXRUNTIME_ROOT=${ONNXRUNTIME_ROOT}"
[ -n "$OpenCV_DIR" ]       && CMAKE_ARGS="${CMAKE_ARGS} -DOpenCV_DIR=${OpenCV_DIR}"
[ -n "$IMGUI_DIR" ]        && CMAKE_ARGS="${CMAKE_ARGS} -DIMGUI_DIR=${IMGUI_DIR}"

export FETCHCONTENT_BASE_DIR="${PROJECT_ROOT}/build/_deps"

echo "[1/3] Configuring CMake (${GENERATOR})..."
echo "  Source:  ${PROJECT_ROOT}"
echo "  Build:   ${BUILD_DIR}"
echo "  GUI:     ${BUILD_GUI}"
echo "  Tests:   ${BUILD_TESTS}"
echo "  Missing dependencies will be fetched automatically."
echo ""

cmake -G "${GENERATOR}" -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" ${CMAKE_ARGS}

echo ""
echo "[2/3] Building with ${JOBS} threads..."
echo ""

cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" -j "${JOBS}"

echo ""
echo "[3/3] Done!"
echo ""
echo "  Build output: ${BUILD_DIR}"
echo ""

if [ "${BUILD_TESTS}" = "ON" ]; then
    echo "  Running tests..."
    cd "${BUILD_DIR}" && ctest --output-on-failure
fi
