#include "HyperMotion/signal/SavitzkyGolay.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>

namespace hm::signal {

static constexpr const char* TAG = "SavitzkyGolay";

SavitzkyGolay::SavitzkyGolay(const SavitzkyGolayConfig& config)
    : config_(config) {
    // Ensure window size is odd
    if (config_.windowSize % 2 == 0) {
        config_.windowSize++;
    }
}

SavitzkyGolay::~SavitzkyGolay() = default;

std::vector<float> SavitzkyGolay::computeCoefficients(int windowSize, int polyOrder) {
    int halfWin = windowSize / 2;
    int n = windowSize;

    // Build Vandermonde matrix
    Eigen::MatrixXf V(n, polyOrder + 1);
    for (int i = 0; i < n; ++i) {
        float x = static_cast<float>(i - halfWin);
        float power = 1.0f;
        for (int j = 0; j <= polyOrder; ++j) {
            V(i, j) = power;
            power *= x;
        }
    }

    // Solve via least squares: coefficients = (V^T V)^{-1} V^T
    // The smoothing coefficients are the first row of (V^T V)^{-1} V^T
    Eigen::MatrixXf VtV = V.transpose() * V;
    Eigen::MatrixXf VtVinv = VtV.inverse();
    Eigen::MatrixXf C = VtVinv * V.transpose();

    // First row gives the smoothing coefficients
    std::vector<float> coeffs(windowSize);
    for (int i = 0; i < windowSize; ++i) {
        coeffs[i] = C(0, i);
    }

    return coeffs;
}

void SavitzkyGolay::filterChannel(std::vector<float>& signal, int windowSize, int polyOrder) {
    int n = static_cast<int>(signal.size());
    if (n < windowSize) return;

    auto coeffs = computeCoefficients(windowSize, polyOrder);
    int halfWin = windowSize / 2;

    std::vector<float> filtered(n);

    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < windowSize; ++j) {
            // Mirror boundary conditions
            int idx = i + j - halfWin;
            if (idx < 0) idx = -idx;
            if (idx >= n) idx = 2 * (n - 1) - idx;
            idx = std::clamp(idx, 0, n - 1);
            sum += coeffs[j] * signal[idx];
        }
        filtered[i] = sum;
    }

    signal = filtered;
}

void SavitzkyGolay::process(std::vector<SkeletonFrame>& frames) {
    if (frames.size() < static_cast<size_t>(config_.windowSize)) return;

    int numFrames = static_cast<int>(frames.size());

    HM_LOG_DEBUG(TAG, "Processing " + std::to_string(numFrames) + " frames (window=" +
                 std::to_string(config_.windowSize) + ", order=" + std::to_string(config_.polyOrder) + ")");

    // Process each joint's position per-axis
    for (int j = 0; j < JOINT_COUNT; ++j) {
        std::vector<float> xSignal(numFrames), ySignal(numFrames), zSignal(numFrames);

        for (int f = 0; f < numFrames; ++f) {
            xSignal[f] = frames[f].joints[j].worldPosition.x;
            ySignal[f] = frames[f].joints[j].worldPosition.y;
            zSignal[f] = frames[f].joints[j].worldPosition.z;
        }

        filterChannel(xSignal, config_.windowSize, config_.polyOrder);
        filterChannel(ySignal, config_.windowSize, config_.polyOrder);
        filterChannel(zSignal, config_.windowSize, config_.polyOrder);

        for (int f = 0; f < numFrames; ++f) {
            frames[f].joints[j].worldPosition.x = xSignal[f];
            frames[f].joints[j].worldPosition.y = ySignal[f];
            frames[f].joints[j].worldPosition.z = zSignal[f];
        }
    }

    // Process root position
    {
        std::vector<float> rx(numFrames), ry(numFrames), rz(numFrames);
        for (int f = 0; f < numFrames; ++f) {
            rx[f] = frames[f].rootPosition.x;
            ry[f] = frames[f].rootPosition.y;
            rz[f] = frames[f].rootPosition.z;
        }
        filterChannel(rx, config_.windowSize, config_.polyOrder);
        filterChannel(ry, config_.windowSize, config_.polyOrder);
        filterChannel(rz, config_.windowSize, config_.polyOrder);
        for (int f = 0; f < numFrames; ++f) {
            frames[f].rootPosition = {rx[f], ry[f], rz[f]};
        }
    }
}

} // namespace hm::signal
