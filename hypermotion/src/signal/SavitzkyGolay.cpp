#include "HyperMotion/signal/SavitzkyGolay.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <numeric>

namespace hm::signal {

static constexpr const char* TAG = "SavitzkyGolay";

SavitzkyGolay::SavitzkyGolay(const SavitzkyGolayConfig& config)
    : config_(config) {
    // Validate and fix window size: must be odd and >= 3
    if (config_.windowSize < 3) {
        HM_LOG_WARN(TAG, "Window size " + std::to_string(config_.windowSize) +
                    " too small, clamping to 3");
        config_.windowSize = 3;
    }
    if (config_.windowSize % 2 == 0) {
        config_.windowSize++;
        HM_LOG_WARN(TAG, "Window size must be odd, adjusted to " +
                    std::to_string(config_.windowSize));
    }

    // Validate polynomial order: must be < window size
    if (config_.polyOrder < 0) {
        HM_LOG_WARN(TAG, "Polynomial order must be non-negative, clamping to 0");
        config_.polyOrder = 0;
    }
    if (config_.polyOrder >= config_.windowSize) {
        config_.polyOrder = config_.windowSize - 1;
        HM_LOG_WARN(TAG, "Polynomial order must be less than window size, adjusted to " +
                    std::to_string(config_.polyOrder));
    }

    HM_LOG_DEBUG(TAG, "Initialized with window=" + std::to_string(config_.windowSize) +
                 " order=" + std::to_string(config_.polyOrder));
}

SavitzkyGolay::~SavitzkyGolay() = default;

std::vector<float> SavitzkyGolay::computeCoefficients(int windowSize, int polyOrder) {
    // Validate inputs
    if (windowSize < 3 || windowSize % 2 == 0) {
        HM_LOG_ERROR(TAG, "Invalid window size for coefficient computation: " +
                     std::to_string(windowSize));
        // Return identity filter (pass-through at centre)
        std::vector<float> identity(windowSize, 0.0f);
        if (windowSize > 0) identity[windowSize / 2] = 1.0f;
        return identity;
    }
    if (polyOrder < 0 || polyOrder >= windowSize) {
        HM_LOG_ERROR(TAG, "Invalid polynomial order " + std::to_string(polyOrder) +
                     " for window size " + std::to_string(windowSize));
        std::vector<float> identity(windowSize, 0.0f);
        identity[windowSize / 2] = 1.0f;
        return identity;
    }

    int halfWin = windowSize / 2;
    int n = windowSize;
    int m = polyOrder + 1;

    // Build Vandermonde matrix V of shape [n x m]
    // V(i, j) = x_i^j where x_i = i - halfWin (centred indices)
    Eigen::MatrixXd V(n, m);
    for (int i = 0; i < n; ++i) {
        double x = static_cast<double>(i - halfWin);
        double power = 1.0;
        for (int j = 0; j < m; ++j) {
            V(i, j) = power;
            power *= x;
        }
    }

    // Solve via pseudo-inverse: C = (V^T V)^{-1} V^T
    // Using QR decomposition for numerical stability instead of direct inverse
    Eigen::MatrixXd VtV = V.transpose() * V;

    // Use LDLT decomposition (symmetric positive definite) for better stability
    Eigen::LDLT<Eigen::MatrixXd> ldlt(VtV);
    if (ldlt.info() != Eigen::Success) {
        HM_LOG_ERROR(TAG, "LDLT decomposition failed for Vandermonde normal equations, "
                     "falling back to direct inverse");
        // Fallback to direct computation
        Eigen::MatrixXd VtVinv = VtV.inverse();
        Eigen::MatrixXd C = VtVinv * V.transpose();
        std::vector<float> coeffs(windowSize);
        for (int i = 0; i < windowSize; ++i) {
            coeffs[i] = static_cast<float>(C(0, i));
        }
        return coeffs;
    }

    // Solve VtV * X = V^T for X, then the first row of X is the smoothing coefficients
    Eigen::MatrixXd Vt = V.transpose();
    Eigen::MatrixXd C = ldlt.solve(Vt);

    // Verify solution quality
    double residual = (VtV * C - Vt).norm();
    if (residual > 1e-8) {
        HM_LOG_WARN(TAG, "Coefficient computation residual is " + std::to_string(residual) +
                    " (may indicate numerical issues)");
    }

    // First row of C gives the smoothing (derivative order 0) coefficients
    std::vector<float> coeffs(windowSize);
    for (int i = 0; i < windowSize; ++i) {
        coeffs[i] = static_cast<float>(C(0, i));
    }

    // Verify coefficients sum to approximately 1.0 (for smoothing, derivative=0)
    double coeffSum = 0.0;
    for (int i = 0; i < windowSize; ++i) {
        coeffSum += coeffs[i];
    }
    if (std::abs(coeffSum - 1.0) > 1e-4) {
        HM_LOG_WARN(TAG, "Smoothing coefficient sum is " + std::to_string(coeffSum) +
                    " (expected ~1.0), normalizing");
        if (std::abs(coeffSum) > 1e-10) {
            for (int i = 0; i < windowSize; ++i) {
                coeffs[i] /= static_cast<float>(coeffSum);
            }
        }
    }

    return coeffs;
}

void SavitzkyGolay::filterChannel(std::vector<float>& signal, int windowSize, int polyOrder) {
    int n = static_cast<int>(signal.size());

    // Validate parameters
    if (n < 3) {
        return; // Signal too short for any meaningful filtering
    }

    // Ensure window size is odd
    if (windowSize % 2 == 0) {
        windowSize++;
    }

    // Ensure polynomial order < window size
    if (polyOrder >= windowSize) {
        polyOrder = windowSize - 1;
    }

    // If signal is shorter than the window, reduce window to fit
    if (n < windowSize) {
        // Reduce window to largest odd number <= n
        windowSize = n;
        if (windowSize % 2 == 0) {
            windowSize--;
        }
        if (windowSize < 3) {
            return; // Cannot filter with fewer than 3 points
        }
        // Also clamp polynomial order
        if (polyOrder >= windowSize) {
            polyOrder = windowSize - 1;
        }
    }

    auto coeffs = computeCoefficients(windowSize, polyOrder);
    int halfWin = windowSize / 2;

    // Create extended signal with mirror boundary conditions
    // Mirror padding: reflect signal values at boundaries
    int padLen = halfWin;
    int extLen = n + 2 * padLen;
    std::vector<float> extended(extLen);

    // Left mirror padding: signal[padLen], signal[padLen-1], ..., signal[1]
    for (int i = 0; i < padLen; ++i) {
        int mirrorIdx = padLen - i;
        if (mirrorIdx >= n) mirrorIdx = n - 1;
        extended[i] = 2.0f * signal[0] - signal[mirrorIdx];
    }

    // Copy original signal
    for (int i = 0; i < n; ++i) {
        extended[padLen + i] = signal[i];
    }

    // Right mirror padding: signal[n-2], signal[n-3], ...
    for (int i = 0; i < padLen; ++i) {
        int mirrorIdx = n - 2 - i;
        if (mirrorIdx < 0) mirrorIdx = 0;
        extended[padLen + n + i] = 2.0f * signal[n - 1] - signal[mirrorIdx];
    }

    // Apply convolution on the extended signal
    std::vector<float> filtered(n);
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int j = 0; j < windowSize; ++j) {
            sum += coeffs[j] * extended[i + j];
        }
        filtered[i] = sum;
    }

    signal = filtered;
}

void SavitzkyGolay::process(std::vector<SkeletonFrame>& frames) {
    int numFrames = static_cast<int>(frames.size());
    if (numFrames < 3) {
        HM_LOG_DEBUG(TAG, "Skipping SavitzkyGolay: only " + std::to_string(numFrames) + " frames");
        return;
    }

    // Validate configuration
    if (config_.polyOrder >= config_.windowSize) {
        HM_LOG_ERROR(TAG, "Polynomial order (" + std::to_string(config_.polyOrder) +
                     ") must be less than window size (" + std::to_string(config_.windowSize) +
                     "). Adjusting.");
        config_.polyOrder = config_.windowSize - 1;
    }

    HM_LOG_INFO(TAG, "Processing " + std::to_string(numFrames) + " frames (window=" +
                std::to_string(config_.windowSize) + ", order=" + std::to_string(config_.polyOrder) + ")");

    // Compute RMS of original signal to measure filtering effect
    auto computeRMS = [](const std::vector<float>& sig) -> float {
        if (sig.empty()) return 0.0f;
        double sumSq = 0.0;
        for (float v : sig) sumSq += static_cast<double>(v) * v;
        return static_cast<float>(std::sqrt(sumSq / sig.size()));
    };

    float totalOrigRMS = 0.0f;
    float totalFilteredRMS = 0.0f;
    int channelCount = 0;

    // Helper: extract, filter, and write-back a 3-channel signal
    auto filterJointXYZ = [&](auto getX, auto getY, auto getZ,
                              auto setX, auto setY, auto setZ) {
        std::vector<float> xSig(numFrames), ySig(numFrames), zSig(numFrames);

        for (int f = 0; f < numFrames; ++f) {
            xSig[f] = getX(f);
            ySig[f] = getY(f);
            zSig[f] = getZ(f);
        }

        totalOrigRMS += computeRMS(xSig) + computeRMS(ySig) + computeRMS(zSig);

        filterChannel(xSig, config_.windowSize, config_.polyOrder);
        filterChannel(ySig, config_.windowSize, config_.polyOrder);
        filterChannel(zSig, config_.windowSize, config_.polyOrder);

        totalFilteredRMS += computeRMS(xSig) + computeRMS(ySig) + computeRMS(zSig);
        channelCount += 3;

        for (int f = 0; f < numFrames; ++f) {
            setX(f, xSig[f]);
            setY(f, ySig[f]);
            setZ(f, zSig[f]);
        }
    };

    // -------------------------------------------------------------------
    // Process each joint's world position per-axis
    // -------------------------------------------------------------------
    for (int j = 0; j < JOINT_COUNT; ++j) {
        filterJointXYZ(
            [&](int f) { return frames[f].joints[j].worldPosition.x; },
            [&](int f) { return frames[f].joints[j].worldPosition.y; },
            [&](int f) { return frames[f].joints[j].worldPosition.z; },
            [&](int f, float v) { frames[f].joints[j].worldPosition.x = v; },
            [&](int f, float v) { frames[f].joints[j].worldPosition.y = v; },
            [&](int f, float v) { frames[f].joints[j].worldPosition.z = v; }
        );
    }

    // -------------------------------------------------------------------
    // Process root position
    // -------------------------------------------------------------------
    filterJointXYZ(
        [&](int f) { return frames[f].rootPosition.x; },
        [&](int f) { return frames[f].rootPosition.y; },
        [&](int f) { return frames[f].rootPosition.z; },
        [&](int f, float v) { frames[f].rootPosition.x = v; },
        [&](int f, float v) { frames[f].rootPosition.y = v; },
        [&](int f, float v) { frames[f].rootPosition.z = v; }
    );

    // -------------------------------------------------------------------
    // Process root velocity
    // -------------------------------------------------------------------
    filterJointXYZ(
        [&](int f) { return frames[f].rootVelocity.x; },
        [&](int f) { return frames[f].rootVelocity.y; },
        [&](int f) { return frames[f].rootVelocity.z; },
        [&](int f, float v) { frames[f].rootVelocity.x = v; },
        [&](int f, float v) { frames[f].rootVelocity.y = v; },
        [&](int f, float v) { frames[f].rootVelocity.z = v; }
    );

    // -------------------------------------------------------------------
    // Process 6D rotation components per joint
    // -------------------------------------------------------------------
    for (int j = 0; j < JOINT_COUNT; ++j) {
        for (int d = 0; d < ROTATION_DIM; d += 3) {
            int d0 = d;
            int d1 = std::min(d + 1, ROTATION_DIM - 1);
            int d2 = std::min(d + 2, ROTATION_DIM - 1);

            filterJointXYZ(
                [&](int f) { return frames[f].joints[j].rotation6D[d0]; },
                [&](int f) { return frames[f].joints[j].rotation6D[d1]; },
                [&](int f) { return frames[f].joints[j].rotation6D[d2]; },
                [&](int f, float v) { frames[f].joints[j].rotation6D[d0] = v; },
                [&](int f, float v) { frames[f].joints[j].rotation6D[d1] = v; },
                [&](int f, float v) { frames[f].joints[j].rotation6D[d2] = v; }
            );
        }
    }

    // -------------------------------------------------------------------
    // Rebuild quaternions and Euler angles from smoothed 6D rotations
    // -------------------------------------------------------------------
    for (int f = 0; f < numFrames; ++f) {
        for (int j = 0; j < JOINT_COUNT; ++j) {
            auto& jt = frames[f].joints[j];
            jt.localRotation = MathUtils::rot6DToQuat(jt.rotation6D);
            jt.localEulerDeg = MathUtils::quatToEulerDeg(jt.localRotation);
        }
    }

    HM_LOG_INFO(TAG, "SavitzkyGolay complete: processed " + std::to_string(channelCount) + " channels");
}

} // namespace hm::signal
