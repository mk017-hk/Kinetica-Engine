#include "HyperMotion/signal/ButterworthFilter.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

#include <cmath>
#include <algorithm>
#include <complex>
#include <vector>
#include <string>
#include <numeric>

namespace hm::signal {

static constexpr const char* TAG = "ButterworthFilter";
static constexpr double PI = 3.14159265358979323846;

ButterworthFilter::ButterworthFilter(const ButterworthConfig& config)
    : config_(config) {
    // Validate cutoff frequencies against Nyquist
    float nyquist = config_.sampleRate * 0.5f;
    if (config_.cutoffFreqBody >= nyquist) {
        config_.cutoffFreqBody = nyquist * 0.9f;
        HM_LOG_WARN(TAG, "Body cutoff frequency exceeds Nyquist, clamped to " +
                    std::to_string(config_.cutoffFreqBody) + " Hz");
    }
    if (config_.cutoffFreqExtrem >= nyquist) {
        config_.cutoffFreqExtrem = nyquist * 0.9f;
        HM_LOG_WARN(TAG, "Extremity cutoff frequency exceeds Nyquist, clamped to " +
                    std::to_string(config_.cutoffFreqExtrem) + " Hz");
    }
    if (config_.order < 1) {
        config_.order = 1;
        HM_LOG_WARN(TAG, "Filter order must be at least 1, clamped to 1");
    }
    if (config_.order > 10) {
        config_.order = 10;
        HM_LOG_WARN(TAG, "Filter order capped at 10 for numerical stability");
    }
}

ButterworthFilter::~ButterworthFilter() = default;

bool ButterworthFilter::isExtremity(int jointIndex) {
    return jointIndex == static_cast<int>(Joint::LeftHand) ||
           jointIndex == static_cast<int>(Joint::RightHand) ||
           jointIndex == static_cast<int>(Joint::LeftFoot) ||
           jointIndex == static_cast<int>(Joint::RightFoot) ||
           jointIndex == static_cast<int>(Joint::LeftToeBase) ||
           jointIndex == static_cast<int>(Joint::RightToeBase);
}

ButterworthFilter::FilterCoefficients ButterworthFilter::designLowpass(
    int order, float cutoffFreq, float sampleRate) {

    FilterCoefficients coeffs;

    // Validate inputs
    float nyquist = sampleRate * 0.5f;
    if (cutoffFreq <= 0.0f || cutoffFreq >= nyquist) {
        HM_LOG_ERROR(TAG, "Invalid cutoff frequency " + std::to_string(cutoffFreq) +
                     " Hz (Nyquist = " + std::to_string(nyquist) + " Hz). Returning pass-through.");
        coeffs.b = {1.0f};
        coeffs.a = {1.0f};
        return coeffs;
    }
    if (order < 1) {
        coeffs.b = {1.0f};
        coeffs.a = {1.0f};
        return coeffs;
    }

    // Pre-warp analog frequency for bilinear transform
    // omega_a = 2 * fs * tan(pi * fc / fs)
    double omega_a = 2.0 * static_cast<double>(sampleRate) *
                     std::tan(PI * static_cast<double>(cutoffFreq) / static_cast<double>(sampleRate));

    // Compute analog Butterworth prototype poles on the unit circle in the s-plane
    // Poles of an N-th order Butterworth filter are at:
    //   s_k = omega_a * exp(j * pi * (2k + N + 1) / (2N)),  k = 0, 1, ..., N-1
    // These lie on a circle of radius omega_a in the left half of the s-plane.

    // We will compute digital coefficients by cascading second-order sections (SOS)
    // and one first-order section if order is odd.

    int numSections = order / 2;
    bool hasFirstOrder = (order % 2) != 0;

    // Polynomial multiplication helper
    auto polyMultiply = [](const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> result(a.size() + b.size() - 1, 0.0);
        for (size_t i = 0; i < a.size(); ++i) {
            for (size_t j = 0; j < b.size(); ++j) {
                result[i + j] += a[i] * b[j];
            }
        }
        return result;
    };

    // c = 2 * fs (bilinear transform constant)
    double c = 2.0 * static_cast<double>(sampleRate);
    double c2 = c * c;
    double w = omega_a;
    double w2 = w * w;

    // Start with b_total = [1], a_total = [1]
    std::vector<double> b_total = {1.0};
    std::vector<double> a_total = {1.0};

    // -------------------------------------------------------------------
    // First-order section if order is odd
    // -------------------------------------------------------------------
    // Analog: H(s) = w / (s + w)
    // Bilinear: s = c * (z-1)/(z+1)
    // H(z) = w*(z+1) / (c*(z-1) + w*(z+1))
    //       = w*(z+1) / ((c+w)*z + (w-c))
    // Normalise by (c+w):
    //   b = [w/(c+w), w/(c+w)]
    //   a = [1, (w-c)/(c+w)]
    if (hasFirstOrder) {
        double denom = c + w;
        if (std::abs(denom) < 1e-15) {
            HM_LOG_ERROR(TAG, "Degenerate first-order section denominator");
            coeffs.b = {1.0f};
            coeffs.a = {1.0f};
            return coeffs;
        }
        std::vector<double> b_sec = {w / denom, w / denom};
        std::vector<double> a_sec = {1.0, (w - c) / denom};
        b_total = polyMultiply(b_total, b_sec);
        a_total = polyMultiply(a_total, a_sec);
    }

    // -------------------------------------------------------------------
    // Second-order sections
    // -------------------------------------------------------------------
    // For each conjugate pole pair k:
    //   theta_k = pi * (2*(k+1) + N - 1) / (2*N)
    //   alpha_k = -2 * cos(theta_k) (coefficient of s in s^2 + alpha*w*s + w^2)
    //
    // Analog: H_k(s) = w^2 / (s^2 + alpha_k*w*s + w^2)
    //
    // After bilinear transform s = c*(z-1)/(z+1):
    //   numerator = w^2 * (z+1)^2 = w^2 * (z^2 + 2z + 1)
    //   denominator = c^2*(z-1)^2 + alpha_k*w*c*(z^2-1) + w^2*(z+1)^2
    //               = (c^2 + alpha_k*w*c + w^2)*z^2
    //                 + 2*(w^2 - c^2)*z
    //                 + (c^2 - alpha_k*w*c + w^2)
    //
    // Normalise by a0 = c^2 + alpha_k*w*c + w^2

    for (int k = 0; k < numSections; ++k) {
        double theta = PI * (2.0 * (k + 1) + order - 1) / (2.0 * order);
        double alpha_k = -2.0 * std::cos(theta);

        double aw = alpha_k * w;

        double a0 = c2 + aw * c + w2;
        double a1 = 2.0 * (w2 - c2);
        double a2 = c2 - aw * c + w2;

        if (std::abs(a0) < 1e-15) {
            HM_LOG_ERROR(TAG, "Degenerate second-order section denominator at section " +
                        std::to_string(k));
            continue;
        }

        std::vector<double> b_sec = {w2 / a0, 2.0 * w2 / a0, w2 / a0};
        std::vector<double> a_sec = {1.0, a1 / a0, a2 / a0};

        b_total = polyMultiply(b_total, b_sec);
        a_total = polyMultiply(a_total, a_sec);
    }

    // -------------------------------------------------------------------
    // Verify DC gain is 1.0
    // -------------------------------------------------------------------
    // At DC (z=1), H(z) = sum(b) / sum(a) should be 1.0 for a lowpass
    double bSum = 0.0, aSum = 0.0;
    for (double v : b_total) bSum += v;
    for (double v : a_total) aSum += v;

    if (std::abs(aSum) > 1e-15) {
        double dcGain = bSum / aSum;
        if (std::abs(dcGain - 1.0) > 1e-6) {
            HM_LOG_DEBUG(TAG, "DC gain correction: " + std::to_string(dcGain) + " -> 1.0");
            // Normalize b coefficients to achieve unity DC gain
            double correction = aSum / bSum;
            for (double& v : b_total) {
                v *= correction;
            }
        }
    }

    // -------------------------------------------------------------------
    // Check filter stability: all poles inside unit circle
    // (all roots of a(z) polynomial must have |z| < 1)
    // For a 2nd-order section with a = [1, a1, a2], stability requires |a2| < 1
    // We do a simple check on the overall denominator coefficients
    // -------------------------------------------------------------------
    bool stable = true;
    if (a_total.size() > 1) {
        // Check that trailing coefficient magnitude < 1 (necessary condition)
        double lastCoeff = std::abs(a_total.back());
        if (lastCoeff >= 1.0) {
            HM_LOG_WARN(TAG, "Filter may be unstable: |a[N]| = " + std::to_string(lastCoeff));
            stable = false;
        }
    }
    if (!stable) {
        HM_LOG_WARN(TAG, "Potentially unstable filter design for cutoff=" +
                    std::to_string(cutoffFreq) + " Hz, order=" + std::to_string(order));
    }

    // Convert to float
    coeffs.b.resize(b_total.size());
    coeffs.a.resize(a_total.size());
    for (size_t i = 0; i < b_total.size(); ++i) coeffs.b[i] = static_cast<float>(b_total[i]);
    for (size_t i = 0; i < a_total.size(); ++i) coeffs.a[i] = static_cast<float>(a_total[i]);

    return coeffs;
}

void ButterworthFilter::filterChannel(
    std::vector<float>& signal, int order, float cutoffFreq, float sampleRate) {

    int n = static_cast<int>(signal.size());
    if (n < 3) return;

    // Validate Nyquist
    if (cutoffFreq <= 0.0f || cutoffFreq >= sampleRate * 0.5f) {
        HM_LOG_WARN(TAG, "Cutoff frequency " + std::to_string(cutoffFreq) +
                    " Hz is invalid for sample rate " + std::to_string(sampleRate) + " Hz, skipping");
        return;
    }

    auto coeffs = designLowpass(order, cutoffFreq, sampleRate);

    int numB = static_cast<int>(coeffs.b.size());
    int numA = static_cast<int>(coeffs.a.size());
    int filterOrder = std::max(numB, numA) - 1;

    if (filterOrder < 1) return;

    // -------------------------------------------------------------------
    // Edge padding to reduce transient artefacts
    // Gustafsson's method: extend signal using reflected values
    // Pad length = 3 * filter order (as in scipy.signal.filtfilt)
    // -------------------------------------------------------------------
    int padLen = std::min(3 * filterOrder, n - 1);

    std::vector<float> padded(n + 2 * padLen);

    // Left padding: reflect about the first sample
    // padded[padLen - i - 1] = 2 * signal[0] - signal[i + 1]
    for (int i = 0; i < padLen; ++i) {
        int srcIdx = std::min(i + 1, n - 1);
        padded[padLen - 1 - i] = 2.0f * signal[0] - signal[srcIdx];
    }

    // Copy original signal
    for (int i = 0; i < n; ++i) {
        padded[padLen + i] = signal[i];
    }

    // Right padding: reflect about the last sample
    // padded[padLen + n + i] = 2 * signal[n-1] - signal[n-2-i]
    for (int i = 0; i < padLen; ++i) {
        int srcIdx = std::max(n - 2 - i, 0);
        padded[padLen + n + i] = 2.0f * signal[n - 1] - signal[srcIdx];
    }

    int paddedLen = static_cast<int>(padded.size());

    // -------------------------------------------------------------------
    // Compute initial conditions for the filter to minimize startup transient
    // Uses the approach from scipy: solve for steady-state response to
    // the initial value (Gustafsson's method simplified)
    // -------------------------------------------------------------------
    auto computeInitialConditions = [&](float initialValue) -> std::vector<float> {
        // For a filter with N states, we solve:
        //   (I - A) * zi = B - A_col1
        // This is a simplification that works for most practical cases.
        // We use a simpler approach: run the filter on a constant signal
        // of length 2*filterOrder to let it settle, then capture state.
        // Instead, we just zero-initialize and accept a small transient
        // at the padded edges (which get trimmed anyway).
        return std::vector<float>(std::max(numB, numA), 0.0f);
    };

    // -------------------------------------------------------------------
    // Forward IIR pass (Direct Form II Transposed)
    // -------------------------------------------------------------------
    auto applyIIRForward = [&](std::vector<float>& sig) {
        int len = static_cast<int>(sig.size());
        int stateLen = std::max(numB, numA);
        std::vector<double> state(stateLen, 0.0);

        // Initialise state for steady-state response to first sample
        // Approximate: fill state buffer so output of first sample ~= first sample
        // This reduces startup transient significantly
        {
            double firstVal = static_cast<double>(sig[0]);
            // Compute steady-state filter state for a constant input of value firstVal
            // For a DC input x[n] = C, the steady-state output is C (DC gain = 1)
            // The state values can be derived from the difference equation
            // z[k] = sum_{i=k+1}^{M} (b[i] - a[i]) * C  where M = max(numB, numA) - 1
            for (int k = 0; k < stateLen - 1; ++k) {
                double bSum = 0.0, aSum = 0.0;
                for (int i = k + 1; i < stateLen; ++i) {
                    if (i < numB) bSum += coeffs.b[i];
                    if (i < numA) aSum += coeffs.a[i];
                }
                state[k] = (bSum - aSum) * firstVal;
            }
        }

        for (int i = 0; i < len; ++i) {
            double input = static_cast<double>(sig[i]);
            double output = static_cast<double>(coeffs.b[0]) * input + state[0];

            // Update state (shift register)
            for (int k = 0; k < stateLen - 1; ++k) {
                double bk1 = (k + 1 < numB) ? static_cast<double>(coeffs.b[k + 1]) : 0.0;
                double ak1 = (k + 1 < numA) ? static_cast<double>(coeffs.a[k + 1]) : 0.0;
                state[k] = bk1 * input - ak1 * output + state[k + 1];
            }
            state[stateLen - 1] = 0.0;

            sig[i] = static_cast<float>(output);
        }
    };

    // -------------------------------------------------------------------
    // Backward IIR pass (same filter, reversed signal)
    // -------------------------------------------------------------------
    auto applyIIRBackward = [&](std::vector<float>& sig) {
        int len = static_cast<int>(sig.size());

        // Reverse the signal
        std::reverse(sig.begin(), sig.end());

        int stateLen = std::max(numB, numA);
        std::vector<double> state(stateLen, 0.0);

        // Initialise state for the reversed signal's first sample
        {
            double firstVal = static_cast<double>(sig[0]);
            for (int k = 0; k < stateLen - 1; ++k) {
                double bSum = 0.0, aSum = 0.0;
                for (int i = k + 1; i < stateLen; ++i) {
                    if (i < numB) bSum += coeffs.b[i];
                    if (i < numA) aSum += coeffs.a[i];
                }
                state[k] = (bSum - aSum) * firstVal;
            }
        }

        for (int i = 0; i < len; ++i) {
            double input = static_cast<double>(sig[i]);
            double output = static_cast<double>(coeffs.b[0]) * input + state[0];

            for (int k = 0; k < stateLen - 1; ++k) {
                double bk1 = (k + 1 < numB) ? static_cast<double>(coeffs.b[k + 1]) : 0.0;
                double ak1 = (k + 1 < numA) ? static_cast<double>(coeffs.a[k + 1]) : 0.0;
                state[k] = bk1 * input - ak1 * output + state[k + 1];
            }
            state[stateLen - 1] = 0.0;

            sig[i] = static_cast<float>(output);
        }

        // Reverse back to original order
        std::reverse(sig.begin(), sig.end());
    };

    // -------------------------------------------------------------------
    // Apply forward-backward filtering for zero-phase distortion
    // -------------------------------------------------------------------
    applyIIRForward(padded);
    applyIIRBackward(padded);

    // -------------------------------------------------------------------
    // Extract the original-length portion (strip padding)
    // -------------------------------------------------------------------
    for (int i = 0; i < n; ++i) {
        signal[i] = padded[padLen + i];
    }
}

void ButterworthFilter::process(std::vector<SkeletonFrame>& frames) {
    int numFrames = static_cast<int>(frames.size());
    if (numFrames < 3) {
        HM_LOG_DEBUG(TAG, "Skipping Butterworth filter: only " + std::to_string(numFrames) + " frames");
        return;
    }

    HM_LOG_INFO(TAG, "Processing " + std::to_string(numFrames) + " frames (order=" +
                std::to_string(config_.order) + ", body_cutoff=" +
                std::to_string(config_.cutoffFreqBody) + "Hz, extremity_cutoff=" +
                std::to_string(config_.cutoffFreqExtrem) + "Hz, sample_rate=" +
                std::to_string(config_.sampleRate) + "Hz)");

    int bodyJoints = 0;
    int extremJoints = 0;

    // Helper: extract, filter, and write-back a 3-channel signal for a joint
    auto filterJointXYZ = [&](int jointIdx) {
        float cutoff = isExtremity(jointIdx) ? config_.cutoffFreqExtrem : config_.cutoffFreqBody;
        if (isExtremity(jointIdx)) {
            extremJoints++;
        } else {
            bodyJoints++;
        }

        std::vector<float> xSig(numFrames), ySig(numFrames), zSig(numFrames);

        for (int f = 0; f < numFrames; ++f) {
            xSig[f] = frames[f].joints[jointIdx].worldPosition.x;
            ySig[f] = frames[f].joints[jointIdx].worldPosition.y;
            zSig[f] = frames[f].joints[jointIdx].worldPosition.z;
        }

        filterChannel(xSig, config_.order, cutoff, config_.sampleRate);
        filterChannel(ySig, config_.order, cutoff, config_.sampleRate);
        filterChannel(zSig, config_.order, cutoff, config_.sampleRate);

        for (int f = 0; f < numFrames; ++f) {
            frames[f].joints[jointIdx].worldPosition.x = xSig[f];
            frames[f].joints[jointIdx].worldPosition.y = ySig[f];
            frames[f].joints[jointIdx].worldPosition.z = zSig[f];
        }
    };

    // -------------------------------------------------------------------
    // Process all joints with appropriate cutoff frequencies
    // -------------------------------------------------------------------
    for (int j = 0; j < JOINT_COUNT; ++j) {
        filterJointXYZ(j);
    }

    // -------------------------------------------------------------------
    // Also filter 6D rotation channels for each joint
    // -------------------------------------------------------------------
    for (int j = 0; j < JOINT_COUNT; ++j) {
        float cutoff = isExtremity(j) ? config_.cutoffFreqExtrem : config_.cutoffFreqBody;

        for (int d = 0; d < ROTATION_DIM; ++d) {
            std::vector<float> rotSig(numFrames);
            for (int f = 0; f < numFrames; ++f) {
                rotSig[f] = frames[f].joints[j].rotation6D[d];
            }

            filterChannel(rotSig, config_.order, cutoff, config_.sampleRate);

            for (int f = 0; f < numFrames; ++f) {
                frames[f].joints[j].rotation6D[d] = rotSig[f];
            }
        }
    }

    // -------------------------------------------------------------------
    // Rebuild quaternions and Euler angles from filtered 6D rotations
    // -------------------------------------------------------------------
    for (int f = 0; f < numFrames; ++f) {
        for (int j = 0; j < JOINT_COUNT; ++j) {
            auto& jt = frames[f].joints[j];
            jt.localRotation = MathUtils::rot6DToQuat(jt.rotation6D);
            jt.localEulerDeg = MathUtils::quatToEulerDeg(jt.localRotation);
        }
    }

    // -------------------------------------------------------------------
    // Filter root position and velocity
    // -------------------------------------------------------------------
    {
        std::vector<float> rx(numFrames), ry(numFrames), rz(numFrames);
        for (int f = 0; f < numFrames; ++f) {
            rx[f] = frames[f].rootPosition.x;
            ry[f] = frames[f].rootPosition.y;
            rz[f] = frames[f].rootPosition.z;
        }
        filterChannel(rx, config_.order, config_.cutoffFreqBody, config_.sampleRate);
        filterChannel(ry, config_.order, config_.cutoffFreqBody, config_.sampleRate);
        filterChannel(rz, config_.order, config_.cutoffFreqBody, config_.sampleRate);
        for (int f = 0; f < numFrames; ++f) {
            frames[f].rootPosition = {rx[f], ry[f], rz[f]};
        }
    }

    {
        std::vector<float> vx(numFrames), vy(numFrames), vz(numFrames);
        for (int f = 0; f < numFrames; ++f) {
            vx[f] = frames[f].rootVelocity.x;
            vy[f] = frames[f].rootVelocity.y;
            vz[f] = frames[f].rootVelocity.z;
        }
        filterChannel(vx, config_.order, config_.cutoffFreqBody, config_.sampleRate);
        filterChannel(vy, config_.order, config_.cutoffFreqBody, config_.sampleRate);
        filterChannel(vz, config_.order, config_.cutoffFreqBody, config_.sampleRate);
        for (int f = 0; f < numFrames; ++f) {
            frames[f].rootVelocity = {vx[f], vy[f], vz[f]};
        }
    }

    HM_LOG_INFO(TAG, "Butterworth complete: " + std::to_string(bodyJoints) +
                " body joints at " + std::to_string(config_.cutoffFreqBody) + "Hz, " +
                std::to_string(extremJoints) + " extremity joints at " +
                std::to_string(config_.cutoffFreqExtrem) + "Hz");
}

} // namespace hm::signal
