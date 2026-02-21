#include "HyperMotion/signal/ButterworthFilter.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

#include <cmath>
#include <algorithm>
#include <complex>

namespace hm::signal {

static constexpr const char* TAG = "ButterworthFilter";
static constexpr float PI = 3.14159265358979323846f;

ButterworthFilter::ButterworthFilter(const ButterworthConfig& config)
    : config_(config) {}

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

    // Pre-warp for bilinear transform
    float warped = 2.0f * sampleRate * std::tan(PI * cutoffFreq / sampleRate);

    // Analog Butterworth prototype poles
    int numSections = order / 2;
    bool hasFirstOrder = (order % 2) != 0;

    // Start with b = [1], a = [1]
    std::vector<double> b_total = {1.0};
    std::vector<double> a_total = {1.0};

    auto polyMultiply = [](const std::vector<double>& a, const std::vector<double>& b) {
        std::vector<double> result(a.size() + b.size() - 1, 0.0);
        for (size_t i = 0; i < a.size(); ++i)
            for (size_t j = 0; j < b.size(); ++j)
                result[i + j] += a[i] * b[j];
        return result;
    };

    float T = 1.0f / sampleRate;

    // First-order section if order is odd
    if (hasFirstOrder) {
        // s / (s + warped) -> bilinear: z-domain coefficients
        double c = 2.0 * sampleRate;
        double denom = c + warped;
        std::vector<double> b_sec = {warped / denom, warped / denom};
        std::vector<double> a_sec = {1.0, (warped - c) / denom};
        b_total = polyMultiply(b_total, b_sec);
        a_total = polyMultiply(a_total, a_sec);
    }

    // Second-order sections
    for (int k = 0; k < numSections; ++k) {
        double theta = PI * (2.0 * (k + 1) + order - 1) / (2.0 * order);
        double alpha = -2.0 * std::cos(theta);

        // Analog: 1 / (s^2 + alpha*warped*s + warped^2)
        // Bilinear transform to z-domain
        double c = 2.0 * sampleRate;
        double c2 = c * c;
        double w2 = static_cast<double>(warped) * warped;
        double aw = alpha * warped;

        double a0 = c2 + aw * c + w2;
        double a1 = 2.0 * (w2 - c2);
        double a2 = c2 - aw * c + w2;

        std::vector<double> b_sec = {w2 / a0, 2.0 * w2 / a0, w2 / a0};
        std::vector<double> a_sec = {1.0, a1 / a0, a2 / a0};

        b_total = polyMultiply(b_total, b_sec);
        a_total = polyMultiply(a_total, a_sec);
    }

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

    auto coeffs = designLowpass(order, cutoffFreq, sampleRate);

    int numCoeffs = static_cast<int>(coeffs.b.size());

    auto applyIIR = [&](std::vector<float>& sig, bool reverse) {
        int len = static_cast<int>(sig.size());
        std::vector<float> output(len, 0.0f);

        for (int i = 0; i < len; ++i) {
            int idx = reverse ? (len - 1 - i) : i;

            float sum = 0.0f;
            for (int k = 0; k < numCoeffs; ++k) {
                int srcIdx = reverse ? (idx + k) : (idx - k);
                if (srcIdx >= 0 && srcIdx < len) {
                    sum += coeffs.b[k] * sig[srcIdx];
                }
            }
            for (int k = 1; k < numCoeffs && k < static_cast<int>(coeffs.a.size()); ++k) {
                int outIdx = reverse ? (idx + k) : (idx - k);
                if (outIdx >= 0 && outIdx < len) {
                    sum -= coeffs.a[k] * output[outIdx];
                }
            }
            output[idx] = sum;
        }

        sig = output;
    };

    // Forward-backward filtering for zero phase distortion
    applyIIR(signal, false);  // Forward pass
    applyIIR(signal, true);   // Backward pass
}

void ButterworthFilter::process(std::vector<SkeletonFrame>& frames) {
    if (frames.size() < 3) return;

    int numFrames = static_cast<int>(frames.size());
    HM_LOG_DEBUG(TAG, "Processing " + std::to_string(numFrames) + " frames");

    for (int j = 0; j < JOINT_COUNT; ++j) {
        float cutoff = isExtremity(j) ? config_.cutoffFreqExtrem : config_.cutoffFreqBody;

        std::vector<float> xSignal(numFrames), ySignal(numFrames), zSignal(numFrames);

        for (int f = 0; f < numFrames; ++f) {
            xSignal[f] = frames[f].joints[j].worldPosition.x;
            ySignal[f] = frames[f].joints[j].worldPosition.y;
            zSignal[f] = frames[f].joints[j].worldPosition.z;
        }

        filterChannel(xSignal, config_.order, cutoff, config_.sampleRate);
        filterChannel(ySignal, config_.order, cutoff, config_.sampleRate);
        filterChannel(zSignal, config_.order, cutoff, config_.sampleRate);

        for (int f = 0; f < numFrames; ++f) {
            frames[f].joints[j].worldPosition.x = xSignal[f];
            frames[f].joints[j].worldPosition.y = ySignal[f];
            frames[f].joints[j].worldPosition.z = zSignal[f];
        }
    }
}

} // namespace hm::signal
