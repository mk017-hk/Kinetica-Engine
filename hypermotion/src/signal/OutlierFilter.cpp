#include "HyperMotion/signal/OutlierFilter.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace hm::signal {

static constexpr const char* TAG = "OutlierFilter";

OutlierFilter::OutlierFilter(const OutlierFilterConfig& config)
    : config_(config) {}

OutlierFilter::~OutlierFilter() = default;

void OutlierFilter::filterChannel(std::vector<float>& signal, int windowSize, float threshold) {
    if (signal.size() < static_cast<size_t>(windowSize)) return;

    int halfWin = windowSize / 2;
    std::vector<float> filtered = signal;

    for (int i = 0; i < static_cast<int>(signal.size()); ++i) {
        int start = std::max(0, i - halfWin);
        int end = std::min(static_cast<int>(signal.size()) - 1, i + halfWin);
        int count = end - start + 1;

        // Extract local window
        std::vector<float> window(count);
        for (int j = 0; j < count; ++j) {
            window[j] = signal[start + j];
        }

        // Compute median
        std::sort(window.begin(), window.end());
        float median = window[count / 2];
        if (count % 2 == 0 && count > 1) {
            median = (window[count / 2 - 1] + window[count / 2]) * 0.5f;
        }

        // Compute MAD (Median Absolute Deviation)
        std::vector<float> absDevs(count);
        for (int j = 0; j < count; ++j) {
            absDevs[j] = std::abs(window[j] - median);
        }
        std::sort(absDevs.begin(), absDevs.end());
        float mad = absDevs[count / 2];
        if (count % 2 == 0 && count > 1) {
            mad = (absDevs[count / 2 - 1] + absDevs[count / 2]) * 0.5f;
        }

        // Scale MAD for normal distribution consistency
        float scaledMAD = mad * 1.4826f;

        // Replace outliers
        if (scaledMAD > 1e-8f) {
            float deviation = std::abs(signal[i] - median) / scaledMAD;
            if (deviation > threshold) {
                filtered[i] = median;
            }
        }
    }

    signal = filtered;
}

void OutlierFilter::process(std::vector<SkeletonFrame>& frames) {
    if (frames.size() < 3) return;

    int numFrames = static_cast<int>(frames.size());

    HM_LOG_DEBUG(TAG, "Processing " + std::to_string(numFrames) + " frames");

    // Process each joint's world position X, Y, Z independently
    for (int j = 0; j < JOINT_COUNT; ++j) {
        std::vector<float> xSignal(numFrames), ySignal(numFrames), zSignal(numFrames);

        for (int f = 0; f < numFrames; ++f) {
            xSignal[f] = frames[f].joints[j].worldPosition.x;
            ySignal[f] = frames[f].joints[j].worldPosition.y;
            zSignal[f] = frames[f].joints[j].worldPosition.z;
        }

        filterChannel(xSignal, config_.windowSize, config_.madThreshold);
        filterChannel(ySignal, config_.windowSize, config_.madThreshold);
        filterChannel(zSignal, config_.windowSize, config_.madThreshold);

        for (int f = 0; f < numFrames; ++f) {
            frames[f].joints[j].worldPosition.x = xSignal[f];
            frames[f].joints[j].worldPosition.y = ySignal[f];
            frames[f].joints[j].worldPosition.z = zSignal[f];
        }
    }

    // Also filter root position
    {
        std::vector<float> rx(numFrames), ry(numFrames), rz(numFrames);
        for (int f = 0; f < numFrames; ++f) {
            rx[f] = frames[f].rootPosition.x;
            ry[f] = frames[f].rootPosition.y;
            rz[f] = frames[f].rootPosition.z;
        }
        filterChannel(rx, config_.windowSize, config_.madThreshold);
        filterChannel(ry, config_.windowSize, config_.madThreshold);
        filterChannel(rz, config_.windowSize, config_.madThreshold);
        for (int f = 0; f < numFrames; ++f) {
            frames[f].rootPosition = {rx[f], ry[f], rz[f]};
        }
    }
}

} // namespace hm::signal
