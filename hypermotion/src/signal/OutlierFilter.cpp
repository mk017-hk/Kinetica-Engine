#include "HyperMotion/signal/OutlierFilter.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
#include <string>

namespace hm::signal {

static constexpr const char* TAG = "OutlierFilter";

// Scale factor to make MAD consistent with standard deviation for a normal distribution.
// MAD * 1.4826 approximates the standard deviation.
static constexpr float kMADScaleFactor = 1.4826f;

// Minimum MAD below which we skip outlier detection (signal is essentially constant)
static constexpr float kMinMAD = 1e-8f;

OutlierFilter::OutlierFilter(const OutlierFilterConfig& config)
    : config_(config) {
    // Ensure window size is at least 3 and odd
    if (config_.windowSize < 3) {
        config_.windowSize = 3;
    }
    if (config_.windowSize % 2 == 0) {
        config_.windowSize++;
    }
    if (config_.madThreshold <= 0.0f) {
        HM_LOG_WARN(TAG, "MAD threshold must be positive, clamping to 1.0");
        config_.madThreshold = 1.0f;
    }
}

OutlierFilter::~OutlierFilter() = default;

float OutlierFilter::computeMedian(std::vector<float>& values) {
    if (values.empty()) return 0.0f;

    size_t n = values.size();
    std::sort(values.begin(), values.end());

    if (n % 2 == 1) {
        return values[n / 2];
    } else {
        return (values[n / 2 - 1] + values[n / 2]) * 0.5f;
    }
}

float OutlierFilter::interpolateFromNeighbours(const std::vector<float>& signal,
                                                const std::vector<bool>& isOutlier,
                                                int index) {
    int n = static_cast<int>(signal.size());

    // Search for nearest non-outlier on the left
    int leftIdx = -1;
    for (int i = index - 1; i >= 0; --i) {
        if (!isOutlier[i]) {
            leftIdx = i;
            break;
        }
    }

    // Search for nearest non-outlier on the right
    int rightIdx = -1;
    for (int i = index + 1; i < n; ++i) {
        if (!isOutlier[i]) {
            rightIdx = i;
            break;
        }
    }

    // Interpolation cases
    if (leftIdx >= 0 && rightIdx >= 0) {
        // Linear interpolation between two valid neighbours
        float t = static_cast<float>(index - leftIdx) /
                  static_cast<float>(rightIdx - leftIdx);
        return signal[leftIdx] * (1.0f - t) + signal[rightIdx] * t;
    } else if (leftIdx >= 0) {
        // Only left neighbour available - use it directly
        return signal[leftIdx];
    } else if (rightIdx >= 0) {
        // Only right neighbour available - use it directly
        return signal[rightIdx];
    } else {
        // No valid neighbours at all (entire signal is outliers) - return original value
        return signal[index];
    }
}

void OutlierFilter::filterChannel(std::vector<float>& signal, int windowSize, float threshold) {
    filterChannelWithMode(signal, windowSize, threshold, OutlierReplaceMode::Median);
}

void OutlierFilter::filterChannelWithMode(std::vector<float>& signal, int windowSize,
                                           float threshold, OutlierReplaceMode mode) {
    int n = static_cast<int>(signal.size());
    if (n < windowSize) return;

    int halfWin = windowSize / 2;

    // -------------------------------------------------------------------
    // Pass 1: Detect all outliers using sliding window MAD
    // -------------------------------------------------------------------
    std::vector<bool> isOutlier(n, false);
    std::vector<float> localMedians(n, 0.0f);
    std::vector<float> localScaledMADs(n, 0.0f);

    for (int i = 0; i < n; ++i) {
        // Compute window boundaries with proper edge handling
        int wStart = std::max(0, i - halfWin);
        int wEnd = std::min(n - 1, i + halfWin);
        int count = wEnd - wStart + 1;

        // Extract local window values
        std::vector<float> window(count);
        for (int j = 0; j < count; ++j) {
            window[j] = signal[wStart + j];
        }

        // Compute median
        float median = computeMedian(window);
        localMedians[i] = median;

        // Compute MAD (Median Absolute Deviation)
        std::vector<float> absDevs(count);
        for (int j = 0; j < count; ++j) {
            absDevs[j] = std::abs(window[j] - median);
        }
        float mad = computeMedian(absDevs);

        // Scale MAD for normal distribution consistency
        float scaledMAD = mad * kMADScaleFactor;
        localScaledMADs[i] = scaledMAD;

        // Mark outlier if deviation exceeds threshold
        if (scaledMAD > kMinMAD) {
            float deviation = std::abs(signal[i] - median) / scaledMAD;
            if (deviation > threshold) {
                isOutlier[i] = true;
            }
        }
        // If scaledMAD is essentially zero, the signal is locally constant.
        // A sample that deviates from a constant neighbourhood is likely an outlier.
        // Use absolute difference from median as fallback.
        else if (std::abs(signal[i] - median) > 1e-4f && count >= 5) {
            // If median is constant but this sample deviates, flag it
            isOutlier[i] = true;
        }
    }

    // -------------------------------------------------------------------
    // Pass 2: Replace outliers according to the selected mode
    // -------------------------------------------------------------------
    std::vector<float> filtered = signal;

    for (int i = 0; i < n; ++i) {
        if (!isOutlier[i]) continue;

        switch (mode) {
            case OutlierReplaceMode::Median:
                filtered[i] = localMedians[i];
                break;

            case OutlierReplaceMode::Interpolate:
                filtered[i] = interpolateFromNeighbours(signal, isOutlier, i);
                break;

            case OutlierReplaceMode::Clamp: {
                // Clamp value to be within threshold * scaledMAD of median
                float scaledMAD = localScaledMADs[i];
                float median = localMedians[i];
                if (scaledMAD > kMinMAD) {
                    float maxDev = threshold * scaledMAD;
                    float deviation = signal[i] - median;
                    if (deviation > maxDev) {
                        filtered[i] = median + maxDev;
                    } else if (deviation < -maxDev) {
                        filtered[i] = median - maxDev;
                    }
                } else {
                    filtered[i] = median;
                }
                break;
            }
        }
    }

    signal = filtered;
}

void OutlierFilter::process(std::vector<SkeletonFrame>& frames) {
    int numFrames = static_cast<int>(frames.size());
    if (numFrames < config_.minFramesRequired) {
        HM_LOG_DEBUG(TAG, "Skipping outlier filter: only " + std::to_string(numFrames) +
                     " frames (minimum " + std::to_string(config_.minFramesRequired) + ")");
        return;
    }

    HM_LOG_INFO(TAG, "Processing " + std::to_string(numFrames) + " frames with window=" +
                std::to_string(config_.windowSize) + " threshold=" +
                std::to_string(config_.madThreshold));

    // Reset statistics
    lastStats_ = {};

    int totalOutliers = 0;
    int totalSamples = 0;
    float maxDev = 0.0f;

    // Helper to process a set of 3 channels (X, Y, Z) and track stats
    auto processXYZ = [&](auto extractX, auto extractY, auto extractZ,
                          auto writeBackX, auto writeBackY, auto writeBackZ) {
        std::vector<float> xSig(numFrames), ySig(numFrames), zSig(numFrames);

        for (int f = 0; f < numFrames; ++f) {
            xSig[f] = extractX(f);
            ySig[f] = extractY(f);
            zSig[f] = extractZ(f);
        }

        // Track max deviation before filtering
        auto trackMax = [&](const std::vector<float>& sig) {
            if (sig.size() < 3) return;
            // Quick pass to find maximum inter-frame jump
            for (size_t i = 1; i < sig.size(); ++i) {
                float diff = std::abs(sig[i] - sig[i - 1]);
                if (diff > maxDev) maxDev = diff;
            }
        };
        trackMax(xSig);
        trackMax(ySig);
        trackMax(zSig);

        // Count outliers by checking before/after
        auto countOutliers = [&](const std::vector<float>& before,
                                 const std::vector<float>& after) {
            int count = 0;
            for (size_t i = 0; i < before.size(); ++i) {
                if (std::abs(before[i] - after[i]) > 1e-8f) {
                    count++;
                }
            }
            return count;
        };

        std::vector<float> xOrig = xSig, yOrig = ySig, zOrig = zSig;

        filterChannelWithMode(xSig, config_.windowSize, config_.madThreshold, config_.replaceMode);
        filterChannelWithMode(ySig, config_.windowSize, config_.madThreshold, config_.replaceMode);
        filterChannelWithMode(zSig, config_.windowSize, config_.madThreshold, config_.replaceMode);

        totalOutliers += countOutliers(xOrig, xSig);
        totalOutliers += countOutliers(yOrig, ySig);
        totalOutliers += countOutliers(zOrig, zSig);
        totalSamples += numFrames * 3;

        for (int f = 0; f < numFrames; ++f) {
            writeBackX(f, xSig[f]);
            writeBackY(f, ySig[f]);
            writeBackZ(f, zSig[f]);
        }
    };

    // -------------------------------------------------------------------
    // Process each joint's world position X, Y, Z independently
    // -------------------------------------------------------------------
    if (config_.processJointPositions) {
        for (int j = 0; j < JOINT_COUNT; ++j) {
            processXYZ(
                [&](int f) { return frames[f].joints[j].worldPosition.x; },
                [&](int f) { return frames[f].joints[j].worldPosition.y; },
                [&](int f) { return frames[f].joints[j].worldPosition.z; },
                [&](int f, float v) { frames[f].joints[j].worldPosition.x = v; },
                [&](int f, float v) { frames[f].joints[j].worldPosition.y = v; },
                [&](int f, float v) { frames[f].joints[j].worldPosition.z = v; }
            );
        }
    }

    // -------------------------------------------------------------------
    // Process 6D rotation channels per joint
    // -------------------------------------------------------------------
    if (config_.processRotation6D) {
        for (int j = 0; j < JOINT_COUNT; ++j) {
            for (int d = 0; d < ROTATION_DIM; d += 3) {
                // Process in triplets to avoid excessive overhead
                int d0 = d, d1 = std::min(d + 1, ROTATION_DIM - 1), d2 = std::min(d + 2, ROTATION_DIM - 1);
                processXYZ(
                    [&](int f) { return frames[f].joints[j].rotation6D[d0]; },
                    [&](int f) { return frames[f].joints[j].rotation6D[d1]; },
                    [&](int f) { return frames[f].joints[j].rotation6D[d2]; },
                    [&](int f, float v) { frames[f].joints[j].rotation6D[d0] = v; },
                    [&](int f, float v) { frames[f].joints[j].rotation6D[d1] = v; },
                    [&](int f, float v) { frames[f].joints[j].rotation6D[d2] = v; }
                );
            }
        }
    }

    // -------------------------------------------------------------------
    // Process root position
    // -------------------------------------------------------------------
    if (config_.processRootPosition) {
        processXYZ(
            [&](int f) { return frames[f].rootPosition.x; },
            [&](int f) { return frames[f].rootPosition.y; },
            [&](int f) { return frames[f].rootPosition.z; },
            [&](int f, float v) { frames[f].rootPosition.x = v; },
            [&](int f, float v) { frames[f].rootPosition.y = v; },
            [&](int f, float v) { frames[f].rootPosition.z = v; }
        );

        // Also process root velocity
        processXYZ(
            [&](int f) { return frames[f].rootVelocity.x; },
            [&](int f) { return frames[f].rootVelocity.y; },
            [&](int f) { return frames[f].rootVelocity.z; },
            [&](int f, float v) { frames[f].rootVelocity.x = v; },
            [&](int f, float v) { frames[f].rootVelocity.y = v; },
            [&](int f, float v) { frames[f].rootVelocity.z = v; }
        );
    }

    // Record statistics
    lastStats_.totalSamples = totalSamples;
    lastStats_.outliersDetected = totalOutliers;
    lastStats_.outliersReplaced = totalOutliers;
    lastStats_.maxDeviation = maxDev;

    if (totalOutliers > 0) {
        float pct = (static_cast<float>(totalOutliers) / static_cast<float>(totalSamples)) * 100.0f;
        HM_LOG_INFO(TAG, "Outlier filter complete: " + std::to_string(totalOutliers) +
                    " outliers replaced out of " + std::to_string(totalSamples) +
                    " samples (" + std::to_string(pct) + "%)");
    } else {
        HM_LOG_DEBUG(TAG, "Outlier filter complete: no outliers detected");
    }
}

} // namespace hm::signal
