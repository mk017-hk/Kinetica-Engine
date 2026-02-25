#pragma once

#include "HyperMotion/core/Types.h"
#include <vector>

namespace hm::signal {

enum class OutlierReplaceMode {
    Median,         // Replace outlier with local median
    Interpolate,    // Linear interpolation from nearest non-outlier neighbours
    Clamp           // Clamp to median +/- threshold * scaledMAD
};

struct OutlierFilterConfig {
    int windowSize = 7;
    float madThreshold = 3.0f;
    OutlierReplaceMode replaceMode = OutlierReplaceMode::Interpolate;
    bool processRootPosition = true;
    bool processJointPositions = true;
    bool processRotation6D = true;
    int minFramesRequired = 3;   // Minimum frames to run filter
};

class OutlierFilter {
public:
    explicit OutlierFilter(const OutlierFilterConfig& config = {});
    ~OutlierFilter();

    // Process skeleton frames in-place, removing outlier joint positions
    void process(std::vector<SkeletonFrame>& frames);

    // Process a single channel (1D signal)
    static void filterChannel(std::vector<float>& signal, int windowSize, float threshold);

    // Process a single channel with replace mode control
    static void filterChannelWithMode(std::vector<float>& signal, int windowSize,
                                       float threshold, OutlierReplaceMode mode);

    // Get statistics from last processing run
    struct ProcessingStats {
        int totalSamples = 0;
        int outliersDetected = 0;
        int outliersReplaced = 0;
        float maxDeviation = 0.0f;
    };
    const ProcessingStats& lastStats() const { return lastStats_; }

private:
    OutlierFilterConfig config_;
    ProcessingStats lastStats_;

    // Compute median of a sorted range
    static float computeMedian(std::vector<float>& values);

    // Find nearest non-outlier neighbours for interpolation
    static float interpolateFromNeighbours(const std::vector<float>& signal,
                                            const std::vector<bool>& isOutlier,
                                            int index);
};

} // namespace hm::signal
