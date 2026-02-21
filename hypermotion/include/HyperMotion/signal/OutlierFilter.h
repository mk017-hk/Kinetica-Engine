#pragma once

#include "HyperMotion/core/Types.h"
#include <vector>

namespace hm::signal {

struct OutlierFilterConfig {
    int windowSize = 7;
    float madThreshold = 3.0f;
};

class OutlierFilter {
public:
    explicit OutlierFilter(const OutlierFilterConfig& config = {});
    ~OutlierFilter();

    // Process skeleton frames in-place, removing outlier joint positions
    void process(std::vector<SkeletonFrame>& frames);

    // Process a single channel (1D signal)
    static void filterChannel(std::vector<float>& signal, int windowSize, float threshold);

private:
    OutlierFilterConfig config_;
};

} // namespace hm::signal
