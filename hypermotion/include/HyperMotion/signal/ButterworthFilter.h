#pragma once

#include "HyperMotion/core/Types.h"
#include <vector>
#include <array>

namespace hm::signal {

struct ButterworthConfig {
    int order = 4;
    float cutoffFreqBody = 12.0f;   // Hz, for body joints
    float cutoffFreqExtrem = 8.0f;  // Hz, for extremities (hands, feet)
    float sampleRate = 30.0f;       // Hz
};

class ButterworthFilter {
public:
    explicit ButterworthFilter(const ButterworthConfig& config = {});
    ~ButterworthFilter();

    // Process skeleton frames in-place
    void process(std::vector<SkeletonFrame>& frames);

    // Filter a single 1D signal (forward-backward for zero phase)
    static void filterChannel(std::vector<float>& signal,
                               int order, float cutoffFreq, float sampleRate);

    // Compute filter coefficients via bilinear transform
    struct FilterCoefficients {
        std::vector<float> b; // Numerator
        std::vector<float> a; // Denominator
    };
    static FilterCoefficients designLowpass(int order, float cutoffFreq, float sampleRate);

private:
    ButterworthConfig config_;

    // Joint indices that are extremities (hands, feet, toes)
    static bool isExtremity(int jointIndex);
};

} // namespace hm::signal
