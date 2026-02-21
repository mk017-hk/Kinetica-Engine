#pragma once

#include "HyperMotion/core/Types.h"
#include <vector>

namespace hm::segmenter {

class MotionFeatureExtractor {
public:
    static constexpr int FEATURE_DIM = 70;

    MotionFeatureExtractor();
    ~MotionFeatureExtractor();

    // Extract 70D feature vector from a SkeletonFrame
    // 22 joints x 3 (Euler XYZ) = 66D + root velocity 3D + angular velocity 1D
    std::vector<float> extract(const SkeletonFrame& frame);

    // Extract features for a sequence of frames
    std::vector<std::vector<float>> extractSequence(const std::vector<SkeletonFrame>& frames);
};

} // namespace hm::segmenter
