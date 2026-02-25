#pragma once

#include "HyperMotion/core/Types.h"
#include <memory>
#include <vector>

namespace hm::skeleton {

struct SkeletonMapperConfig {
    float minConfidenceThreshold = 0.2f;
    bool useVelocitySmoothing = true;
    float velocitySmoothingAlpha = 0.5f;
};

class SkeletonMapper {
public:
    explicit SkeletonMapper(const SkeletonMapperConfig& config = {});
    ~SkeletonMapper();

    // Map a single detected person (COCO 17 keypoints in 3D) to a SkeletonFrame
    SkeletonFrame mapToSkeleton(const DetectedPerson& person, double timestamp, int frameIndex);

    // Map a sequence of pose results to animation frames
    std::vector<SkeletonFrame> mapSequence(const std::vector<PoseFrameResult>& poseResults,
                                            int trackingID);

    // Get rest-pose T-pose skeleton frame
    static SkeletonFrame getRestPose();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::skeleton
