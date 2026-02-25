#pragma once

#include "HyperMotion/core/Types.h"
#include <vector>
#include <array>

namespace hm::segmenter {

/// Extracts 70D feature vectors from SkeletonFrames for motion classification.
///
/// Feature breakdown (70D):
///   - 22 joints x 3 Euler angles (normalized) = 66D
///   - Root velocity 3D (normalized) = 3D
///   - Root angular velocity magnitude = 1D
///
/// Also supports extended features and heuristic classification fallback.
class MotionFeatureExtractor {
public:
    static constexpr int FEATURE_DIM = 70;
    static constexpr int EXTENDED_FEATURE_DIM = 140;  // base + delta features

    MotionFeatureExtractor();
    ~MotionFeatureExtractor();

    /// Extract 70D feature vector from a single SkeletonFrame.
    std::vector<float> extract(const SkeletonFrame& frame);

    /// Extract features for a sequence of frames.
    std::vector<std::vector<float>> extractSequence(const std::vector<SkeletonFrame>& frames);

    /// Extract extended 140D features (base + delta) for a sequence.
    /// Delta features capture frame-to-frame differences.
    std::vector<std::vector<float>> extractSequenceExtended(
        const std::vector<SkeletonFrame>& frames);

    /// Body-part summary statistics for a window of frames.
    struct BodyPartStats {
        float torsoVelocity = 0.0f;
        float leftArmVelocity = 0.0f;
        float rightArmVelocity = 0.0f;
        float leftLegVelocity = 0.0f;
        float rightLegVelocity = 0.0f;
        float torsoROM = 0.0f;
        float leftArmROM = 0.0f;
        float rightArmROM = 0.0f;
        float leftLegROM = 0.0f;
        float rightLegROM = 0.0f;
    };

    BodyPartStats computeBodyPartStats(const std::vector<SkeletonFrame>& frames,
                                        int startFrame, int endFrame);

    /// Heuristic motion type classification when no TCN model is available.
    MotionType classifyHeuristic(const std::vector<SkeletonFrame>& frames,
                                  int startFrame, int endFrame);

private:
    /// Compute angular velocity of a joint group across frames.
    float computeJointGroupVelocity(const std::vector<SkeletonFrame>& frames,
                                     const int* jointIndices, int numJoints,
                                     int startFrame, int endFrame);

    /// Compute range of motion (max-min Euler angle spread) for a joint group.
    float computeJointGroupROM(const std::vector<SkeletonFrame>& frames,
                                const int* jointIndices, int numJoints,
                                int startFrame, int endFrame);
};

} // namespace hm::segmenter
