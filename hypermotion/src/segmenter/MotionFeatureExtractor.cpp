#include "HyperMotion/segmenter/MotionFeatureExtractor.h"
#include "HyperMotion/core/MathUtils.h"

#include <cmath>

namespace hm::segmenter {

MotionFeatureExtractor::MotionFeatureExtractor() = default;
MotionFeatureExtractor::~MotionFeatureExtractor() = default;

std::vector<float> MotionFeatureExtractor::extract(const SkeletonFrame& frame) {
    std::vector<float> features(FEATURE_DIM, 0.0f);
    int idx = 0;

    // 22 joints x 3 Euler angles = 66D
    for (int j = 0; j < JOINT_COUNT; ++j) {
        const auto& euler = frame.joints[j].localEulerDeg;
        features[idx++] = euler.x / 180.0f;  // Normalize to [-1, 1]
        features[idx++] = euler.y / 180.0f;
        features[idx++] = euler.z / 180.0f;
    }

    // Root velocity 3D (normalize by typical sprint speed ~800 cm/s)
    features[idx++] = frame.rootVelocity.x / 800.0f;
    features[idx++] = frame.rootVelocity.y / 800.0f;
    features[idx++] = frame.rootVelocity.z / 800.0f;

    // Root angular velocity magnitude (normalize by typical turn rate ~360 deg/s)
    float angularMag = frame.rootAngularVel.length();
    features[idx++] = angularMag / 360.0f;

    return features;
}

std::vector<std::vector<float>> MotionFeatureExtractor::extractSequence(
    const std::vector<SkeletonFrame>& frames) {
    std::vector<std::vector<float>> features;
    features.reserve(frames.size());
    for (const auto& frame : frames) {
        features.push_back(extract(frame));
    }
    return features;
}

} // namespace hm::segmenter
