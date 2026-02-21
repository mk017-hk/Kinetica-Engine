#include "HyperMotion/signal/QuaternionSmoother.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

namespace hm::signal {

static constexpr const char* TAG = "QuaternionSmoother";

QuaternionSmoother::QuaternionSmoother(const QuaternionSmootherConfig& config)
    : config_(config) {}

QuaternionSmoother::~QuaternionSmoother() = default;

void QuaternionSmoother::smoothQuatSequence(std::vector<Quat>& quaternions, float alpha) {
    if (quaternions.size() < 2) return;

    // Handle quaternion double-cover: ensure consistent signs
    for (size_t i = 1; i < quaternions.size(); ++i) {
        if (quaternions[i].dot(quaternions[i - 1]) < 0.0f) {
            quaternions[i] = Quat{-quaternions[i].w, -quaternions[i].x,
                                   -quaternions[i].y, -quaternions[i].z};
        }
    }

    // Forward pass: exponential moving average with SLERP
    std::vector<Quat> smoothed = quaternions;
    for (size_t i = 1; i < smoothed.size(); ++i) {
        smoothed[i] = MathUtils::safeSlerp(smoothed[i], smoothed[i - 1], alpha);
    }

    // Backward pass for symmetric smoothing
    for (int i = static_cast<int>(smoothed.size()) - 2; i >= 0; --i) {
        smoothed[i] = MathUtils::safeSlerp(smoothed[i], smoothed[i + 1], alpha);
    }

    quaternions = smoothed;
}

void QuaternionSmoother::process(std::vector<SkeletonFrame>& frames) {
    if (frames.size() < 2) return;

    int numFrames = static_cast<int>(frames.size());
    HM_LOG_DEBUG(TAG, "Smoothing quaternions for " + std::to_string(numFrames) + " frames");

    // Smooth each joint's rotation sequence
    for (int j = 0; j < JOINT_COUNT; ++j) {
        std::vector<Quat> rotSequence(numFrames);
        for (int f = 0; f < numFrames; ++f) {
            rotSequence[f] = frames[f].joints[j].localRotation;
        }

        smoothQuatSequence(rotSequence, config_.smoothingFactor);

        for (int f = 0; f < numFrames; ++f) {
            frames[f].joints[j].localRotation = rotSequence[f];
            frames[f].joints[j].rotation6D = MathUtils::quatToRot6D(rotSequence[f]);
            frames[f].joints[j].localEulerDeg = MathUtils::quatToEulerDeg(rotSequence[f]);
        }
    }

    // Smooth root rotation
    {
        std::vector<Quat> rootRots(numFrames);
        for (int f = 0; f < numFrames; ++f) {
            rootRots[f] = frames[f].rootRotation;
        }
        smoothQuatSequence(rootRots, config_.smoothingFactor);
        for (int f = 0; f < numFrames; ++f) {
            frames[f].rootRotation = rootRots[f];
        }
    }
}

} // namespace hm::signal
