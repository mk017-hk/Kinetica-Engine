#pragma once

#include "HyperMotion/core/Types.h"
#include <vector>
#include <array>

namespace hm::signal {

struct QuaternionSmootherConfig {
    float smoothingFactor = 0.3f;

    // Per-joint smoothing factor overrides (0 = use default, otherwise this value is used)
    // Allows stronger smoothing on noisy joints (e.g. extremities)
    std::array<float, JOINT_COUNT> perJointSmoothing{};

    // Temporal window for multi-frame averaging (1 = single-pass EMA, >1 = windowed)
    int temporalWindow = 1;

    // Root rotation smoothing factor (often needs different amount from joints)
    float rootSmoothingFactor = 0.3f;

    // Maximum allowed angular velocity (deg/frame) before clamping.
    // Helps prevent physically implausible rapid rotations.
    // 0 = disabled.
    float maxAngularVelocityDegPerFrame = 0.0f;

    // If true, rebuild 6D rotation and Euler angles from smoothed quaternions
    bool updateDerivedRotations = true;
};

class QuaternionSmoother {
public:
    explicit QuaternionSmoother(const QuaternionSmootherConfig& config = {});
    ~QuaternionSmoother();

    // Process skeleton frames in-place with SLERP-based smoothing
    void process(std::vector<SkeletonFrame>& frames);

    // Smooth a single quaternion sequence
    static void smoothQuatSequence(std::vector<Quat>& quaternions, float alpha);

private:
    QuaternionSmootherConfig config_;

    // Get the effective smoothing factor for a given joint
    float getJointSmoothing(int jointIndex) const;

    // Fix double-cover issues across the entire sequence (ensure consistent hemisphere)
    static void fixDoubleCover(std::vector<Quat>& quaternions);

    // Enforce angular velocity limits between consecutive frames
    static void clampAngularVelocity(std::vector<Quat>& quaternions, float maxDegPerFrame);

    // Apply windowed quaternion averaging using iterative SLERP
    static void windowedSmoothing(std::vector<Quat>& quaternions, int windowSize, float alpha);

    // Compute angular distance between two quaternions in degrees
    static float angularDistanceDeg(const Quat& a, const Quat& b);

    // Normalise quaternion, handling near-zero norm gracefully
    static Quat safeNormalize(const Quat& q);
};

} // namespace hm::signal
