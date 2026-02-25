#pragma once

#include "HyperMotion/core/Types.h"
#include <array>
#include <memory>

namespace hm::skeleton {

class RotationSolver {
public:
    RotationSolver();
    ~RotationSolver();

    // Compute local joint rotations from world positions
    // Uses direction-matching: rotation that takes rest-pose bone direction to current direction
    void solve(const std::array<Vec3, JOINT_COUNT>& worldPositions,
               std::array<JointTransform, JOINT_COUNT>& outJoints,
               Quat& outRootRotation);

    // Solve a single joint rotation given parent world rotation,
    // rest-pose direction, and current direction
    static Quat solveJointRotation(const Quat& parentWorldRotation,
                                    const Vec3& restBoneDirection,
                                    const Vec3& currentBoneDirection);

    // Convert all joint quaternions to 6D rotation and euler representations
    static void fillRotationRepresentations(std::array<JointTransform, JOINT_COUNT>& joints);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::skeleton
