#include "HyperMotion/skeleton/RotationSolver.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

#include <cmath>

namespace hm::skeleton {

static constexpr const char* TAG = "RotationSolver";
static constexpr float kEpsilon = 1e-7f;

struct RotationSolver::Impl {
    // Rest-pose bone directions (normalized) for each joint relative to parent
    std::array<Vec3, JOINT_COUNT> restDirections{};

    Impl() {
        const auto& offsets = getRestPoseBoneOffsets();
        for (int i = 0; i < JOINT_COUNT; ++i) {
            restDirections[i] = offsets[i].normalized();
        }
    }
};

RotationSolver::RotationSolver()
    : impl_(std::make_unique<Impl>()) {}

RotationSolver::~RotationSolver() = default;

void RotationSolver::solve(
    const std::array<Vec3, JOINT_COUNT>& worldPositions,
    std::array<JointTransform, JOINT_COUNT>& outJoints,
    Quat& outRootRotation) {

    const auto& offsets = getRestPoseBoneOffsets();
    std::array<Quat, JOINT_COUNT> worldRotations;

    // Compute root rotation using two direction vectors for full 3-DOF constraint:
    // 1. Spine direction (primary axis)
    // 2. Hip left-to-right direction (constrains twist around spine axis)
    Vec3 currentSpineDir = (worldPositions[static_cast<int>(Joint::Spine)] -
                            worldPositions[static_cast<int>(Joint::Hips)]).normalized();
    Vec3 restSpineDir = offsets[static_cast<int>(Joint::Spine)].normalized();

    Vec3 restHipVec = offsets[static_cast<int>(Joint::RightUpLeg)] -
                      offsets[static_cast<int>(Joint::LeftUpLeg)];
    Vec3 worldHipVec = worldPositions[static_cast<int>(Joint::RightUpLeg)] -
                       worldPositions[static_cast<int>(Joint::LeftUpLeg)];

    if (currentSpineDir.lengthSq() > kEpsilon && restSpineDir.lengthSq() > kEpsilon &&
        restHipVec.lengthSq() > kEpsilon && worldHipVec.lengthSq() > kEpsilon) {
        // Build orthonormal frames via Gram-Schmidt
        Vec3 ry = restSpineDir;
        Vec3 rx = (restHipVec - ry * ry.dot(restHipVec)).normalized();
        Vec3 rz = rx.cross(ry).normalized();

        Vec3 wy = currentSpineDir;
        Vec3 wx = (worldHipVec - wy * wy.dot(worldHipVec)).normalized();
        Vec3 wz = wx.cross(wy).normalized();

        Mat3 restFrame, worldFrame;
        restFrame.setCol(0, rx); restFrame.setCol(1, ry); restFrame.setCol(2, rz);
        worldFrame.setCol(0, wx); worldFrame.setCol(1, wy); worldFrame.setCol(2, wz);
        outRootRotation = MathUtils::mat3ToQuat(worldFrame * restFrame.transposed()).normalized();
    } else {
        outRootRotation = Quat::identity();
    }

    worldRotations[0] = outRootRotation;
    outJoints[0].localRotation = Quat::identity(); // Root local is identity; rotation stored in rootRotation

    // Solve each joint in hierarchy order
    for (int i = 1; i < JOINT_COUNT; ++i) {
        int parent = JOINT_PARENT[i];

        Vec3 boneDir = (worldPositions[i] - worldPositions[parent]);
        float boneLength = boneDir.length();

        if (boneLength < kEpsilon) {
            // Zero-length bone: inherit parent rotation
            worldRotations[i] = worldRotations[parent];
            outJoints[i].localRotation = Quat::identity();
            continue;
        }

        boneDir = boneDir / boneLength; // normalize

        // Rest-pose direction in world space (rotated by parent world rotation)
        Vec3 restDir = worldRotations[parent].rotate(impl_->restDirections[i]);

        if (restDir.lengthSq() < kEpsilon) {
            worldRotations[i] = worldRotations[parent];
            outJoints[i].localRotation = Quat::identity();
            continue;
        }

        // Check for near-parallel vectors (edge case)
        float dot = restDir.dot(boneDir);

        if (dot > 1.0f - kEpsilon) {
            // Already aligned
            worldRotations[i] = worldRotations[parent];
            outJoints[i].localRotation = Quat::identity();
        } else {
            // Compute rotation from rest to current
            Quat worldDelta = MathUtils::rotationBetween(restDir, boneDir);
            worldRotations[i] = worldDelta * worldRotations[parent];

            // Local rotation = parent_world_inverse * world_rotation
            outJoints[i].localRotation = (worldRotations[parent].conjugate() * worldRotations[i]).normalized();
        }

        outJoints[i].worldPosition = worldPositions[i];
    }

    outJoints[0].worldPosition = worldPositions[0];

    // Fill all rotation representations
    fillRotationRepresentations(outJoints);
}

Quat RotationSolver::solveJointRotation(
    const Quat& parentWorldRotation,
    const Vec3& restBoneDirection,
    const Vec3& currentBoneDirection) {

    Vec3 restDirWorld = parentWorldRotation.rotate(restBoneDirection.normalized());
    Vec3 currentDir = currentBoneDirection.normalized();

    if (restDirWorld.lengthSq() < kEpsilon || currentDir.lengthSq() < kEpsilon) {
        return Quat::identity();
    }

    Quat worldDelta = MathUtils::rotationBetween(restDirWorld, currentDir);
    Quat worldRot = worldDelta * parentWorldRotation;
    return (parentWorldRotation.conjugate() * worldRot).normalized();
}

void RotationSolver::fillRotationRepresentations(
    std::array<JointTransform, JOINT_COUNT>& joints) {

    for (int i = 0; i < JOINT_COUNT; ++i) {
        joints[i].rotation6D = MathUtils::quatToRot6D(joints[i].localRotation);
        joints[i].localEulerDeg = MathUtils::quatToEulerDeg(joints[i].localRotation);
    }
}

} // namespace hm::skeleton
