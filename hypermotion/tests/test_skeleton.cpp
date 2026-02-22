#include <gtest/gtest.h>
#include "HyperMotion/skeleton/RotationSolver.h"
#include "HyperMotion/core/MathUtils.h"
#include "test_helpers.h"
#include <cmath>

using namespace hm;
using namespace hm::skeleton;

// ---------------------------------------------------------------
// RotationSolver::solveJointRotation
// ---------------------------------------------------------------

TEST(RotationSolverTest, SolveJointRotation_SameDirection) {
    // If rest and current directions are the same, should return identity
    Quat parentRot = Quat::identity();
    Vec3 restDir{0, 1, 0};
    Vec3 currentDir{0, 1, 0};

    Quat result = RotationSolver::solveJointRotation(parentRot, restDir, currentDir);
    EXPECT_TRUE(test::quatNearEqual(result, Quat::identity(), test::kLooseEps));
}

TEST(RotationSolverTest, SolveJointRotation_90DegBend) {
    // Rest: pointing up, Current: pointing forward
    Quat parentRot = Quat::identity();
    Vec3 restDir{0, 1, 0};
    Vec3 currentDir{0, 0, 1};

    Quat result = RotationSolver::solveJointRotation(parentRot, restDir, currentDir);

    // Applying the local rotation to the rest direction (rotated by parent)
    // should give us the current direction
    Vec3 rotatedRest = (parentRot * result).rotate(restDir);
    Vec3 expectedDir = currentDir.normalized();
    EXPECT_NEAR(rotatedRest.x, expectedDir.x, test::kLooseEps);
    EXPECT_NEAR(rotatedRest.y, expectedDir.y, test::kLooseEps);
    EXPECT_NEAR(rotatedRest.z, expectedDir.z, test::kLooseEps);
}

TEST(RotationSolverTest, SolveJointRotation_WithParentRotation) {
    // Parent is rotated 45 degrees around Y
    Quat parentRot = MathUtils::fromAxisAngle({0, 1, 0}, 45.0f);
    Vec3 restDir{1, 0, 0};
    Vec3 currentDir{0, 0, -1};

    Quat result = RotationSolver::solveJointRotation(parentRot, restDir, currentDir);

    // The combined rotation (parent * local) applied to rest should give current
    Vec3 rotatedRest = (parentRot * result).rotate(restDir);
    Vec3 expectedDir = currentDir.normalized();
    EXPECT_NEAR(rotatedRest.x, expectedDir.x, test::kLooseEps);
    EXPECT_NEAR(rotatedRest.y, expectedDir.y, test::kLooseEps);
    EXPECT_NEAR(rotatedRest.z, expectedDir.z, test::kLooseEps);
}

// ---------------------------------------------------------------
// RotationSolver::fillRotationRepresentations
// ---------------------------------------------------------------

TEST(RotationSolverTest, FillRotationRepresentations) {
    std::array<JointTransform, JOINT_COUNT> joints;
    for (int i = 0; i < JOINT_COUNT; ++i) {
        joints[i].localRotation = MathUtils::fromAxisAngle(
            {0, 1, 0}, static_cast<float>(i) * 10.0f);
    }

    RotationSolver::fillRotationRepresentations(joints);

    for (int i = 0; i < JOINT_COUNT; ++i) {
        // 6D should be a valid rotation representation
        Quat recovered = MathUtils::rot6DToQuat(joints[i].rotation6D);
        EXPECT_TRUE(test::quatNearEqual(recovered, joints[i].localRotation, test::kLooseEps))
            << "Joint " << i << " 6D -> Quat mismatch";

        // Euler should also round-trip
        Quat fromEuler = MathUtils::eulerDegToQuat(joints[i].localEulerDeg);
        EXPECT_TRUE(test::quatNearEqual(fromEuler, joints[i].localRotation, test::kLooseEps))
            << "Joint " << i << " Euler -> Quat mismatch";
    }
}

// ---------------------------------------------------------------
// RotationSolver::solve (full skeleton)
// ---------------------------------------------------------------

TEST(RotationSolverTest, SolveFullSkeleton) {
    // Create T-pose world positions via FK with identity rotations
    std::array<Quat, JOINT_COUNT> identityRots;
    for (auto& q : identityRots) q = Quat::identity();
    Vec3 rootPos{0, 90, 0};
    auto worldPos = MathUtils::forwardKinematics(rootPos, Quat::identity(), identityRots);

    RotationSolver solver;
    std::array<JointTransform, JOINT_COUNT> joints;
    Quat rootRot;
    solver.solve(worldPos, joints, rootRot);

    // For T-pose, all local rotations should be near identity
    for (int i = 0; i < JOINT_COUNT; ++i) {
        EXPECT_TRUE(test::quatNearEqual(joints[i].localRotation, Quat::identity(), 0.1f))
            << "Joint " << JOINT_NAMES[i] << " should be near identity for T-pose";
    }
}

TEST(RotationSolverTest, SolvePreservesWorldPositions) {
    // Create a non-trivial pose
    std::array<Quat, JOINT_COUNT> localRots;
    for (auto& q : localRots) q = Quat::identity();
    localRots[static_cast<int>(Joint::LeftArm)] =
        MathUtils::fromAxisAngle({0, 0, 1}, -45.0f);
    localRots[static_cast<int>(Joint::RightArm)] =
        MathUtils::fromAxisAngle({0, 0, 1}, 45.0f);

    Vec3 rootPos{0, 90, 0};
    auto worldPos = MathUtils::forwardKinematics(rootPos, Quat::identity(), localRots);

    // Solve rotations from world positions
    RotationSolver solver;
    std::array<JointTransform, JOINT_COUNT> joints;
    Quat rootRot;
    solver.solve(worldPos, joints, rootRot);

    // Recover world positions from solved rotations
    std::array<Quat, JOINT_COUNT> solvedRots;
    for (int i = 0; i < JOINT_COUNT; ++i)
        solvedRots[i] = joints[i].localRotation;

    auto recovered = MathUtils::forwardKinematics(rootPos, rootRot, solvedRots);

    // Recovered positions should be close to originals
    for (int i = 0; i < JOINT_COUNT; ++i) {
        EXPECT_NEAR(recovered[i].x, worldPos[i].x, 1.0f)
            << "Joint " << JOINT_NAMES[i] << " X";
        EXPECT_NEAR(recovered[i].y, worldPos[i].y, 1.0f)
            << "Joint " << JOINT_NAMES[i] << " Y";
        EXPECT_NEAR(recovered[i].z, worldPos[i].z, 1.0f)
            << "Joint " << JOINT_NAMES[i] << " Z";
    }
}

// ---------------------------------------------------------------
// Rest pose bone offsets
// ---------------------------------------------------------------

TEST(SkeletonTest, RestPoseBoneOffsets_CorrectCount) {
    const auto& offsets = getRestPoseBoneOffsets();
    EXPECT_EQ(offsets.size(), static_cast<size_t>(JOINT_COUNT));
}

TEST(SkeletonTest, RestPoseBoneOffsets_RootIsZero) {
    const auto& offsets = getRestPoseBoneOffsets();
    EXPECT_FLOAT_EQ(offsets[0].x, 0.0f);
    EXPECT_FLOAT_EQ(offsets[0].y, 0.0f);
    EXPECT_FLOAT_EQ(offsets[0].z, 0.0f);
}

TEST(SkeletonTest, RestPoseBoneOffsets_SymmetricArms) {
    const auto& offsets = getRestPoseBoneOffsets();
    // LeftShoulder(6) and RightShoulder(10) should be mirrored in X
    EXPECT_NEAR(offsets[6].x, -offsets[10].x, test::kEps);
    EXPECT_NEAR(offsets[6].y, offsets[10].y, test::kEps);
}

TEST(SkeletonTest, JointHierarchy_HipsIsRoot) {
    EXPECT_EQ(JOINT_PARENT[0], -1);
}

TEST(SkeletonTest, JointHierarchy_SpineParentIsHips) {
    EXPECT_EQ(JOINT_PARENT[1], 0);
}

TEST(SkeletonTest, JointHierarchy_AllParentsValid) {
    for (int i = 0; i < JOINT_COUNT; ++i) {
        if (i == 0) {
            EXPECT_EQ(JOINT_PARENT[i], -1);
        } else {
            EXPECT_GE(JOINT_PARENT[i], 0);
            EXPECT_LT(JOINT_PARENT[i], i); // Parent should come before child
        }
    }
}
