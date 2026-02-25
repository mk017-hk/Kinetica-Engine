#include <gtest/gtest.h>
#include "HyperMotion/core/MathUtils.h"
#include "test_helpers.h"
#include <cmath>

using namespace hm;

// ---------------------------------------------------------------
// Euler <-> Quaternion round-trip
// ---------------------------------------------------------------

TEST(MathUtilsTest, EulerQuatRoundTrip_Identity) {
    Vec3 euler{0, 0, 0};
    Quat q = MathUtils::eulerDegToQuat(euler);
    Vec3 back = MathUtils::quatToEulerDeg(q);
    EXPECT_NEAR(back.x, 0.0f, test::kEps);
    EXPECT_NEAR(back.y, 0.0f, test::kEps);
    EXPECT_NEAR(back.z, 0.0f, test::kEps);
}

TEST(MathUtilsTest, EulerQuatRoundTrip_90X) {
    Vec3 euler{90, 0, 0};
    Quat q = MathUtils::eulerDegToQuat(euler);
    Vec3 back = MathUtils::quatToEulerDeg(q);
    EXPECT_NEAR(back.x, 90.0f, test::kLooseEps);
}

TEST(MathUtilsTest, EulerQuatRoundTrip_Combined) {
    Vec3 euler{30, 45, 60};
    Quat q = MathUtils::eulerDegToQuat(euler);
    Vec3 back = MathUtils::quatToEulerDeg(q);
    EXPECT_NEAR(back.x, euler.x, test::kLooseEps);
    EXPECT_NEAR(back.y, euler.y, test::kLooseEps);
    EXPECT_NEAR(back.z, euler.z, test::kLooseEps);
}

// ---------------------------------------------------------------
// Quaternion <-> Mat3 round-trip
// ---------------------------------------------------------------

TEST(MathUtilsTest, QuatMat3RoundTrip_Identity) {
    Quat q = Quat::identity();
    Mat3 m = MathUtils::quatToMat3(q);
    Quat back = MathUtils::mat3ToQuat(m);
    EXPECT_TRUE(test::quatNearEqual(q, back));
}

TEST(MathUtilsTest, QuatMat3RoundTrip_90Y) {
    Quat q = MathUtils::fromAxisAngle({0, 1, 0}, 90.0f);
    Mat3 m = MathUtils::quatToMat3(q);
    Quat back = MathUtils::mat3ToQuat(m);
    EXPECT_TRUE(test::quatNearEqual(q, back));
}

TEST(MathUtilsTest, RotationMatrixDeterminant) {
    Quat q = MathUtils::fromAxisAngle({1, 1, 1}, 73.0f);
    Mat3 m = MathUtils::quatToMat3(q);
    EXPECT_NEAR(m.determinant(), 1.0f, test::kEps);
}

TEST(MathUtilsTest, RotationMatrixOrthogonal) {
    Quat q = MathUtils::eulerDegToQuat({25, -40, 110});
    Mat3 m = MathUtils::quatToMat3(q);
    Mat3 mT = m.transposed();
    Mat3 product = m * mT;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            float expected = (i == j) ? 1.0f : 0.0f;
            EXPECT_NEAR(product.m[i][j], expected, test::kEps);
        }
}

// ---------------------------------------------------------------
// 6D Rotation round-trip
// ---------------------------------------------------------------

TEST(MathUtilsTest, Rot6DRoundTrip_Identity) {
    Quat q = Quat::identity();
    Vec6 r6d = MathUtils::quatToRot6D(q);
    Quat back = MathUtils::rot6DToQuat(r6d);
    EXPECT_TRUE(test::quatNearEqual(q, back));
}

TEST(MathUtilsTest, Rot6DRoundTrip_Arbitrary) {
    Quat q = MathUtils::eulerDegToQuat({15, -73, 42});
    Vec6 r6d = MathUtils::quatToRot6D(q);
    Quat back = MathUtils::rot6DToQuat(r6d);
    EXPECT_TRUE(test::quatNearEqual(q, back));
}

TEST(MathUtilsTest, Rot6DMat3RoundTrip) {
    Quat q = MathUtils::fromAxisAngle({0, 0, 1}, 45.0f);
    Mat3 m = MathUtils::quatToMat3(q);
    Vec6 r6d = MathUtils::mat3ToRot6D(m);
    Mat3 back = MathUtils::rot6DToMat3(r6d);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(back.m[i][j], m.m[i][j], test::kEps);
}

// ---------------------------------------------------------------
// RotationBetween
// ---------------------------------------------------------------

TEST(MathUtilsTest, RotationBetween_SameVector) {
    Vec3 v{1, 0, 0};
    Quat q = MathUtils::rotationBetween(v, v);
    EXPECT_TRUE(test::quatNearEqual(q, Quat::identity()));
}

TEST(MathUtilsTest, RotationBetween_OrthogonalVectors) {
    Vec3 from{1, 0, 0}, to{0, 1, 0};
    Quat q = MathUtils::rotationBetween(from, to);
    Vec3 rotated = q.rotate(from);
    EXPECT_NEAR(rotated.x, to.x, test::kEps);
    EXPECT_NEAR(rotated.y, to.y, test::kEps);
    EXPECT_NEAR(rotated.z, to.z, test::kEps);
}

TEST(MathUtilsTest, RotationBetween_OppositeVectors) {
    Vec3 from{0, 1, 0}, to{0, -1, 0};
    Quat q = MathUtils::rotationBetween(from, to);
    Vec3 rotated = q.rotate(from);
    EXPECT_NEAR(rotated.x, to.x, test::kEps);
    EXPECT_NEAR(rotated.y, to.y, test::kEps);
    EXPECT_NEAR(rotated.z, to.z, test::kEps);
}

TEST(MathUtilsTest, RotationBetween_Arbitrary) {
    Vec3 from{1, 2, 3}, to{-3, 1, 2};
    Quat q = MathUtils::rotationBetween(from, to);
    Vec3 rotated = q.rotate(from.normalized());
    Vec3 expected = to.normalized();
    EXPECT_NEAR(rotated.x, expected.x, test::kEps);
    EXPECT_NEAR(rotated.y, expected.y, test::kEps);
    EXPECT_NEAR(rotated.z, expected.z, test::kEps);
}

// ---------------------------------------------------------------
// SafeSlerp
// ---------------------------------------------------------------

TEST(MathUtilsTest, Slerp_t0) {
    Quat a = Quat::identity();
    Quat b = MathUtils::fromAxisAngle({0, 1, 0}, 90.0f);
    Quat r = MathUtils::safeSlerp(a, b, 0.0f);
    EXPECT_TRUE(test::quatNearEqual(r, a));
}

TEST(MathUtilsTest, Slerp_t1) {
    Quat a = Quat::identity();
    Quat b = MathUtils::fromAxisAngle({0, 1, 0}, 90.0f);
    Quat r = MathUtils::safeSlerp(a, b, 1.0f);
    EXPECT_TRUE(test::quatNearEqual(r, b));
}

TEST(MathUtilsTest, Slerp_tHalf) {
    Quat a = Quat::identity();
    Quat b = MathUtils::fromAxisAngle({0, 1, 0}, 90.0f);
    Quat r = MathUtils::safeSlerp(a, b, 0.5f);
    Quat expected = MathUtils::fromAxisAngle({0, 1, 0}, 45.0f);
    EXPECT_TRUE(test::quatNearEqual(r, expected));
}

TEST(MathUtilsTest, Slerp_NegativeDotHandling) {
    // b is negative hemisphere from a; should still give shortest path
    Quat a = Quat::identity();
    Quat b = MathUtils::fromAxisAngle({0, 1, 0}, 10.0f);
    Quat bNeg = {-b.w, -b.x, -b.y, -b.z};
    Quat r1 = MathUtils::safeSlerp(a, b, 0.5f);
    Quat r2 = MathUtils::safeSlerp(a, bNeg, 0.5f);
    EXPECT_TRUE(test::quatNearEqual(r1, r2));
}

// ---------------------------------------------------------------
// AxisAngle
// ---------------------------------------------------------------

TEST(MathUtilsTest, AxisAngle_RoundTrip) {
    Vec3 axis{0, 1, 0};
    float angle = 60.0f;
    Quat q = MathUtils::fromAxisAngle(axis, angle);

    Vec3 outAxis;
    float outAngle;
    MathUtils::toAxisAngle(q, outAxis, outAngle);
    EXPECT_NEAR(outAngle, angle, test::kLooseEps);
    EXPECT_NEAR(std::abs(outAxis.dot(axis)), 1.0f, test::kEps);
}

TEST(MathUtilsTest, AxisAngle_ZeroAngle) {
    Quat q = MathUtils::fromAxisAngle({1, 0, 0}, 0.0f);
    EXPECT_TRUE(test::quatNearEqual(q, Quat::identity()));
}

// ---------------------------------------------------------------
// LookRotation
// ---------------------------------------------------------------

TEST(MathUtilsTest, LookRotation_Forward) {
    Quat q = MathUtils::lookRotation({0, 0, 1});
    Vec3 result = q.rotate({0, 0, 1});
    EXPECT_NEAR(result.z, 1.0f, test::kEps);
}

// ---------------------------------------------------------------
// Forward / Inverse Kinematics
// ---------------------------------------------------------------

TEST(MathUtilsTest, FK_IdentityRotations) {
    std::array<Quat, JOINT_COUNT> localRots;
    for (auto& q : localRots) q = Quat::identity();
    Vec3 rootPos{0, 90, 0};

    auto worldPos = MathUtils::forwardKinematics(rootPos, Quat::identity(), localRots);

    // Root should be at rootPos
    EXPECT_NEAR(worldPos[0].x, 0.0f, test::kEps);
    EXPECT_NEAR(worldPos[0].y, 90.0f, test::kEps);

    // Spine should be above hips (rest offset is {0, 10, 0})
    EXPECT_NEAR(worldPos[1].y, 100.0f, test::kEps);
}

TEST(MathUtilsTest, FK_IK_RoundTrip) {
    // Create a pose with some rotations
    std::array<Quat, JOINT_COUNT> localRots;
    for (auto& q : localRots) q = Quat::identity();
    localRots[1] = MathUtils::fromAxisAngle({0, 0, 1}, 15.0f); // Spine bend

    Vec3 rootPos{10, 90, 5};
    Quat rootRot = MathUtils::fromAxisAngle({0, 1, 0}, 30.0f);

    auto worldPos = MathUtils::forwardKinematics(rootPos, rootRot, localRots);

    // Now recover via IK
    Vec3 recoveredRoot;
    Quat recoveredRootRot;
    std::array<Quat, JOINT_COUNT> recoveredLocal;
    MathUtils::inverseKinematics(worldPos, recoveredRoot, recoveredRootRot, recoveredLocal);

    // Root position should match exactly
    EXPECT_NEAR(recoveredRoot.x, rootPos.x, test::kEps);
    EXPECT_NEAR(recoveredRoot.y, rootPos.y, test::kEps);
    EXPECT_NEAR(recoveredRoot.z, rootPos.z, test::kEps);

    // Verify FK with recovered values produces same world positions
    auto recomputedPos = MathUtils::forwardKinematics(
        recoveredRoot, recoveredRootRot, recoveredLocal);
    for (int i = 0; i < JOINT_COUNT; ++i) {
        EXPECT_NEAR(recomputedPos[i].x, worldPos[i].x, test::kLooseEps)
            << "Joint " << i << " X mismatch";
        EXPECT_NEAR(recomputedPos[i].y, worldPos[i].y, test::kLooseEps)
            << "Joint " << i << " Y mismatch";
        EXPECT_NEAR(recomputedPos[i].z, worldPos[i].z, test::kLooseEps)
            << "Joint " << i << " Z mismatch";
    }
}

// ---------------------------------------------------------------
// Skeleton <-> Vector serialization
// ---------------------------------------------------------------

TEST(MathUtilsTest, SkeletonVectorRoundTrip) {
    SkeletonFrame frame;
    for (int i = 0; i < JOINT_COUNT; ++i) {
        Quat q = MathUtils::fromAxisAngle({0, 1, 0}, static_cast<float>(i) * 5.0f);
        frame.joints[i].rotation6D = MathUtils::quatToRot6D(q);
        frame.joints[i].localRotation = q;
    }

    auto vec = MathUtils::skeletonToVector(frame);
    EXPECT_EQ(vec.size(), static_cast<size_t>(FRAME_DIM));

    auto recovered = MathUtils::vectorToSkeleton(vec);
    for (int i = 0; i < JOINT_COUNT; ++i) {
        for (int d = 0; d < ROTATION_DIM; ++d) {
            EXPECT_NEAR(recovered.joints[i].rotation6D[d],
                        frame.joints[i].rotation6D[d], test::kEps)
                << "Joint " << i << " dim " << d;
        }
    }
}

// ---------------------------------------------------------------
// Clip <-> Matrix serialization
// ---------------------------------------------------------------

TEST(MathUtilsTest, ClipMatrixRoundTrip) {
    AnimClip clip;
    clip.fps = 30.0f;
    clip.frames.resize(5);
    for (int f = 0; f < 5; ++f) {
        clip.frames[f].frameIndex = f;
        for (int j = 0; j < JOINT_COUNT; ++j) {
            Quat q = MathUtils::fromAxisAngle({1, 0, 0},
                static_cast<float>(f * 10 + j));
            clip.frames[f].joints[j].rotation6D = MathUtils::quatToRot6D(q);
        }
    }

    auto mat = MathUtils::clipToMatrix(clip);
    EXPECT_EQ(mat.rows(), 5);
    EXPECT_EQ(mat.cols(), FRAME_DIM);

    auto recovered = MathUtils::matrixToClip(mat, clip);
    EXPECT_EQ(recovered.frames.size(), 5u);

    for (int f = 0; f < 5; ++f) {
        for (int j = 0; j < JOINT_COUNT; ++j) {
            for (int d = 0; d < ROTATION_DIM; ++d) {
                EXPECT_NEAR(recovered.frames[f].joints[j].rotation6D[d],
                            clip.frames[f].joints[j].rotation6D[d], test::kEps);
            }
        }
    }
}

// ---------------------------------------------------------------
// GramSchmidt
// ---------------------------------------------------------------

TEST(MathUtilsTest, GramSchmidt_Orthogonalize) {
    Vec3 a{1, 0, 0};
    Vec3 b{1, 1, 0};
    Vec3 result = MathUtils::gramSchmidtColumn(a, b);
    // result should be orthogonal to a
    EXPECT_NEAR(a.dot(result), 0.0f, test::kEps);
    // result should be unit length
    EXPECT_NEAR(result.length(), 1.0f, test::kEps);
}
