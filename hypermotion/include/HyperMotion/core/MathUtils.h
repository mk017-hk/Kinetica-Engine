#pragma once

#include "HyperMotion/core/Types.h"
#include <Eigen/Dense>
#include <vector>

namespace hm {

class MathUtils {
public:
    // ---------------------------------------------------------------
    // Rotation Conversions
    // ---------------------------------------------------------------

    // Quaternion <-> Euler (degrees, XYZ intrinsic order)
    static Vec3 quatToEulerDeg(const Quat& q);
    static Quat eulerDegToQuat(const Vec3& euler);

    // Quaternion <-> Rotation Matrix (3x3)
    static Mat3 quatToMat3(const Quat& q);
    static Quat mat3ToQuat(const Mat3& m);

    // 6D Rotation Representation (Zhou et al., "On the Continuity of Rotation Representations")
    // Uses first two columns of rotation matrix, recovers third via Gram-Schmidt
    static Vec6 quatToRot6D(const Quat& q);
    static Quat rot6DToQuat(const Vec6& r6d);
    static Vec6 mat3ToRot6D(const Mat3& m);
    static Mat3 rot6DToMat3(const Vec6& r6d);

    // ---------------------------------------------------------------
    // Rotation Utilities
    // ---------------------------------------------------------------

    // Rotation that takes vector 'from' to vector 'to'
    static Quat rotationBetween(const Vec3& from, const Vec3& to);

    // Safe spherical linear interpolation
    static Quat safeSlerp(const Quat& a, const Quat& b, float t);

    // LookRotation: rotation from forward direction (Z) and up hint (Y)
    static Quat lookRotation(const Vec3& forward, const Vec3& up = {0, 1, 0});

    // Angle-axis construction
    static Quat fromAxisAngle(const Vec3& axis, float angleDeg);
    static void toAxisAngle(const Quat& q, Vec3& axis, float& angleDeg);

    // ---------------------------------------------------------------
    // Forward / Inverse Kinematics
    // ---------------------------------------------------------------

    // Forward kinematics: root + local rotations -> world positions
    static std::array<Vec3, JOINT_COUNT> forwardKinematics(
        const Vec3& rootPos,
        const Quat& rootRot,
        const std::array<Quat, JOINT_COUNT>& localRotations);

    // Inverse kinematics: world positions -> root + local rotations
    static void inverseKinematics(
        const std::array<Vec3, JOINT_COUNT>& worldPositions,
        Vec3& outRootPos,
        Quat& outRootRot,
        std::array<Quat, JOINT_COUNT>& outLocalRotations);

    // ---------------------------------------------------------------
    // Serialization: SkeletonFrame <-> flat float vector
    // ---------------------------------------------------------------

    // Frame -> 132D float vector (22 joints x 6D rotation)
    static std::vector<float> skeletonToVector(const SkeletonFrame& frame);

    // 132D float vector -> frame rotations
    static SkeletonFrame vectorToSkeleton(const std::vector<float>& vec,
                                           const SkeletonFrame& templateFrame = {});

    // ---------------------------------------------------------------
    // Clip <-> Eigen matrix [frames x 132]
    // ---------------------------------------------------------------

    static Eigen::MatrixXf clipToMatrix(const AnimClip& clip);
    static AnimClip matrixToClip(const Eigen::MatrixXf& mat,
                                  const AnimClip& templateClip = {});

    // ---------------------------------------------------------------
    // Gram-Schmidt orthonormalization for 6D rotation recovery
    // ---------------------------------------------------------------

    static Vec3 gramSchmidtColumn(const Vec3& a, const Vec3& b);

private:
    static constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
    static constexpr float kRadToDeg = 180.0f / 3.14159265358979323846f;
    static constexpr float kEpsilon = 1e-7f;
};

} // namespace hm
