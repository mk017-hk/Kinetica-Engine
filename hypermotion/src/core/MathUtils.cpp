#include "HyperMotion/core/MathUtils.h"

#include <cmath>
#include <algorithm>

namespace hm {

// ---------------------------------------------------------------
// Quaternion <-> Euler (degrees, XYZ intrinsic order)
// ---------------------------------------------------------------

Vec3 MathUtils::quatToEulerDeg(const Quat& q) {
    // XYZ intrinsic (equivalent to ZYX extrinsic)
    float sinr_cosp = 2.0f * (q.w * q.x + q.y * q.z);
    float cosr_cosp = 1.0f - 2.0f * (q.x * q.x + q.y * q.y);
    float roll = std::atan2(sinr_cosp, cosr_cosp);

    float sinp = 2.0f * (q.w * q.y - q.z * q.x);
    float pitch;
    if (std::abs(sinp) >= 1.0f)
        pitch = std::copysign(3.14159265358979323846f / 2.0f, sinp);
    else
        pitch = std::asin(sinp);

    float siny_cosp = 2.0f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
    float yaw = std::atan2(siny_cosp, cosy_cosp);

    return {roll * kRadToDeg, pitch * kRadToDeg, yaw * kRadToDeg};
}

Quat MathUtils::eulerDegToQuat(const Vec3& euler) {
    float rx = euler.x * kDegToRad * 0.5f;
    float ry = euler.y * kDegToRad * 0.5f;
    float rz = euler.z * kDegToRad * 0.5f;

    float cx = std::cos(rx), sx = std::sin(rx);
    float cy = std::cos(ry), sy = std::sin(ry);
    float cz = std::cos(rz), sz = std::sin(rz);

    return Quat{
        cx * cy * cz + sx * sy * sz,
        sx * cy * cz - cx * sy * sz,
        cx * sy * cz + sx * cy * sz,
        cx * cy * sz - sx * sy * cz
    }.normalized();
}

// ---------------------------------------------------------------
// Quaternion <-> Rotation Matrix
// ---------------------------------------------------------------

Mat3 MathUtils::quatToMat3(const Quat& q) {
    float xx = q.x * q.x, yy = q.y * q.y, zz = q.z * q.z;
    float xy = q.x * q.y, xz = q.x * q.z, yz = q.y * q.z;
    float wx = q.w * q.x, wy = q.w * q.y, wz = q.w * q.z;

    Mat3 m;
    m.m[0][0] = 1.0f - 2.0f * (yy + zz);
    m.m[0][1] = 2.0f * (xy - wz);
    m.m[0][2] = 2.0f * (xz + wy);
    m.m[1][0] = 2.0f * (xy + wz);
    m.m[1][1] = 1.0f - 2.0f * (xx + zz);
    m.m[1][2] = 2.0f * (yz - wx);
    m.m[2][0] = 2.0f * (xz - wy);
    m.m[2][1] = 2.0f * (yz + wx);
    m.m[2][2] = 1.0f - 2.0f * (xx + yy);
    return m;
}

Quat MathUtils::mat3ToQuat(const Mat3& m) {
    float trace = m.m[0][0] + m.m[1][1] + m.m[2][2];
    Quat q;

    if (trace > 0.0f) {
        float s = 0.5f / std::sqrt(trace + 1.0f);
        q.w = 0.25f / s;
        q.x = (m.m[2][1] - m.m[1][2]) * s;
        q.y = (m.m[0][2] - m.m[2][0]) * s;
        q.z = (m.m[1][0] - m.m[0][1]) * s;
    } else if (m.m[0][0] > m.m[1][1] && m.m[0][0] > m.m[2][2]) {
        float s = 2.0f * std::sqrt(1.0f + m.m[0][0] - m.m[1][1] - m.m[2][2]);
        q.w = (m.m[2][1] - m.m[1][2]) / s;
        q.x = 0.25f * s;
        q.y = (m.m[0][1] + m.m[1][0]) / s;
        q.z = (m.m[0][2] + m.m[2][0]) / s;
    } else if (m.m[1][1] > m.m[2][2]) {
        float s = 2.0f * std::sqrt(1.0f + m.m[1][1] - m.m[0][0] - m.m[2][2]);
        q.w = (m.m[0][2] - m.m[2][0]) / s;
        q.x = (m.m[0][1] + m.m[1][0]) / s;
        q.y = 0.25f * s;
        q.z = (m.m[1][2] + m.m[2][1]) / s;
    } else {
        float s = 2.0f * std::sqrt(1.0f + m.m[2][2] - m.m[0][0] - m.m[1][1]);
        q.w = (m.m[1][0] - m.m[0][1]) / s;
        q.x = (m.m[0][2] + m.m[2][0]) / s;
        q.y = (m.m[1][2] + m.m[2][1]) / s;
        q.z = 0.25f * s;
    }

    return q.normalized();
}

// ---------------------------------------------------------------
// 6D Rotation (Zhou et al.)
// ---------------------------------------------------------------

Vec6 MathUtils::mat3ToRot6D(const Mat3& m) {
    // First two columns of rotation matrix
    return {m.m[0][0], m.m[1][0], m.m[2][0],
            m.m[0][1], m.m[1][1], m.m[2][1]};
}

Mat3 MathUtils::rot6DToMat3(const Vec6& r6d) {
    // Recover rotation matrix from first two columns via Gram-Schmidt
    Vec3 a1{r6d[0], r6d[1], r6d[2]};
    Vec3 a2{r6d[3], r6d[4], r6d[5]};

    // Normalize first column
    Vec3 b1 = a1.normalized();

    // Second column: remove projection onto b1, normalize
    Vec3 b2 = gramSchmidtColumn(b1, a2);

    // Third column: cross product
    Vec3 b3 = b1.cross(b2);

    Mat3 m;
    m.setCol(0, b1);
    m.setCol(1, b2);
    m.setCol(2, b3);
    return m;
}

Vec6 MathUtils::quatToRot6D(const Quat& q) {
    return mat3ToRot6D(quatToMat3(q));
}

Quat MathUtils::rot6DToQuat(const Vec6& r6d) {
    return mat3ToQuat(rot6DToMat3(r6d));
}

Vec3 MathUtils::gramSchmidtColumn(const Vec3& a, const Vec3& b) {
    float dot = a.dot(b);
    Vec3 proj = a * dot;
    Vec3 result = b - proj;
    return result.normalized();
}

// ---------------------------------------------------------------
// Rotation Utilities
// ---------------------------------------------------------------

Quat MathUtils::rotationBetween(const Vec3& from, const Vec3& to) {
    Vec3 f = from.normalized();
    Vec3 t = to.normalized();

    float dot = f.dot(t);

    if (dot > 1.0f - kEpsilon) {
        return Quat::identity();
    }

    if (dot < -1.0f + kEpsilon) {
        // 180 degree rotation: find perpendicular axis
        Vec3 axis = Vec3{1, 0, 0}.cross(f);
        if (axis.lengthSq() < kEpsilon)
            axis = Vec3{0, 1, 0}.cross(f);
        axis = axis.normalized();
        return Quat{0.0f, axis.x, axis.y, axis.z};
    }

    Vec3 axis = f.cross(t);
    float w = 1.0f + dot;
    return Quat{w, axis.x, axis.y, axis.z}.normalized();
}

Quat MathUtils::safeSlerp(const Quat& a, const Quat& b, float t) {
    Quat bAdj = b;
    float dot = a.dot(b);

    // Ensure shortest path
    if (dot < 0.0f) {
        bAdj = Quat{-b.w, -b.x, -b.y, -b.z};
        dot = -dot;
    }

    if (dot > 0.9995f) {
        // Linear interpolation for very close quaternions
        Quat result{
            a.w + t * (bAdj.w - a.w),
            a.x + t * (bAdj.x - a.x),
            a.y + t * (bAdj.y - a.y),
            a.z + t * (bAdj.z - a.z)
        };
        return result.normalized();
    }

    float theta = std::acos(std::clamp(dot, -1.0f, 1.0f));
    float sinTheta = std::sin(theta);
    float wa = std::sin((1.0f - t) * theta) / sinTheta;
    float wb = std::sin(t * theta) / sinTheta;

    return Quat{
        wa * a.w + wb * bAdj.w,
        wa * a.x + wb * bAdj.x,
        wa * a.y + wb * bAdj.y,
        wa * a.z + wb * bAdj.z
    }.normalized();
}

Quat MathUtils::lookRotation(const Vec3& forward, const Vec3& up) {
    Vec3 f = forward.normalized();
    Vec3 r = up.cross(f).normalized();
    Vec3 u = f.cross(r);

    Mat3 m;
    m.setCol(0, r);
    m.setCol(1, u);
    m.setCol(2, f);
    return mat3ToQuat(m);
}

Quat MathUtils::fromAxisAngle(const Vec3& axis, float angleDeg) {
    float half = angleDeg * kDegToRad * 0.5f;
    float s = std::sin(half);
    Vec3 a = axis.normalized();
    return Quat{std::cos(half), a.x * s, a.y * s, a.z * s}.normalized();
}

void MathUtils::toAxisAngle(const Quat& q, Vec3& axis, float& angleDeg) {
    Quat qn = q.normalized();
    float halfAngle = std::acos(std::clamp(qn.w, -1.0f, 1.0f));
    float sinHalf = std::sin(halfAngle);

    angleDeg = halfAngle * 2.0f * kRadToDeg;

    if (sinHalf > kEpsilon) {
        axis = Vec3{qn.x / sinHalf, qn.y / sinHalf, qn.z / sinHalf};
    } else {
        axis = {1.0f, 0.0f, 0.0f};
        angleDeg = 0.0f;
    }
}

// ---------------------------------------------------------------
// Forward Kinematics
// ---------------------------------------------------------------

std::array<Vec3, JOINT_COUNT> MathUtils::forwardKinematics(
    const Vec3& rootPos,
    const Quat& rootRot,
    const std::array<Quat, JOINT_COUNT>& localRotations)
{
    const auto& offsets = getRestPoseBoneOffsets();
    std::array<Vec3, JOINT_COUNT> worldPos;
    std::array<Quat, JOINT_COUNT> worldRot;

    for (int i = 0; i < JOINT_COUNT; ++i) {
        int parent = JOINT_PARENT[i];
        if (parent < 0) {
            // Root joint
            worldRot[i] = rootRot * localRotations[i];
            worldPos[i] = rootPos;
        } else {
            worldRot[i] = worldRot[parent] * localRotations[i];
            Vec3 rotatedOffset = worldRot[parent].rotate(offsets[i]);
            worldPos[i] = worldPos[parent] + rotatedOffset;
        }
    }

    return worldPos;
}

// ---------------------------------------------------------------
// Inverse Kinematics
// ---------------------------------------------------------------

void MathUtils::inverseKinematics(
    const std::array<Vec3, JOINT_COUNT>& worldPositions,
    Vec3& outRootPos,
    Quat& outRootRot,
    std::array<Quat, JOINT_COUNT>& outLocalRotations)
{
    const auto& offsets = getRestPoseBoneOffsets();

    outRootPos = worldPositions[0]; // Hips position

    // Compute root rotation from hip-to-spine direction
    Vec3 spineDir = (worldPositions[static_cast<int>(Joint::Spine)] - worldPositions[0]).normalized();
    Vec3 restSpineDir = offsets[static_cast<int>(Joint::Spine)].normalized();
    outRootRot = rotationBetween(restSpineDir, spineDir);

    // Compute world rotations for each joint
    std::array<Quat, JOINT_COUNT> worldRot;
    worldRot[0] = outRootRot;

    for (int i = 0; i < JOINT_COUNT; ++i) {
        int parent = JOINT_PARENT[i];
        if (parent < 0) {
            // Root: already computed
            continue;
        }

        // Current bone direction in world space
        Vec3 boneDir = (worldPositions[i] - worldPositions[parent]).normalized();

        // Rest-pose bone direction in world space (rotated by parent world rotation)
        Vec3 restDir = worldRot[parent].rotate(offsets[i].normalized());

        if (boneDir.lengthSq() < kEpsilon || restDir.lengthSq() < kEpsilon) {
            worldRot[i] = worldRot[parent];
            outLocalRotations[i] = Quat::identity();
        } else {
            // Rotation from rest to current in world space
            Quat delta = rotationBetween(restDir, boneDir);
            worldRot[i] = delta * worldRot[parent];

            // Local rotation = inverse(parentWorld) * worldRot
            outLocalRotations[i] = worldRot[parent].conjugate() * worldRot[i];
            outLocalRotations[i] = outLocalRotations[i].normalized();
        }
    }

    outLocalRotations[0] = Quat::identity();
}

// ---------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------

std::vector<float> MathUtils::skeletonToVector(const SkeletonFrame& frame) {
    std::vector<float> vec(FRAME_DIM);
    for (int i = 0; i < JOINT_COUNT; ++i) {
        const auto& r6d = frame.joints[i].rotation6D;
        for (int d = 0; d < ROTATION_DIM; ++d) {
            vec[i * ROTATION_DIM + d] = r6d[d];
        }
    }
    return vec;
}

SkeletonFrame MathUtils::vectorToSkeleton(const std::vector<float>& vec,
                                            const SkeletonFrame& templateFrame) {
    SkeletonFrame frame = templateFrame;
    for (int i = 0; i < JOINT_COUNT; ++i) {
        Vec6 r6d;
        for (int d = 0; d < ROTATION_DIM; ++d) {
            r6d[d] = vec[i * ROTATION_DIM + d];
        }
        frame.joints[i].rotation6D = r6d;
        frame.joints[i].localRotation = rot6DToQuat(r6d);
        frame.joints[i].localEulerDeg = quatToEulerDeg(frame.joints[i].localRotation);
    }
    return frame;
}

Eigen::MatrixXf MathUtils::clipToMatrix(const AnimClip& clip) {
    int numFrames = static_cast<int>(clip.frames.size());
    Eigen::MatrixXf mat(numFrames, FRAME_DIM);

    for (int f = 0; f < numFrames; ++f) {
        auto vec = skeletonToVector(clip.frames[f]);
        for (int d = 0; d < FRAME_DIM; ++d) {
            mat(f, d) = vec[d];
        }
    }
    return mat;
}

AnimClip MathUtils::matrixToClip(const Eigen::MatrixXf& mat,
                                   const AnimClip& templateClip) {
    AnimClip clip = templateClip;
    clip.frames.resize(mat.rows());

    for (int f = 0; f < mat.rows(); ++f) {
        std::vector<float> vec(FRAME_DIM);
        for (int d = 0; d < FRAME_DIM; ++d) {
            vec[d] = mat(f, d);
        }
        SkeletonFrame templateFrame;
        if (f < static_cast<int>(templateClip.frames.size()))
            templateFrame = templateClip.frames[f];
        clip.frames[f] = vectorToSkeleton(vec, templateFrame);
        clip.frames[f].frameIndex = f;
    }
    return clip;
}

} // namespace hm
