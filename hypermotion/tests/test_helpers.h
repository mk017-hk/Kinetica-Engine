#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/core/MathUtils.h"
#include <cmath>
#include <random>

namespace hm::test {

// Floating point comparison tolerance
constexpr float kEps = 1e-4f;
constexpr float kLooseEps = 1e-2f;

// Compare quaternions accounting for double-cover (q == -q)
inline bool quatNearEqual(const Quat& a, const Quat& b, float eps = kEps) {
    float dotVal = a.w * b.w + a.x * b.x + a.y * b.y + a.z * b.z;
    return std::abs(std::abs(dotVal) - 1.0f) < eps;
}

inline bool vec3NearEqual(const Vec3& a, const Vec3& b, float eps = kEps) {
    return std::abs(a.x - b.x) < eps &&
           std::abs(a.y - b.y) < eps &&
           std::abs(a.z - b.z) < eps;
}

// Create a skeleton frame with identity rotations at given root position
inline SkeletonFrame makeIdentityFrame(const Vec3& rootPos = {0, 90, 0},
                                        int frameIdx = 0,
                                        double timestamp = 0.0) {
    SkeletonFrame frame;
    frame.frameIndex = frameIdx;
    frame.timestamp = timestamp;
    frame.rootPosition = rootPos;
    frame.rootRotation = Quat::identity();
    for (int i = 0; i < JOINT_COUNT; ++i) {
        frame.joints[i].localRotation = Quat::identity();
        frame.joints[i].rotation6D = MathUtils::quatToRot6D(Quat::identity());
        frame.joints[i].localEulerDeg = {0, 0, 0};
        frame.joints[i].confidence = 1.0f;
    }
    // Set world positions via forward kinematics
    std::array<Quat, JOINT_COUNT> localRots;
    for (int i = 0; i < JOINT_COUNT; ++i)
        localRots[i] = Quat::identity();
    auto worldPos = MathUtils::forwardKinematics(rootPos, Quat::identity(), localRots);
    for (int i = 0; i < JOINT_COUNT; ++i)
        frame.joints[i].worldPosition = worldPos[i];
    return frame;
}

// Create a sequence of frames simulating linear motion along X axis
inline std::vector<SkeletonFrame> makeWalkingSequence(int numFrames = 60,
                                                       float speed = 100.0f,
                                                       float fps = 30.0f) {
    std::vector<SkeletonFrame> frames(numFrames);
    float dt = 1.0f / fps;
    for (int i = 0; i < numFrames; ++i) {
        frames[i] = makeIdentityFrame(
            {speed * i * dt, 90.0f, 0.0f}, i, i * dt);
        frames[i].rootVelocity = {speed, 0, 0};
    }
    return frames;
}

// Create a simple AnimClip
inline AnimClip makeTestClip(int numFrames = 30, float fps = 30.0f,
                              const std::string& name = "test_clip") {
    AnimClip clip;
    clip.name = name;
    clip.fps = fps;
    clip.trackingID = 0;
    clip.frames = makeWalkingSequence(numFrames, 100.0f, fps);

    if (numFrames > 20) {
        MotionSegment seg1;
        seg1.type = MotionType::Walk;
        seg1.startFrame = 0;
        seg1.endFrame = numFrames / 2;
        seg1.confidence = 0.95f;

        MotionSegment seg2;
        seg2.type = MotionType::Jog;
        seg2.startFrame = numFrames / 2;
        seg2.endFrame = numFrames - 1;
        seg2.confidence = 0.9f;

        clip.segments = {seg1, seg2};
    }
    return clip;
}

// Inject a spike outlier into a frame sequence at a specific joint
inline void injectOutlier(std::vector<SkeletonFrame>& frames,
                           int frameIdx, int jointIdx,
                           const Vec3& offset = {500, 500, 500}) {
    auto& pos = frames[frameIdx].joints[jointIdx].worldPosition;
    pos.x += offset.x;
    pos.y += offset.y;
    pos.z += offset.z;
}

} // namespace hm::test
