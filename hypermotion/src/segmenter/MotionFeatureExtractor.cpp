#include "HyperMotion/segmenter/MotionFeatureExtractor.h"
#include "HyperMotion/core/MathUtils.h"

#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>

namespace hm::segmenter {

static constexpr float kSprintSpeed = 800.0f;   // cm/s
static constexpr float kMaxTurnRate = 360.0f;    // deg/s

// Joint group indices for body-part analysis
static constexpr int kTorsoJoints[] = {0, 1, 2, 3, 4, 5};   // Hips -> Head
static constexpr int kLeftArmJoints[] = {6, 7, 8, 9};         // LShoulder -> LHand
static constexpr int kRightArmJoints[] = {10, 11, 12, 13};    // RShoulder -> RHand
static constexpr int kLeftLegJoints[] = {14, 15, 16, 17};     // LUpLeg -> LToeBase
static constexpr int kRightLegJoints[] = {18, 19, 20, 21};    // RUpLeg -> RToeBase

MotionFeatureExtractor::MotionFeatureExtractor() = default;
MotionFeatureExtractor::~MotionFeatureExtractor() = default;

std::vector<float> MotionFeatureExtractor::extract(const SkeletonFrame& frame) {
    std::vector<float> features(FEATURE_DIM, 0.0f);
    int idx = 0;

    // 22 joints x 3 Euler angles = 66D (normalized to [-1, 1])
    for (int j = 0; j < JOINT_COUNT; ++j) {
        const auto& euler = frame.joints[j].localEulerDeg;
        features[idx++] = euler.x / 180.0f;
        features[idx++] = euler.y / 180.0f;
        features[idx++] = euler.z / 180.0f;
    }

    // Root velocity 3D (normalized by sprint speed)
    features[idx++] = frame.rootVelocity.x / kSprintSpeed;
    features[idx++] = frame.rootVelocity.y / kSprintSpeed;
    features[idx++] = frame.rootVelocity.z / kSprintSpeed;

    // Root angular velocity magnitude
    float angularMag = frame.rootAngularVel.length();
    features[idx++] = angularMag / kMaxTurnRate;

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

std::vector<std::vector<float>> MotionFeatureExtractor::extractSequenceExtended(
    const std::vector<SkeletonFrame>& frames) {

    auto baseFeatures = extractSequence(frames);
    int numFrames = static_cast<int>(frames.size());

    std::vector<std::vector<float>> extended(numFrames,
        std::vector<float>(EXTENDED_FEATURE_DIM, 0.0f));

    for (int f = 0; f < numFrames; ++f) {
        // Copy base features [0..69]
        std::copy(baseFeatures[f].begin(), baseFeatures[f].end(), extended[f].begin());

        // Delta features [70..139]: frame-to-frame differences
        if (f > 0) {
            for (int d = 0; d < FEATURE_DIM; ++d) {
                extended[f][FEATURE_DIM + d] = baseFeatures[f][d] - baseFeatures[f - 1][d];
            }
        }
        // else: delta features remain zero for the first frame
    }

    return extended;
}

float MotionFeatureExtractor::computeJointGroupVelocity(
    const std::vector<SkeletonFrame>& frames,
    const int* jointIndices, int numJoints,
    int startFrame, int endFrame) {

    if (endFrame <= startFrame) return 0.0f;

    float totalVelocity = 0.0f;
    int count = 0;

    for (int f = startFrame + 1; f <= endFrame && f < static_cast<int>(frames.size()); ++f) {
        for (int ji = 0; ji < numJoints; ++ji) {
            int j = jointIndices[ji];
            Vec3 delta{
                frames[f].joints[j].localEulerDeg.x - frames[f - 1].joints[j].localEulerDeg.x,
                frames[f].joints[j].localEulerDeg.y - frames[f - 1].joints[j].localEulerDeg.y,
                frames[f].joints[j].localEulerDeg.z - frames[f - 1].joints[j].localEulerDeg.z
            };
            totalVelocity += delta.length();
            count++;
        }
    }

    return count > 0 ? totalVelocity / count : 0.0f;
}

float MotionFeatureExtractor::computeJointGroupROM(
    const std::vector<SkeletonFrame>& frames,
    const int* jointIndices, int numJoints,
    int startFrame, int endFrame) {

    if (endFrame <= startFrame) return 0.0f;

    float totalROM = 0.0f;

    for (int ji = 0; ji < numJoints; ++ji) {
        int j = jointIndices[ji];

        // Track min/max Euler per axis across the window
        float minX = std::numeric_limits<float>::max(), maxX = std::numeric_limits<float>::lowest();
        float minY = minX, maxY = maxX;
        float minZ = minX, maxZ = maxX;

        for (int f = startFrame; f <= endFrame && f < static_cast<int>(frames.size()); ++f) {
            const auto& e = frames[f].joints[j].localEulerDeg;
            minX = std::min(minX, e.x); maxX = std::max(maxX, e.x);
            minY = std::min(minY, e.y); maxY = std::max(maxY, e.y);
            minZ = std::min(minZ, e.z); maxZ = std::max(maxZ, e.z);
        }

        // ROM = sum of axis ranges
        totalROM += (maxX - minX) + (maxY - minY) + (maxZ - minZ);
    }

    return numJoints > 0 ? totalROM / numJoints : 0.0f;
}

MotionFeatureExtractor::BodyPartStats MotionFeatureExtractor::computeBodyPartStats(
    const std::vector<SkeletonFrame>& frames, int startFrame, int endFrame) {

    BodyPartStats stats;

    stats.torsoVelocity    = computeJointGroupVelocity(frames, kTorsoJoints, 6, startFrame, endFrame);
    stats.leftArmVelocity  = computeJointGroupVelocity(frames, kLeftArmJoints, 4, startFrame, endFrame);
    stats.rightArmVelocity = computeJointGroupVelocity(frames, kRightArmJoints, 4, startFrame, endFrame);
    stats.leftLegVelocity  = computeJointGroupVelocity(frames, kLeftLegJoints, 4, startFrame, endFrame);
    stats.rightLegVelocity = computeJointGroupVelocity(frames, kRightLegJoints, 4, startFrame, endFrame);

    stats.torsoROM    = computeJointGroupROM(frames, kTorsoJoints, 6, startFrame, endFrame);
    stats.leftArmROM  = computeJointGroupROM(frames, kLeftArmJoints, 4, startFrame, endFrame);
    stats.rightArmROM = computeJointGroupROM(frames, kRightArmJoints, 4, startFrame, endFrame);
    stats.leftLegROM  = computeJointGroupROM(frames, kLeftLegJoints, 4, startFrame, endFrame);
    stats.rightLegROM = computeJointGroupROM(frames, kRightLegJoints, 4, startFrame, endFrame);

    return stats;
}

MotionType MotionFeatureExtractor::classifyHeuristic(
    const std::vector<SkeletonFrame>& frames, int startFrame, int endFrame) {

    if (frames.empty() || startFrame >= endFrame) return MotionType::Unknown;

    int count = endFrame - startFrame;
    float totalSpeed = 0.0f;
    float totalAngularVel = 0.0f;
    float maxSpeed = 0.0f;
    float totalAcceleration = 0.0f;
    float prevSpeed = 0.0f;
    float totalVerticalVel = 0.0f;
    float maxHeight = 0.0f;
    float minHeight = std::numeric_limits<float>::max();
    float totalLateralVel = 0.0f;

    for (int f = startFrame; f < endFrame && f < static_cast<int>(frames.size()); ++f) {
        float speed = frames[f].rootVelocity.length();
        totalSpeed += speed;
        maxSpeed = std::max(maxSpeed, speed);
        totalAngularVel += frames[f].rootAngularVel.length();
        totalVerticalVel += std::abs(frames[f].rootVelocity.y);
        totalLateralVel += std::abs(frames[f].rootVelocity.x);
        maxHeight = std::max(maxHeight, frames[f].rootPosition.y);
        minHeight = std::min(minHeight, frames[f].rootPosition.y);

        if (f > startFrame) {
            totalAcceleration += speed - prevSpeed;
        }
        prevSpeed = speed;
    }

    float avgSpeed = totalSpeed / count;
    float avgAngularVel = totalAngularVel / count;
    float avgAcceleration = count > 1 ? totalAcceleration / (count - 1) : 0.0f;
    float avgVerticalVel = totalVerticalVel / count;
    float avgLateralVel = totalLateralVel / count;
    float heightRange = maxHeight - minHeight;

    // Check body part activity
    auto bodyStats = computeBodyPartStats(frames, startFrame, endFrame);
    float legActivity = (bodyStats.leftLegVelocity + bodyStats.rightLegVelocity) * 0.5f;
    float armActivity = (bodyStats.leftArmVelocity + bodyStats.rightArmVelocity) * 0.5f;
    float legAsymmetry = std::abs(bodyStats.leftLegVelocity - bodyStats.rightLegVelocity);
    float armAsymmetry = std::abs(bodyStats.leftArmVelocity - bodyStats.rightArmVelocity);

    // --- Classify based on multi-feature analysis ---

    if (avgSpeed < 10.0f) return MotionType::Idle;

    // Jump: significant vertical displacement or upward velocity
    if (heightRange > 15.0f && avgVerticalVel > 30.0f) {
        return MotionType::Jump;
    }

    // Slide: high lateral velocity relative to forward, low height
    if (avgLateralVel > 100.0f && avgLateralVel > avgSpeed * 0.6f) {
        return MotionType::Slide;
    }

    // Tackle: high leg asymmetry with forward velocity and body lowering
    if (legAsymmetry > 20.0f && legActivity > 25.0f && avgSpeed > 100.0f && heightRange > 10.0f) {
        return MotionType::Tackle;
    }

    // High angular velocity => turning (distinguish left/right by yaw direction)
    if (avgAngularVel > 90.0f && avgSpeed > 10.0f) {
        // Accumulate signed angular velocity around Y axis to determine turn direction
        float totalYaw = 0.0f;
        for (int f = startFrame; f < endFrame && f < static_cast<int>(frames.size()); ++f) {
            totalYaw += frames[f].rootAngularVel.y;
        }
        return totalYaw > 0.0f ? MotionType::TurnLeft : MotionType::TurnRight;
    }

    // Strong deceleration from high speed
    if (avgAcceleration < -200.0f && maxSpeed > 300.0f) {
        return MotionType::Decelerate;
    }

    // Kick: high leg asymmetry with low root speed, one leg swinging hard
    if (legActivity > 30.0f && avgSpeed < 100.0f && legAsymmetry > 15.0f) {
        return MotionType::Kick;
    }

    // Shield: arms spread wide, low speed, torso facing opponent direction
    if (armActivity > 25.0f && armAsymmetry < 8.0f && legActivity < 10.0f && avgSpeed < 80.0f) {
        return MotionType::Shield;
    }

    // Receive: some arm activity (ball control), low leg activity, low speed
    if (armActivity > 20.0f && legActivity < 5.0f && avgSpeed < 50.0f) {
        return MotionType::Receive;
    }

    // Celebrate: high arm activity with moderate or low speed
    if (armActivity > 35.0f && avgSpeed < 150.0f && legActivity < 15.0f) {
        return MotionType::Celebrate;
    }

    // Goalkeeper: high arm ROM, wide stance, low forward speed
    if (bodyStats.leftArmROM > 40.0f && bodyStats.rightArmROM > 40.0f &&
        avgSpeed < 100.0f && legActivity > 10.0f) {
        return MotionType::Goalkeeper;
    }

    // Speed-based locomotion classification
    if (avgSpeed < 120.0f) return MotionType::Walk;
    if (avgSpeed < 250.0f) return MotionType::Jog;
    return MotionType::Sprint;
}

} // namespace hm::segmenter
