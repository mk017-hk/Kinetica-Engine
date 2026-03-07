#include "HyperMotion/motion/CanonicalMotionBuilder.h"
#include "HyperMotion/core/Logger.h"
#include "HyperMotion/core/MathUtils.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace hm::motion {

static constexpr const char* TAG = "CanonicalMotionBuilder";

struct CanonicalMotionBuilder::Impl {
    CanonicalMotionBuilderConfig config;
};

CanonicalMotionBuilder::CanonicalMotionBuilder(const CanonicalMotionBuilderConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

CanonicalMotionBuilder::~CanonicalMotionBuilder() = default;
CanonicalMotionBuilder::CanonicalMotionBuilder(CanonicalMotionBuilder&&) noexcept = default;
CanonicalMotionBuilder& CanonicalMotionBuilder::operator=(CanonicalMotionBuilder&&) noexcept = default;

// Compute limb length (distance from joint to parent)
static float computeLimbLength(const SkeletonFrame& frame, int jointIdx) {
    int parentIdx = JOINT_PARENT[jointIdx];
    if (parentIdx < 0) return 0.0f;
    Vec3 diff = frame.joints[jointIdx].worldPosition -
                frame.joints[parentIdx].worldPosition;
    return diff.length();
}

// Solve root orientation from hip-to-spine direction and hip axis
static Quat solveRootOrientation(const SkeletonFrame& frame) {
    // Forward direction: hips → spine (projected to XZ plane)
    Vec3 hips = frame.joints[static_cast<int>(Joint::Hips)].worldPosition;
    Vec3 spine = frame.joints[static_cast<int>(Joint::Spine)].worldPosition;

    // Up vector is the hip-to-spine direction
    Vec3 up = (spine - hips).normalized();
    if (up.length() < 1e-6f) up = Vec3(0, 1, 0);

    // Lateral direction: left hip to right hip
    Vec3 leftHip = frame.joints[static_cast<int>(Joint::LeftUpLeg)].worldPosition;
    Vec3 rightHip = frame.joints[static_cast<int>(Joint::RightUpLeg)].worldPosition;
    Vec3 lateral = (rightHip - leftHip).normalized();
    if (lateral.length() < 1e-6f) lateral = Vec3(1, 0, 0);

    // Forward = lateral cross up (Y-up, right-handed)
    Vec3 forward = lateral.cross(up).normalized();
    if (forward.length() < 1e-6f) forward = Vec3(0, 0, 1);

    // Recompute lateral to ensure orthogonality
    lateral = up.cross(forward).normalized();

    // Build rotation matrix [lateral, up, forward] → quaternion
    Mat3 rotMat;
    rotMat.setCol(0, lateral);
    rotMat.setCol(1, up);
    rotMat.setCol(2, forward);

    return MathUtils::mat3ToQuat(rotMat);
}

// Convert world-space joint positions to local-space relative to parent
static void worldToLocal(const SkeletonFrame& frame, const Quat& rootRot,
                         std::array<Quat, JOINT_COUNT>& localRotations,
                         std::array<Vec3, JOINT_COUNT>& localPositions) {
    const auto& offsets = getRestPoseBoneOffsets();

    for (int i = 0; i < JOINT_COUNT; ++i) {
        int parent = JOINT_PARENT[i];
        if (parent < 0) {
            // Root joint: local rotation is relative to world
            localRotations[i] = rootRot.conjugate() * frame.joints[i].localRotation;
            localPositions[i] = offsets[i];
        } else {
            // Child joint: compute offset from parent in parent's local space
            Vec3 worldDiff = frame.joints[i].worldPosition -
                            frame.joints[parent].worldPosition;
            localPositions[i] = worldDiff;

            // Local rotation: parent-relative
            Quat parentWorldRot = frame.joints[parent].localRotation;
            localRotations[i] = parentWorldRot.conjugate() * frame.joints[i].localRotation;
        }

        localRotations[i] = localRotations[i].normalized();
    }
}

std::array<float, JOINT_COUNT> CanonicalMotionBuilder::measureLimbLengths(
    const std::vector<SkeletonFrame>& frames) {

    std::array<std::vector<float>, JOINT_COUNT> allLengths;

    for (const auto& frame : frames) {
        for (int j = 0; j < JOINT_COUNT; ++j) {
            float len = computeLimbLength(frame, j);
            if (len > 0.0f) {
                allLengths[j].push_back(len);
            }
        }
    }

    // Return median per joint
    std::array<float, JOINT_COUNT> result{};
    for (int j = 0; j < JOINT_COUNT; ++j) {
        auto& v = allLengths[j];
        if (v.empty()) {
            // Use rest-pose offset length as fallback
            result[j] = getRestPoseBoneOffsets()[j].length();
        } else {
            std::sort(v.begin(), v.end());
            result[j] = v[v.size() / 2];
        }
    }
    return result;
}

CanonicalMotion CanonicalMotionBuilder::build(
    const std::vector<SkeletonFrame>& frames,
    int trackingID, float fps) const {

    CanonicalMotion motion;
    motion.trackingID = trackingID;
    motion.fps = fps;

    if (frames.empty()) return motion;

    // Measure limb lengths from the input data
    motion.measuredLimbLengths = measureLimbLengths(frames);

    // Determine canonical limb lengths
    for (int j = 0; j < JOINT_COUNT; ++j) {
        float target = impl_->config.targetLimbLengths[j];
        motion.canonicalLimbLengths[j] = (target > 0.0f)
            ? target
            : motion.measuredLimbLengths[j];
    }

    // Optional limb length stabilisation via EMA
    std::array<float, JOINT_COUNT> runningLengths = motion.canonicalLimbLengths;

    float dt = (fps > 0.0f) ? 1.0f / fps : 1.0f / 30.0f;
    Vec3 prevRootPos = frames[0].rootPosition;
    Quat prevRootRot = Quat::identity();

    motion.frames.reserve(frames.size());
    motion.rootTrajectory.reserve(frames.size());

    for (size_t i = 0; i < frames.size(); ++i) {
        const auto& sf = frames[i];
        CanonicalFrame cf;
        cf.timestamp = sf.timestamp;
        cf.frameIndex = sf.frameIndex;

        // Root position
        cf.rootPosition = sf.rootPosition;

        // Root orientation
        if (impl_->config.solveRootOrientation) {
            cf.rootRotation = solveRootOrientation(sf);
        } else {
            cf.rootRotation = sf.rootRotation;
        }

        // Root velocity
        if (i > 0) {
            cf.rootVelocity = (cf.rootPosition - prevRootPos) * (1.0f / dt);

            // Angular velocity around Y axis
            Quat deltaRot = cf.rootRotation * prevRootRot.conjugate();
            deltaRot = deltaRot.normalized();
            // Extract Y-axis rotation angle
            float angle = 2.0f * std::acos(std::clamp(std::abs(deltaRot.w), 0.0f, 1.0f));
            float sign = (deltaRot.y >= 0.0f) ? 1.0f : -1.0f;
            cf.rootAngularVelocity = sign * angle / dt;
        }
        prevRootPos = cf.rootPosition;
        prevRootRot = cf.rootRotation;

        // Convert to local space
        worldToLocal(sf, cf.rootRotation, cf.localRotations, cf.localPositions);

        // Stabilise limb lengths if enabled
        if (impl_->config.stabiliseLimbLengths) {
            float alpha = impl_->config.limbLengthAlpha;
            for (int j = 0; j < JOINT_COUNT; ++j) {
                if (JOINT_PARENT[j] < 0) continue;
                float currentLen = cf.localPositions[j].length();
                if (currentLen > 1e-6f) {
                    runningLengths[j] = (1.0f - alpha) * runningLengths[j] +
                                        alpha * currentLen;
                    // Scale the local position to the stabilised length
                    Vec3 dir = cf.localPositions[j].normalized();
                    cf.localPositions[j] = dir * runningLengths[j];
                }
            }
        }

        // Copy confidence values
        for (int j = 0; j < JOINT_COUNT; ++j) {
            cf.confidence[j] = sf.joints[j].confidence;
        }

        motion.rootTrajectory.push_back(cf.rootPosition);
        motion.frames.push_back(std::move(cf));
    }

    // Update canonical limb lengths with final stabilised values
    if (impl_->config.stabiliseLimbLengths) {
        motion.canonicalLimbLengths = runningLengths;
    }

    HM_LOG_INFO(TAG, "Built canonical motion: " +
                std::to_string(motion.frames.size()) + " frames, tracking=" +
                std::to_string(trackingID));
    return motion;
}

std::vector<SkeletonFrame> CanonicalMotionBuilder::toSkeletonFrames(
    const CanonicalMotion& motion) const {

    std::vector<SkeletonFrame> frames;
    frames.reserve(motion.frames.size());

    for (const auto& cf : motion.frames) {
        SkeletonFrame sf;
        sf.timestamp = cf.timestamp;
        sf.frameIndex = cf.frameIndex;
        sf.trackingID = motion.trackingID;
        sf.rootPosition = cf.rootPosition;
        sf.rootRotation = cf.rootRotation;
        sf.rootVelocity = cf.rootVelocity;

        // Reconstruct world positions via forward kinematics
        std::array<Quat, JOINT_COUNT> worldRotations;
        std::array<Vec3, JOINT_COUNT> worldPositions;

        for (int j = 0; j < JOINT_COUNT; ++j) {
            int parent = JOINT_PARENT[j];
            if (parent < 0) {
                worldRotations[j] = cf.rootRotation * cf.localRotations[j];
                worldPositions[j] = cf.rootPosition;
            } else {
                worldRotations[j] = worldRotations[parent] * cf.localRotations[j];
                worldPositions[j] = worldPositions[parent] +
                    worldRotations[parent].rotate(cf.localPositions[j]);
            }

            sf.joints[j].localRotation = cf.localRotations[j];
            sf.joints[j].worldPosition = worldPositions[j];
            sf.joints[j].confidence = cf.confidence[j];

            // Compute Euler angles and 6D rotation from the local rotation
            sf.joints[j].localEulerDeg = MathUtils::quatToEulerDeg(cf.localRotations[j]);
            sf.joints[j].rotation6D = MathUtils::quatToRot6D(cf.localRotations[j]);
        }

        frames.push_back(std::move(sf));
    }

    return frames;
}

void CanonicalMotionBuilder::process(AnimClip& clip) const {
    if (clip.frames.empty()) return;

    auto canonical = build(clip.frames, clip.trackingID, clip.fps);
    clip.frames = toSkeletonFrames(canonical);

    HM_LOG_DEBUG(TAG, "Processed clip '" + clip.name +
                 "' through canonical builder (" +
                 std::to_string(clip.frames.size()) + " frames)");
}

} // namespace hm::motion
