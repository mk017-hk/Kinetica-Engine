#include "HyperMotion/skeleton/SkeletonMapper.h"
#include "HyperMotion/skeleton/RotationSolver.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

#include <cmath>
#include <algorithm>

namespace hm::skeleton {

static constexpr const char* TAG = "SkeletonMapper";

// COCO keypoint indices
enum COCOKeypoint : int {
    COCO_NOSE = 0,
    COCO_LEFT_EYE = 1, COCO_RIGHT_EYE = 2,
    COCO_LEFT_EAR = 3, COCO_RIGHT_EAR = 4,
    COCO_LEFT_SHOULDER = 5, COCO_RIGHT_SHOULDER = 6,
    COCO_LEFT_ELBOW = 7, COCO_RIGHT_ELBOW = 8,
    COCO_LEFT_WRIST = 9, COCO_RIGHT_WRIST = 10,
    COCO_LEFT_HIP = 11, COCO_RIGHT_HIP = 12,
    COCO_LEFT_KNEE = 13, COCO_RIGHT_KNEE = 14,
    COCO_LEFT_ANKLE = 15, COCO_RIGHT_ANKLE = 16
};

struct SkeletonMapper::Impl {
    SkeletonMapperConfig config;
    RotationSolver rotSolver;
    Vec3 prevRootPos;
    Quat prevRootRot = Quat::identity();
    bool hasPrevFrame = false;

    Vec3 getKeypoint3D(const DetectedPerson& person, int idx) {
        return person.keypoints3D[idx].position;
    }

    float getConfidence(const DetectedPerson& person, int idx) {
        return person.keypoints3D[idx].confidence;
    }

    Vec3 midpoint(const Vec3& a, const Vec3& b) {
        return (a + b) * 0.5f;
    }

    Vec3 interpolate(const Vec3& a, const Vec3& b, float t) {
        return a + (b - a) * t;
    }

    std::array<Vec3, JOINT_COUNT> mapCOCOToGameSkeleton(const DetectedPerson& person) {
        std::array<Vec3, JOINT_COUNT> positions{};

        Vec3 leftHip = getKeypoint3D(person, COCO_LEFT_HIP);
        Vec3 rightHip = getKeypoint3D(person, COCO_RIGHT_HIP);
        Vec3 leftShoulder = getKeypoint3D(person, COCO_LEFT_SHOULDER);
        Vec3 rightShoulder = getKeypoint3D(person, COCO_RIGHT_SHOULDER);

        Vec3 hipCenter = midpoint(leftHip, rightHip);
        Vec3 shoulderCenter = midpoint(leftShoulder, rightShoulder);
        Vec3 nose = getKeypoint3D(person, COCO_NOSE);

        // Hips
        positions[static_cast<int>(Joint::Hips)] = hipCenter;

        // Spine chain: interpolate between hips and shoulder midpoint
        positions[static_cast<int>(Joint::Spine)] = interpolate(hipCenter, shoulderCenter, 0.33f);
        positions[static_cast<int>(Joint::Spine1)] = interpolate(hipCenter, shoulderCenter, 0.66f);
        positions[static_cast<int>(Joint::Spine2)] = shoulderCenter;

        // Neck and Head
        positions[static_cast<int>(Joint::Neck)] = interpolate(shoulderCenter, nose, 0.5f);
        positions[static_cast<int>(Joint::Head)] = nose;

        // Left arm chain
        positions[static_cast<int>(Joint::LeftShoulder)] = leftShoulder;
        positions[static_cast<int>(Joint::LeftArm)] = leftShoulder;
        Vec3 leftElbow = getKeypoint3D(person, COCO_LEFT_ELBOW);
        positions[static_cast<int>(Joint::LeftForeArm)] = leftElbow;
        Vec3 leftWrist = getKeypoint3D(person, COCO_LEFT_WRIST);
        positions[static_cast<int>(Joint::LeftHand)] = leftWrist;

        // Right arm chain
        positions[static_cast<int>(Joint::RightShoulder)] = rightShoulder;
        positions[static_cast<int>(Joint::RightArm)] = rightShoulder;
        Vec3 rightElbow = getKeypoint3D(person, COCO_RIGHT_ELBOW);
        positions[static_cast<int>(Joint::RightForeArm)] = rightElbow;
        Vec3 rightWrist = getKeypoint3D(person, COCO_RIGHT_WRIST);
        positions[static_cast<int>(Joint::RightHand)] = rightWrist;

        // Left leg chain
        positions[static_cast<int>(Joint::LeftUpLeg)] = leftHip;
        Vec3 leftKnee = getKeypoint3D(person, COCO_LEFT_KNEE);
        positions[static_cast<int>(Joint::LeftLeg)] = leftKnee;
        Vec3 leftAnkle = getKeypoint3D(person, COCO_LEFT_ANKLE);
        positions[static_cast<int>(Joint::LeftFoot)] = leftAnkle;
        // Toe: extend forward from ankle
        Vec3 leftFootDir = (leftAnkle - leftKnee).normalized();
        Vec3 leftToe = leftAnkle + Vec3{0.0f, -3.0f, 8.0f};
        positions[static_cast<int>(Joint::LeftToeBase)] = leftToe;

        // Right leg chain
        positions[static_cast<int>(Joint::RightUpLeg)] = rightHip;
        Vec3 rightKnee = getKeypoint3D(person, COCO_RIGHT_KNEE);
        positions[static_cast<int>(Joint::RightLeg)] = rightKnee;
        Vec3 rightAnkle = getKeypoint3D(person, COCO_RIGHT_ANKLE);
        positions[static_cast<int>(Joint::RightFoot)] = rightAnkle;
        Vec3 rightToe = rightAnkle + Vec3{0.0f, -3.0f, 8.0f};
        positions[static_cast<int>(Joint::RightToeBase)] = rightToe;

        return positions;
    }

    Quat computeRootRotation(const DetectedPerson& person) {
        Vec3 leftHip = getKeypoint3D(person, COCO_LEFT_HIP);
        Vec3 rightHip = getKeypoint3D(person, COCO_RIGHT_HIP);
        Vec3 leftShoulder = getKeypoint3D(person, COCO_LEFT_SHOULDER);
        Vec3 rightShoulder = getKeypoint3D(person, COCO_RIGHT_SHOULDER);

        Vec3 hipCenter = midpoint(leftHip, rightHip);
        Vec3 shoulderCenter = midpoint(leftShoulder, rightShoulder);

        // Up direction: hips to shoulders
        Vec3 up = (shoulderCenter - hipCenter).normalized();

        // Right direction: left hip to right hip
        Vec3 right = (rightHip - leftHip).normalized();

        // Forward direction: cross product
        Vec3 forward = right.cross(up).normalized();
        up = forward.cross(right).normalized();

        Mat3 rotMat;
        rotMat.setCol(0, right);
        rotMat.setCol(1, up);
        rotMat.setCol(2, forward);

        return MathUtils::mat3ToQuat(rotMat);
    }
};

SkeletonMapper::SkeletonMapper(const SkeletonMapperConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

SkeletonMapper::~SkeletonMapper() = default;

SkeletonFrame SkeletonMapper::mapToSkeleton(const DetectedPerson& person,
                                             double timestamp, int frameIndex) {
    SkeletonFrame frame;
    frame.timestamp = timestamp;
    frame.frameIndex = frameIndex;
    frame.trackingID = person.id;

    // Map COCO keypoints to game skeleton positions
    auto worldPositions = impl_->mapCOCOToGameSkeleton(person);

    // Set root position
    frame.rootPosition = worldPositions[0];

    // Compute root rotation from body orientation
    frame.rootRotation = impl_->computeRootRotation(person);

    // Solve joint rotations
    Quat rootRot;
    impl_->rotSolver.solve(worldPositions, frame.joints, rootRot);
    frame.rootRotation = rootRot;

    // Compute velocities from previous frame
    if (impl_->hasPrevFrame) {
        float dt = 1.0f / 30.0f; // Assume 30fps; would use real dt if available
        frame.rootVelocity = (frame.rootPosition - impl_->prevRootPos) / dt;

        // Angular velocity from quaternion difference
        Quat deltaRot = frame.rootRotation * impl_->prevRootRot.conjugate();
        Vec3 axis;
        float angle;
        MathUtils::toAxisAngle(deltaRot, axis, angle);
        frame.rootAngularVel = axis * (angle / dt);
    }

    impl_->prevRootPos = frame.rootPosition;
    impl_->prevRootRot = frame.rootRotation;
    impl_->hasPrevFrame = true;

    // Set confidence per joint based on source keypoints
    for (int j = 0; j < JOINT_COUNT; ++j) {
        frame.joints[j].worldPosition = worldPositions[j];
    }

    // Fill 6D and Euler representations
    RotationSolver::fillRotationRepresentations(frame.joints);

    return frame;
}

std::vector<SkeletonFrame> SkeletonMapper::mapSequence(
    const std::vector<PoseFrameResult>& poseResults, int trackingID) {

    std::vector<SkeletonFrame> frames;
    impl_->hasPrevFrame = false;

    for (const auto& poseFrame : poseResults) {
        for (const auto& person : poseFrame.persons) {
            if (person.id == trackingID) {
                frames.push_back(mapToSkeleton(person, poseFrame.timestamp,
                                               static_cast<int>(frames.size())));
                break;
            }
        }
    }

    return frames;
}

SkeletonFrame SkeletonMapper::getRestPose() {
    SkeletonFrame frame;
    frame.timestamp = 0.0;
    frame.frameIndex = 0;
    frame.rootPosition = {0, 0, 0};
    frame.rootRotation = Quat::identity();

    const auto& offsets = getRestPoseBoneOffsets();

    // Compute world positions from offsets
    std::array<Vec3, JOINT_COUNT> worldPos;
    worldPos[0] = {0, 90, 0}; // Hips at ~90cm height

    for (int i = 1; i < JOINT_COUNT; ++i) {
        int parent = JOINT_PARENT[i];
        worldPos[i] = worldPos[parent] + offsets[i];
    }

    for (int i = 0; i < JOINT_COUNT; ++i) {
        frame.joints[i].localRotation = Quat::identity();
        frame.joints[i].localEulerDeg = {0, 0, 0};
        frame.joints[i].rotation6D = MathUtils::quatToRot6D(Quat::identity());
        frame.joints[i].worldPosition = worldPos[i];
        frame.joints[i].confidence = 1.0f;
    }

    return frame;
}

} // namespace hm::skeleton
