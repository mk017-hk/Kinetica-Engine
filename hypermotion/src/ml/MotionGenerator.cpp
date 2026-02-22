#include "HyperMotion/ml/MotionGenerator.h"
#include "HyperMotion/core/MathUtils.h"
#include "HyperMotion/core/Logger.h"

#include <chrono>
#include <cmath>
#include <algorithm>

namespace hm::ml {

static constexpr const char* TAG = "MotionGenerator";

// Joint angle limits (degrees) for plausibility checking
struct JointLimits {
    float minX, maxX, minY, maxY, minZ, maxZ;
};

static const JointLimits JOINT_LIMITS[JOINT_COUNT] = {
    {-60, 60, -60, 60, -60, 60},     // Hips
    {-30, 60, -30, 30, -30, 30},     // Spine
    {-30, 50, -30, 30, -30, 30},     // Spine1
    {-30, 40, -40, 40, -30, 30},     // Spine2
    {-60, 60, -70, 70, -40, 40},     // Neck
    {-50, 50, -60, 60, -30, 30},     // Head
    {-20, 20, -10, 10, -40, 20},     // LeftShoulder
    {-180, 60, -90, 90, -90, 90},    // LeftArm
    {0, 150, -10, 10, -90, 90},      // LeftForeArm
    {-70, 70, -30, 30, -80, 80},     // LeftHand
    {-20, 20, -10, 10, -20, 40},     // RightShoulder
    {-180, 60, -90, 90, -90, 90},    // RightArm
    {0, 150, -10, 10, -90, 90},      // RightForeArm
    {-70, 70, -30, 30, -80, 80},     // RightHand
    {-120, 30, -45, 45, -45, 45},    // LeftUpLeg
    {0, 150, -10, 10, -10, 10},      // LeftLeg
    {-40, 50, -20, 30, -20, 20},     // LeftFoot
    {-30, 60, -10, 10, -10, 10},     // LeftToeBase
    {-120, 30, -45, 45, -45, 45},    // RightUpLeg
    {0, 150, -10, 10, -10, 10},      // RightLeg
    {-40, 50, -20, 30, -20, 20},     // RightFoot
    {-30, 60, -10, 10, -10, 10}      // RightToeBase
};

struct MotionGenerator::Impl {
    MotionGeneratorConfig config;
    MotionDiffusionModel diffusionModel;
    bool initialized = false;

    Impl(const MotionGeneratorConfig& cfg)
        : config(cfg), diffusionModel(cfg.diffusionConfig) {}

    void applyJointLimits(std::vector<SkeletonFrame>& frames) {
        for (auto& frame : frames) {
            for (int j = 0; j < JOINT_COUNT; ++j) {
                auto& euler = frame.joints[j].localEulerDeg;
                const auto& limits = JOINT_LIMITS[j];

                euler.x = std::clamp(euler.x, limits.minX, limits.maxX);
                euler.y = std::clamp(euler.y, limits.minY, limits.maxY);
                euler.z = std::clamp(euler.z, limits.minZ, limits.maxZ);

                frame.joints[j].localRotation = MathUtils::eulerDegToQuat(euler);
                frame.joints[j].rotation6D = MathUtils::quatToRot6D(frame.joints[j].localRotation);
            }
        }
    }

    void applyFootContactCleanup(std::vector<SkeletonFrame>& frames) {
        if (frames.size() < 2) return;

        int leftFootIdx = static_cast<int>(Joint::LeftFoot);
        int rightFootIdx = static_cast<int>(Joint::RightFoot);

        for (size_t f = 1; f < frames.size(); ++f) {
            auto& curr = frames[f];
            const auto& prev = frames[f - 1];
            float dt = 1.0f / 30.0f;

            Vec3 leftVel = (curr.joints[leftFootIdx].worldPosition -
                           prev.joints[leftFootIdx].worldPosition) / dt;
            if (leftVel.length() < 2.0f && curr.joints[leftFootIdx].worldPosition.y < 5.0f) {
                curr.joints[leftFootIdx].worldPosition.y = 0.0f;
            }

            Vec3 rightVel = (curr.joints[rightFootIdx].worldPosition -
                            prev.joints[rightFootIdx].worldPosition) / dt;
            if (rightVel.length() < 2.0f && curr.joints[rightFootIdx].worldPosition.y < 5.0f) {
                curr.joints[rightFootIdx].worldPosition.y = 0.0f;
            }
        }
    }

    float computeQuality(const std::vector<SkeletonFrame>& frames) {
        if (frames.size() < 3) return 0.0f;

        float totalJerk = 0.0f;
        for (size_t f = 2; f < frames.size(); ++f) {
            for (int j = 0; j < JOINT_COUNT; ++j) {
                Vec3 acc = frames[f].joints[j].worldPosition -
                           frames[f - 1].joints[j].worldPosition * 2.0f +
                           frames[f - 2].joints[j].worldPosition;
                totalJerk += acc.length();
            }
        }
        float avgJerk = totalJerk / ((frames.size() - 2) * JOINT_COUNT);
        return std::clamp(std::exp(-avgJerk * 0.01f), 0.0f, 1.0f);
    }
};

MotionGenerator::MotionGenerator(const MotionGeneratorConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

MotionGenerator::~MotionGenerator() = default;
MotionGenerator::MotionGenerator(MotionGenerator&&) noexcept = default;
MotionGenerator& MotionGenerator::operator=(MotionGenerator&&) noexcept = default;

bool MotionGenerator::initialize() {
    if (!impl_->diffusionModel.initialize()) {
        HM_LOG_ERROR(TAG, "Failed to initialize diffusion model");
        return false;
    }

    impl_->initialized = true;
    HM_LOG_INFO(TAG, "Motion generator initialized");
    return true;
}

bool MotionGenerator::isInitialized() const {
    return impl_->initialized;
}

GeneratedMotion MotionGenerator::generate(const MotionCondition& condition) {
    auto startTime = std::chrono::high_resolution_clock::now();

    GeneratedMotion result;

    if (!impl_->initialized) {
        HM_LOG_ERROR(TAG, "Not initialized");
        return result;
    }

    // Flatten condition to float vector
    auto condArray = condition.flatten();
    std::vector<float> condVec(condArray.begin(), condArray.end());

    // Generate frames via ONNX diffusion
    result.frames = impl_->diffusionModel.generate(condVec);

    if (result.frames.empty()) {
        HM_LOG_ERROR(TAG, "Generation produced empty output");
        return result;
    }

    // Set root positions from condition velocity
    for (size_t f = 1; f < result.frames.size(); ++f) {
        float dt = 1.0f / 30.0f;
        result.frames[f].rootPosition = result.frames[f - 1].rootPosition +
                                         condition.velocity * dt;
        result.frames[f].rootVelocity = condition.velocity;
    }
    if (!result.frames.empty()) {
        result.frames[0].rootVelocity = condition.velocity;
    }

    // Post-processing
    if (impl_->config.enableJointLimits) {
        impl_->applyJointLimits(result.frames);
    }

    // Compute forward kinematics for world positions
    for (auto& frame : result.frames) {
        std::array<Quat, JOINT_COUNT> localRots;
        for (int j = 0; j < JOINT_COUNT; ++j) {
            localRots[j] = frame.joints[j].localRotation;
        }
        auto worldPos = MathUtils::forwardKinematics(
            frame.rootPosition, frame.rootRotation, localRots);
        for (int j = 0; j < JOINT_COUNT; ++j) {
            frame.joints[j].worldPosition = worldPos[j];
        }
    }

    if (impl_->config.enableFootContactCleanup) {
        impl_->applyFootContactCleanup(result.frames);
    }

    if (impl_->config.enablePlausibilityCheck) {
        result.quality = impl_->computeQuality(result.frames);
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    result.inferenceTimeMs = std::chrono::duration<float, std::milli>(
        endTime - startTime).count();

    HM_LOG_DEBUG(TAG, "Generated " + std::to_string(result.frames.size()) + " frames in " +
                 std::to_string(result.inferenceTimeMs) + "ms (quality=" +
                 std::to_string(result.quality) + ")");

    return result;
}

std::vector<GeneratedMotion> MotionGenerator::generateBatch(
    const std::vector<MotionCondition>& conditions) {

    std::vector<GeneratedMotion> results;
    results.reserve(conditions.size());
    for (const auto& cond : conditions) {
        results.push_back(generate(cond));
    }
    return results;
}

} // namespace hm::ml
