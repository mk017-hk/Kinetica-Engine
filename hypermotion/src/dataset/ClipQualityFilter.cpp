#include "HyperMotion/dataset/ClipQualityFilter.h"
#include "HyperMotion/core/Logger.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace hm::dataset {

static constexpr const char* TAG = "ClipQualityFilter";

// Limb pairs for consistency checking (parent joint, child joint)
static constexpr std::pair<int, int> LIMB_PAIRS[] = {
    {static_cast<int>(Joint::LeftArm), static_cast<int>(Joint::LeftForeArm)},
    {static_cast<int>(Joint::LeftForeArm), static_cast<int>(Joint::LeftHand)},
    {static_cast<int>(Joint::RightArm), static_cast<int>(Joint::RightForeArm)},
    {static_cast<int>(Joint::RightForeArm), static_cast<int>(Joint::RightHand)},
    {static_cast<int>(Joint::LeftUpLeg), static_cast<int>(Joint::LeftLeg)},
    {static_cast<int>(Joint::LeftLeg), static_cast<int>(Joint::LeftFoot)},
    {static_cast<int>(Joint::RightUpLeg), static_cast<int>(Joint::RightLeg)},
    {static_cast<int>(Joint::RightLeg), static_cast<int>(Joint::RightFoot)},
};
static constexpr int NUM_LIMB_PAIRS = 8;

ClipQualityFilter::ClipQualityFilter(const ClipQualityConfig& config)
    : config_(config) {}

float ClipQualityFilter::computeAvgConfidence(const AnimClip& clip) const {
    if (clip.frames.empty()) return 0.0f;
    float total = 0;
    int count = 0;
    for (const auto& frame : clip.frames) {
        for (int j = 0; j < JOINT_COUNT; ++j) {
            total += frame.joints[j].confidence;
            count++;
        }
    }
    return count > 0 ? total / count : 0.0f;
}

float ClipQualityFilter::computeMaxJitter(const AnimClip& clip) const {
    if (clip.frames.size() < 3) return 0.0f;
    float maxJitter = 0;

    for (int j = 0; j < JOINT_COUNT; ++j) {
        for (size_t i = 1; i + 1 < clip.frames.size(); ++i) {
            const auto& prev = clip.frames[i - 1].joints[j].worldPosition;
            const auto& curr = clip.frames[i].joints[j].worldPosition;
            const auto& next = clip.frames[i + 1].joints[j].worldPosition;

            // Jitter = deviation from midpoint of prev and next
            Vec3 mid{(prev.x + next.x) * 0.5f,
                     (prev.y + next.y) * 0.5f,
                     (prev.z + next.z) * 0.5f};
            float dx = curr.x - mid.x;
            float dy = curr.y - mid.y;
            float dz = curr.z - mid.z;
            float jitter = std::sqrt(dx * dx + dy * dy + dz * dz);
            maxJitter = std::max(maxJitter, jitter);
        }
    }
    return maxJitter;
}

float ClipQualityFilter::computeLimbConsistency(const AnimClip& clip) const {
    if (clip.frames.size() < 2) return 1.0f;

    float worstConsistency = 1.0f;

    for (int p = 0; p < NUM_LIMB_PAIRS; ++p) {
        auto [parentIdx, childIdx] = LIMB_PAIRS[p];

        // Compute limb length for each frame
        std::vector<float> lengths;
        lengths.reserve(clip.frames.size());

        for (const auto& frame : clip.frames) {
            const auto& p1 = frame.joints[parentIdx].worldPosition;
            const auto& p2 = frame.joints[childIdx].worldPosition;
            float dx = p2.x - p1.x, dy = p2.y - p1.y, dz = p2.z - p1.z;
            float len = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (len > 0.1f) lengths.push_back(len);
        }

        if (lengths.size() < 2) continue;

        // Compute median
        std::sort(lengths.begin(), lengths.end());
        float median = lengths[lengths.size() / 2];
        if (median < 0.1f) continue;

        // Check max deviation from median
        float maxDev = 0;
        for (float l : lengths) {
            maxDev = std::max(maxDev, std::abs(l - median) / median);
        }

        worstConsistency = std::min(worstConsistency, 1.0f - maxDev);
    }

    return std::max(0.0f, worstConsistency);
}

int ClipQualityFilter::countMissingJointFrames(const AnimClip& clip) const {
    int maxConsecutive = 0;
    int currentRun = 0;

    for (const auto& frame : clip.frames) {
        bool hasMissing = false;
        for (int j = 0; j < JOINT_COUNT; ++j) {
            if (frame.joints[j].confidence < config_.minMissingJointConfidence) {
                hasMissing = true;
                break;
            }
        }
        if (hasMissing) {
            currentRun++;
            maxConsecutive = std::max(maxConsecutive, currentRun);
        } else {
            currentRun = 0;
        }
    }
    return maxConsecutive;
}

QualityResult ClipQualityFilter::assess(const AnimClip& clip) const {
    QualityResult result;
    result.accepted = true;

    // Duration check
    float duration = clip.frames.empty() ? 0.0f :
        static_cast<float>(clip.frames.size() - 1) / clip.fps;
    if (duration < config_.minDurationSec) {
        result.accepted = false;
        result.rejections.push_back({
            QualityRejection::TooShort,
            "Clip too short: " + std::to_string(duration) + "s",
            duration, config_.minDurationSec
        });
    }

    // Confidence check
    result.avgConfidence = computeAvgConfidence(clip);
    if (result.avgConfidence < config_.minAvgConfidence) {
        result.accepted = false;
        result.rejections.push_back({
            QualityRejection::LowConfidence,
            "Low average confidence: " + std::to_string(result.avgConfidence),
            result.avgConfidence, config_.minAvgConfidence
        });
    }

    // Jitter check
    result.maxJitter = computeMaxJitter(clip);
    if (result.maxJitter > config_.maxJitterThreshold) {
        result.accepted = false;
        result.rejections.push_back({
            QualityRejection::ExcessiveJitter,
            "Excessive jitter: " + std::to_string(result.maxJitter) + " cm",
            result.maxJitter, config_.maxJitterThreshold
        });
    }

    // Limb consistency check
    result.limbConsistency = computeLimbConsistency(clip);
    if (result.limbConsistency < (1.0f - config_.limbLengthTolerancePct)) {
        result.accepted = false;
        result.rejections.push_back({
            QualityRejection::InconsistentLimbs,
            "Inconsistent limb lengths: " + std::to_string(result.limbConsistency),
            result.limbConsistency, 1.0f - config_.limbLengthTolerancePct
        });
    }

    // Missing joints check
    result.missingJointFrames = countMissingJointFrames(clip);
    if (result.missingJointFrames > config_.maxMissingJointFrames) {
        result.accepted = false;
        result.rejections.push_back({
            QualityRejection::MissingJoints,
            "Missing joints: " + std::to_string(result.missingJointFrames) + " consecutive frames",
            static_cast<float>(result.missingJointFrames),
            static_cast<float>(config_.maxMissingJointFrames)
        });
    }

    // Overall score (weighted average of metrics)
    result.overallScore =
        0.30f * result.avgConfidence +
        0.25f * std::max(0.0f, 1.0f - result.maxJitter / config_.maxJitterThreshold) +
        0.25f * result.limbConsistency +
        0.20f * (1.0f - static_cast<float>(result.missingJointFrames) /
                 std::max(1, config_.maxMissingJointFrames * 2));
    result.overallScore = std::clamp(result.overallScore, 0.0f, 1.0f);

    return result;
}

ClipQualityFilter::FilterResult ClipQualityFilter::filter(
    const std::vector<AnimClip>& clips) const {

    FilterResult result;
    for (const auto& clip : clips) {
        auto qr = assess(clip);
        if (qr.accepted) {
            result.accepted.push_back(clip);
            result.acceptedResults.push_back(qr);
        } else {
            result.rejected.push_back(clip);
            result.rejectedResults.push_back(qr);
        }
    }

    HM_LOG_INFO(TAG, "Quality filter: " + std::to_string(result.accepted.size()) +
                " accepted, " + std::to_string(result.rejected.size()) + " rejected");
    return result;
}

} // namespace hm::dataset
