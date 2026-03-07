#pragma once

#include "HyperMotion/core/Types.h"
#include <string>
#include <vector>

namespace hm::dataset {

struct ClipQualityConfig {
    float minDurationSec = 0.5f;
    float minAvgConfidence = 0.3f;        // minimum average joint confidence
    float maxJitterThreshold = 50.0f;     // cm/frame max joint position jitter
    float limbLengthTolerancePct = 0.3f;  // 30% deviation from median limb length
    int maxMissingJointFrames = 5;        // max consecutive frames with missing joints
    float minMissingJointConfidence = 0.1f;
};

/// Reason a clip was rejected.
struct QualityRejection {
    enum Reason {
        TooShort,
        LowConfidence,
        ExcessiveJitter,
        InconsistentLimbs,
        MissingJoints
    };
    Reason reason;
    std::string description;
    float value = 0.0f;     // the measured value
    float threshold = 0.0f; // the threshold it failed
};

/// Result of quality assessment for a single clip.
struct QualityResult {
    bool accepted = false;
    float overallScore = 0.0f;  // 0-1, higher is better
    float avgConfidence = 0.0f;
    float maxJitter = 0.0f;
    float limbConsistency = 0.0f;
    int missingJointFrames = 0;
    std::vector<QualityRejection> rejections;
};

/// Filters animation clips based on quality metrics.
/// Rejects clips with missing joints, excessive jitter, inconsistent
/// limb lengths, or other data quality issues.
class ClipQualityFilter {
public:
    explicit ClipQualityFilter(const ClipQualityConfig& config = {});
    ~ClipQualityFilter() = default;

    /// Assess quality of a single clip.
    QualityResult assess(const AnimClip& clip) const;

    /// Filter a batch of clips, returning only accepted clips.
    struct FilterResult {
        std::vector<AnimClip> accepted;
        std::vector<AnimClip> rejected;
        std::vector<QualityResult> acceptedResults;
        std::vector<QualityResult> rejectedResults;
    };

    FilterResult filter(const std::vector<AnimClip>& clips) const;

private:
    ClipQualityConfig config_;

    float computeAvgConfidence(const AnimClip& clip) const;
    float computeMaxJitter(const AnimClip& clip) const;
    float computeLimbConsistency(const AnimClip& clip) const;
    int countMissingJointFrames(const AnimClip& clip) const;
};

} // namespace hm::dataset
