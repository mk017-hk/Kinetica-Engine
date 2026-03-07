#pragma once

#include "HyperMotion/core/Types.h"
#include <string>
#include <vector>
#include <memory>

namespace hm::dataset {

struct ClipExtractorConfig {
    float minClipDurationSec = 0.5f;
    float maxClipDurationSec = 5.0f;
    float fps = 30.0f;

    // Segmentation triggers
    float velocityChangeThreshold = 150.0f;   // cm/s change between windows
    float directionChangeThreshold = 45.0f;   // degrees
    float jumpVelocityThreshold = 100.0f;     // vertical cm/s for jump detect
    float stopVelocityThreshold = 10.0f;      // cm/s for sudden stop

    // Analysis window
    int analysisWindowFrames = 5;             // frames to average over
};

/// Metadata attached to each extracted clip.
struct ClipMetadata {
    int playerID = -1;
    std::string motionType;
    float confidence = 0.0f;
    float durationSec = 0.0f;
    int startFrame = 0;
    int endFrame = 0;
    float avgVelocity = 0.0f;
    float maxVelocity = 0.0f;
    Vec3 avgDirection;
    bool hasFootContacts = false;
};

/// Extracts animation clips from continuous skeleton sequences by detecting
/// motion boundaries (velocity changes, direction changes, jumps, stops).
class ClipExtractor {
public:
    explicit ClipExtractor(const ClipExtractorConfig& config = {});
    ~ClipExtractor();

    ClipExtractor(const ClipExtractor&) = delete;
    ClipExtractor& operator=(const ClipExtractor&) = delete;
    ClipExtractor(ClipExtractor&&) noexcept;
    ClipExtractor& operator=(ClipExtractor&&) noexcept;

    /// Extract clips from a continuous skeleton sequence for a single player.
    struct ExtractionResult {
        std::vector<AnimClip> clips;
        std::vector<ClipMetadata> metadata;
    };

    ExtractionResult extract(const std::vector<SkeletonFrame>& frames,
                              int playerID = -1);

    /// Extract clips using existing motion segments as boundaries.
    ExtractionResult extractFromSegments(
        const std::vector<SkeletonFrame>& frames,
        const std::vector<MotionSegment>& segments,
        int playerID = -1);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::dataset
