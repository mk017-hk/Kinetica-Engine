#pragma once

#include "HyperMotion/core/Types.h"
#include <vector>
#include <memory>

namespace hm::motion {

/// Configuration for foot contact detection.
struct FootContactDetectorConfig {
    float velocityThreshold = 2.0f;     // cm/s — foot considered planted below this
    float heightThreshold = 5.0f;       // cm — max height above ground for contact
    float groundHeight = 0.0f;          // Y coordinate of the ground plane
    float transitionSmoothing = 0.15f;  // EMA alpha for blend transitions
    int stabilityWindow = 3;            // frames of consistent state to confirm contact
    float fps = 30.0f;                  // frame rate for velocity computation
};

/// Detects per-frame foot contact from skeleton animation data.
///
/// Uses three signals:
///   1. Foot velocity (finite difference of foot world position)
///   2. Foot height relative to ground plane
///   3. Temporal stability — requires consistent state across a short window
///
/// Output is stored as FootContact per frame in the AnimClip.
class FootContactDetector {
public:
    explicit FootContactDetector(const FootContactDetectorConfig& config = {});
    ~FootContactDetector();

    FootContactDetector(const FootContactDetector&) = delete;
    FootContactDetector& operator=(const FootContactDetector&) = delete;
    FootContactDetector(FootContactDetector&&) noexcept;
    FootContactDetector& operator=(FootContactDetector&&) noexcept;

    /// Detect foot contacts for a sequence of skeleton frames.
    std::vector<FootContact> detect(const std::vector<SkeletonFrame>& frames) const;

    /// Detect and store foot contacts directly into an AnimClip.
    void process(AnimClip& clip) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::motion
