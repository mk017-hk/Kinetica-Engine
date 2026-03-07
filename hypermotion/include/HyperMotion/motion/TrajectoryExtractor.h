#pragma once

#include "HyperMotion/core/Types.h"
#include <vector>
#include <memory>

namespace hm::motion {

/// Configuration for trajectory extraction and prediction.
struct TrajectoryExtractorConfig {
    float predictionHorizon = 1.5f;   // seconds into the future
    float predictionStep = 0.5f;      // seconds between trajectory points (0.5, 1.0, 1.5)
    int velocitySmoothingWindow = 5;  // frames to average for velocity estimation
    float fps = 30.0f;               // frame rate
};

/// Extracts root motion trajectory and predicts future positions.
///
/// For each frame in a clip, computes:
///   - Root velocity (smoothed over a window)
///   - Direction of movement
///   - Predicted trajectory points at t+0.5s, t+1.0s, t+1.5s
///
/// Prediction uses constant-velocity extrapolation from the smoothed
/// velocity at the current frame. This is lightweight and suitable for
/// motion matching lookups.
class TrajectoryExtractor {
public:
    explicit TrajectoryExtractor(const TrajectoryExtractorConfig& config = {});
    ~TrajectoryExtractor();

    TrajectoryExtractor(const TrajectoryExtractor&) = delete;
    TrajectoryExtractor& operator=(const TrajectoryExtractor&) = delete;
    TrajectoryExtractor(TrajectoryExtractor&&) noexcept;
    TrajectoryExtractor& operator=(TrajectoryExtractor&&) noexcept;

    /// Extract per-frame trajectory predictions for a skeleton sequence.
    std::vector<std::vector<TrajectoryPoint>> extract(
        const std::vector<SkeletonFrame>& frames) const;

    /// Extract and store trajectories directly into an AnimClip.
    void process(AnimClip& clip) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::motion
