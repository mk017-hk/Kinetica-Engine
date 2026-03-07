#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/pose/DepthLifter.h"
#include <memory>
#include <string>
#include <vector>

namespace hm::pose {

/// Configuration for the PoseLifter abstraction.
struct PoseLifterConfig {
    DepthLifterConfig depthConfig;
    float minConfidenceThreshold = 0.1f;
};

/// Abstraction layer for 2D-to-3D pose lifting.
///
/// Wraps DepthLifter and provides a clean interface for converting
/// Pose2D sequences to Pose3D sequences.  Future backends (e.g. learned
/// temporal lifters) can be swapped in behind this interface.
class PoseLifter {
public:
    explicit PoseLifter(const PoseLifterConfig& config = {});
    ~PoseLifter();

    PoseLifter(const PoseLifter&) = delete;
    PoseLifter& operator=(const PoseLifter&) = delete;
    PoseLifter(PoseLifter&&) noexcept;
    PoseLifter& operator=(PoseLifter&&) noexcept;

    bool initialize();
    bool isInitialized() const;

    /// Lift a single Pose2D to Pose3D.
    Pose3D lift(const Pose2D& pose2D, const BBox& bbox);

    /// Lift a sequence of Pose2D frames to Pose3D.
    std::vector<Pose3D> liftSequence(
        const std::vector<Pose2D>& poses,
        const std::vector<BBox>& bboxes);

    /// Access the underlying DepthLifter.
    DepthLifter& depthLifter();
    const DepthLifter& depthLifter() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::pose
