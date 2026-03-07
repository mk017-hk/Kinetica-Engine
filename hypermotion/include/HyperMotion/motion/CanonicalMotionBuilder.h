#pragma once

#include "HyperMotion/core/Types.h"
#include <memory>
#include <vector>

namespace hm::motion {

/// Configuration for the canonical motion builder.
struct CanonicalMotionBuilderConfig {
    /// Target limb lengths (cm) for the canonical skeleton.
    /// Index by Joint enum.  Zero means "use rest-pose default".
    std::array<float, JOINT_COUNT> targetLimbLengths{};

    /// Whether to stabilise limb lengths across the sequence.
    bool stabiliseLimbLengths = true;

    /// EMA alpha for limb-length smoothing (lower = more stable).
    float limbLengthAlpha = 0.05f;

    /// Whether to separate root motion from local joint transforms.
    bool extractRootMotion = true;

    /// Whether to solve root orientation from the hip-to-spine direction.
    bool solveRootOrientation = true;

    /// Height of the ground plane (Y-up coordinate system, cm).
    float groundHeight = 0.0f;
};

/// Per-frame canonical representation: local-space joint transforms with
/// root motion stored separately.
struct CanonicalFrame {
    /// Root position in world space (for root motion).
    Vec3 rootPosition;

    /// Root rotation in world space.
    Quat rootRotation = Quat::identity();

    /// Root velocity in world space (cm/s).
    Vec3 rootVelocity;

    /// Root angular velocity (rad/s around Y axis).
    float rootAngularVelocity = 0.0f;

    /// Local-space joint rotations relative to parent.
    std::array<Quat, JOINT_COUNT> localRotations;

    /// Local-space joint positions relative to parent (canonical bone lengths).
    std::array<Vec3, JOINT_COUNT> localPositions;

    /// Joint confidence values (propagated from input).
    std::array<float, JOINT_COUNT> confidence{};

    double timestamp = 0.0;
    int frameIndex = 0;
};

/// Complete canonical motion sequence for one player.
struct CanonicalMotion {
    int trackingID = -1;
    float fps = 30.0f;
    std::vector<CanonicalFrame> frames;

    /// Root motion trajectory: positions extracted from rootPosition.
    std::vector<Vec3> rootTrajectory;

    /// Measured limb lengths from the original data (before stabilisation).
    std::array<float, JOINT_COUNT> measuredLimbLengths{};

    /// Final canonical limb lengths used for the output.
    std::array<float, JOINT_COUNT> canonicalLimbLengths{};
};

/// Converts raw SkeletonFrame sequences into a canonical representation with:
///   - Fixed internal skeleton proportions (stabilised limb lengths)
///   - Joint transforms in local space (relative to parent)
///   - Root motion stored separately (position + orientation)
///   - Root orientation solved from hip-to-spine direction
///
/// This canonical form is the standard motion format consumed by downstream
/// modules (segmenter, fingerprinting, export, ML training).
class CanonicalMotionBuilder {
public:
    explicit CanonicalMotionBuilder(const CanonicalMotionBuilderConfig& config = {});
    ~CanonicalMotionBuilder();

    CanonicalMotionBuilder(const CanonicalMotionBuilder&) = delete;
    CanonicalMotionBuilder& operator=(const CanonicalMotionBuilder&) = delete;
    CanonicalMotionBuilder(CanonicalMotionBuilder&&) noexcept;
    CanonicalMotionBuilder& operator=(CanonicalMotionBuilder&&) noexcept;

    /// Build canonical motion from a sequence of skeleton frames.
    CanonicalMotion build(const std::vector<SkeletonFrame>& frames,
                          int trackingID = -1, float fps = 30.0f) const;

    /// Convert a canonical motion back to SkeletonFrame format.
    /// This applies forward kinematics using canonical limb lengths.
    std::vector<SkeletonFrame> toSkeletonFrames(const CanonicalMotion& motion) const;

    /// Convert an AnimClip to canonical form and back, updating the clip
    /// in place with the stabilised skeleton.
    void process(AnimClip& clip) const;

    /// Measure limb lengths from a sequence of skeleton frames.
    /// Returns median limb length per joint across all frames.
    static std::array<float, JOINT_COUNT> measureLimbLengths(
        const std::vector<SkeletonFrame>& frames);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::motion
