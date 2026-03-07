#pragma once

#include "HyperMotion/core/Types.h"
#include <array>
#include <vector>
#include <memory>

namespace hm::analysis {

/// Configuration for motion interpolation.
struct MotionInterpolatorConfig {
    bool normalizeOutput = true;  // L2-normalize interpolated embeddings
};

/// Generates new motion embeddings by interpolating between existing ones.
///
/// Supports linear (LERP) and spherical (SLERP) interpolation in
/// the 128D embedding space. The interpolated embeddings can be used
/// to query the MotionSearch for blended animations.
class MotionInterpolator {
public:
    explicit MotionInterpolator(const MotionInterpolatorConfig& config = {});
    ~MotionInterpolator();

    MotionInterpolator(const MotionInterpolator&) = delete;
    MotionInterpolator& operator=(const MotionInterpolator&) = delete;
    MotionInterpolator(MotionInterpolator&&) noexcept;
    MotionInterpolator& operator=(MotionInterpolator&&) noexcept;

    /// Linear interpolation between two embeddings.
    /// t=0 returns a, t=1 returns b.
    std::array<float, MOTION_EMBEDDING_DIM> lerp(
        const std::array<float, MOTION_EMBEDDING_DIM>& a,
        const std::array<float, MOTION_EMBEDDING_DIM>& b,
        float t) const;

    /// Spherical linear interpolation between two L2-normalized embeddings.
    std::array<float, MOTION_EMBEDDING_DIM> slerp(
        const std::array<float, MOTION_EMBEDDING_DIM>& a,
        const std::array<float, MOTION_EMBEDDING_DIM>& b,
        float t) const;

    /// Generate a sequence of interpolated embeddings between two endpoints.
    /// Returns numSteps+1 embeddings including both endpoints.
    std::vector<std::array<float, MOTION_EMBEDDING_DIM>> interpolateSequence(
        const std::array<float, MOTION_EMBEDDING_DIM>& a,
        const std::array<float, MOTION_EMBEDDING_DIM>& b,
        int numSteps,
        bool useSlerp = true) const;

    /// Weighted average of multiple embeddings (for multi-way blending).
    std::array<float, MOTION_EMBEDDING_DIM> blend(
        const std::vector<std::array<float, MOTION_EMBEDDING_DIM>>& embeddings,
        const std::vector<float>& weights) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::analysis
