#include "HyperMotion/analysis/MotionInterpolator.h"
#include "HyperMotion/core/Logger.h"

#include <cmath>
#include <algorithm>
#include <numeric>

namespace hm::analysis {

static constexpr const char* TAG = "MotionInterpolator";

struct MotionInterpolator::Impl {
    MotionInterpolatorConfig config;

    // L2 normalize an embedding in-place.
    static void l2Normalize(std::array<float, MOTION_EMBEDDING_DIM>& v) {
        float norm = 0.0f;
        for (float x : v) norm += x * x;
        norm = std::sqrt(norm);
        if (norm > 1e-8f) {
            for (float& x : v) x /= norm;
        }
    }

    // Dot product of two embeddings.
    static float dot(const std::array<float, MOTION_EMBEDDING_DIM>& a,
                     const std::array<float, MOTION_EMBEDDING_DIM>& b) {
        float d = 0.0f;
        for (int i = 0; i < MOTION_EMBEDDING_DIM; ++i) d += a[i] * b[i];
        return d;
    }
};

MotionInterpolator::MotionInterpolator(const MotionInterpolatorConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

MotionInterpolator::~MotionInterpolator() = default;
MotionInterpolator::MotionInterpolator(MotionInterpolator&&) noexcept = default;
MotionInterpolator& MotionInterpolator::operator=(MotionInterpolator&&) noexcept = default;

std::array<float, MOTION_EMBEDDING_DIM> MotionInterpolator::lerp(
    const std::array<float, MOTION_EMBEDDING_DIM>& a,
    const std::array<float, MOTION_EMBEDDING_DIM>& b,
    float t) const {

    t = std::clamp(t, 0.0f, 1.0f);
    std::array<float, MOTION_EMBEDDING_DIM> result;
    for (int i = 0; i < MOTION_EMBEDDING_DIM; ++i) {
        result[i] = a[i] * (1.0f - t) + b[i] * t;
    }
    if (impl_->config.normalizeOutput) {
        Impl::l2Normalize(result);
    }
    return result;
}

std::array<float, MOTION_EMBEDDING_DIM> MotionInterpolator::slerp(
    const std::array<float, MOTION_EMBEDDING_DIM>& a,
    const std::array<float, MOTION_EMBEDDING_DIM>& b,
    float t) const {

    t = std::clamp(t, 0.0f, 1.0f);

    float dotProd = Impl::dot(a, b);
    dotProd = std::clamp(dotProd, -1.0f, 1.0f);

    float theta = std::acos(dotProd);

    // Fall back to lerp for very small angles
    if (theta < 1e-6f) {
        return lerp(a, b, t);
    }

    float sinTheta = std::sin(theta);
    float wa = std::sin((1.0f - t) * theta) / sinTheta;
    float wb = std::sin(t * theta) / sinTheta;

    std::array<float, MOTION_EMBEDDING_DIM> result;
    for (int i = 0; i < MOTION_EMBEDDING_DIM; ++i) {
        result[i] = wa * a[i] + wb * b[i];
    }

    if (impl_->config.normalizeOutput) {
        Impl::l2Normalize(result);
    }
    return result;
}

std::vector<std::array<float, MOTION_EMBEDDING_DIM>>
MotionInterpolator::interpolateSequence(
    const std::array<float, MOTION_EMBEDDING_DIM>& a,
    const std::array<float, MOTION_EMBEDDING_DIM>& b,
    int numSteps,
    bool useSlerp) const {

    numSteps = std::max(1, numSteps);
    std::vector<std::array<float, MOTION_EMBEDDING_DIM>> result;
    result.reserve(numSteps + 1);

    for (int i = 0; i <= numSteps; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(numSteps);
        if (useSlerp) {
            result.push_back(slerp(a, b, t));
        } else {
            result.push_back(lerp(a, b, t));
        }
    }

    return result;
}

std::array<float, MOTION_EMBEDDING_DIM> MotionInterpolator::blend(
    const std::vector<std::array<float, MOTION_EMBEDDING_DIM>>& embeddings,
    const std::vector<float>& weights) const {

    std::array<float, MOTION_EMBEDDING_DIM> result{};

    if (embeddings.empty() || weights.empty()) return result;

    // Normalize weights to sum to 1
    float weightSum = 0.0f;
    for (float w : weights) weightSum += w;
    if (weightSum < 1e-8f) return result;

    size_t count = std::min(embeddings.size(), weights.size());
    for (size_t j = 0; j < count; ++j) {
        float w = weights[j] / weightSum;
        for (int i = 0; i < MOTION_EMBEDDING_DIM; ++i) {
            result[i] += embeddings[j][i] * w;
        }
    }

    if (impl_->config.normalizeOutput) {
        Impl::l2Normalize(result);
    }
    return result;
}

} // namespace hm::analysis
