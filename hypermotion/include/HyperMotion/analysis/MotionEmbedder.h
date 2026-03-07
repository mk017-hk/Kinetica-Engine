#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/ml/OnnxInference.h"
#include <array>
#include <string>
#include <vector>
#include <memory>

namespace hm::analysis {

/// Configuration for the motion embedder.
struct MotionEmbedderConfig {
    std::string onnxModelPath;   // path to motion_encoder.onnx
    bool useGPU = true;
    int seqLen = 64;             // expected input sequence length
    float fps = 30.0f;
};

/// ONNX-based motion embedder: converts animation clips into 128D embeddings.
///
/// Wraps the motion encoder ONNX model trained by the Python pipeline.
/// Input: joint world positions [seq_len, 66] (22 joints * 3D)
/// Output: 128D L2-normalized motion embedding
class MotionEmbedder {
public:
    explicit MotionEmbedder(const MotionEmbedderConfig& config = {});
    ~MotionEmbedder();

    MotionEmbedder(const MotionEmbedder&) = delete;
    MotionEmbedder& operator=(const MotionEmbedder&) = delete;
    MotionEmbedder(MotionEmbedder&&) noexcept;
    MotionEmbedder& operator=(MotionEmbedder&&) noexcept;

    /// Load the ONNX model. Returns false if model not found.
    bool initialize();
    bool isInitialized() const;

    /// Compute embedding for a sequence of skeleton frames.
    std::array<float, MOTION_EMBEDDING_DIM> embed(
        const std::vector<SkeletonFrame>& frames) const;

    /// Compute embeddings for an animation clip.
    std::array<float, MOTION_EMBEDDING_DIM> embedClip(const AnimClip& clip) const;

    /// Compute embeddings for a batch of clips.
    std::vector<std::array<float, MOTION_EMBEDDING_DIM>> embedBatch(
        const std::vector<AnimClip>& clips) const;

    /// Prepare the 66D-per-frame input from skeleton frames
    /// (normalised joint world positions).
    static std::vector<float> prepareInput(
        const std::vector<SkeletonFrame>& frames, int seqLen);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::analysis
