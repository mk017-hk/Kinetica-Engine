#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/ml/OnnxInference.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace hm::segmenter {

/// ONNX-based TCN classifier for per-frame motion classification.
/// The model is trained in Python and exported to ONNX.
class TemporalConvNet {
public:
    TemporalConvNet();
    ~TemporalConvNet();

    TemporalConvNet(const TemporalConvNet&) = delete;
    TemporalConvNet& operator=(const TemporalConvNet&) = delete;
    TemporalConvNet(TemporalConvNet&&) noexcept;
    TemporalConvNet& operator=(TemporalConvNet&&) noexcept;

    bool load(const std::string& onnxPath, bool useGPU = true);
    bool isLoaded() const;

    /// Classify a sequence of feature vectors.
    /// @param features  [numFrames][70] feature vectors from MotionFeatureExtractor.
    /// @return Per-frame logits [numFrames][MOTION_TYPE_COUNT].
    std::vector<std::array<float, MOTION_TYPE_COUNT>> classify(
        const std::vector<std::array<float, 70>>& features);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::segmenter
