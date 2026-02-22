#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/ml/OnnxInference.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace hm::segmenter {

// =========================================================================
// ONNX inference version — used by MotionSegmenter at runtime
// =========================================================================

/// ONNX-based TCN classifier for per-frame motion classification.
class TemporalConvNetOnnx {
public:
    TemporalConvNetOnnx();
    ~TemporalConvNetOnnx();

    TemporalConvNetOnnx(const TemporalConvNetOnnx&) = delete;
    TemporalConvNetOnnx& operator=(const TemporalConvNetOnnx&) = delete;
    TemporalConvNetOnnx(TemporalConvNetOnnx&&) noexcept;
    TemporalConvNetOnnx& operator=(TemporalConvNetOnnx&&) noexcept;

    bool load(const std::string& onnxPath, bool useGPU = true);
    bool isLoaded() const;

    /// Classify a sequence of feature vectors.
    /// @param features  [numFrames][70] from MotionFeatureExtractor.
    /// @return Per-frame logits [numFrames][MOTION_TYPE_COUNT].
    std::vector<std::array<float, MOTION_TYPE_COUNT>> classify(
        const std::vector<std::array<float, 70>>& features);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// =========================================================================
// LibTorch training version — used by hm_train CLI
// =========================================================================

#ifdef HM_HAS_TORCH
#include <torch/torch.h>

/// Temporal causal conv block with residual connection.
struct TemporalBlockImpl : torch::nn::Module {
    TemporalBlockImpl(int inChannels, int outChannels, int kernelSize,
                       int dilation, float dropout);
    torch::Tensor forward(torch::Tensor x);

private:
    int padding_;
    torch::nn::Conv1d conv1_{nullptr}, conv2_{nullptr};
    torch::nn::BatchNorm1d bn1_{nullptr}, bn2_{nullptr};
    torch::nn::Dropout drop_{nullptr};
    torch::nn::Conv1d residual_{nullptr};  // 1x1 conv when channels differ
    bool needsResidual_ = false;
};

TORCH_MODULE(TemporalBlock);

/// 6-layer dilated causal TCN for motion classification (~600K params).
///
/// Dilations: [1, 2, 4, 8, 16, 32], hidden: 128, kernel: 3.
/// Output: Conv1d(hidden, numClasses, 1) per-frame logits.
struct TemporalConvNetImpl : torch::nn::Module {
    TemporalConvNetImpl(int inputDim, int hiddenDim, int numClasses);

    /// @param x [B, inputDim, T]
    /// @return  [B, numClasses, T]
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::ModuleList blocks_{nullptr};
    torch::nn::Conv1d output_{nullptr};
};

TORCH_MODULE(TemporalConvNet);

#endif  // HM_HAS_TORCH

} // namespace hm::segmenter
