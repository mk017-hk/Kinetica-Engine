#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/ml/OnnxInference.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace hm::style {

static constexpr int STYLE_INPUT_DIM = 201;  // 132 + 3 + 66

// =========================================================================
// ONNX inference version — used at runtime
// =========================================================================

/// ONNX-based style encoder: variable-length motion -> 64D embedding.
class StyleEncoderOnnx {
public:
    StyleEncoderOnnx();
    ~StyleEncoderOnnx();

    StyleEncoderOnnx(const StyleEncoderOnnx&) = delete;
    StyleEncoderOnnx& operator=(const StyleEncoderOnnx&) = delete;
    StyleEncoderOnnx(StyleEncoderOnnx&&) noexcept;
    StyleEncoderOnnx& operator=(StyleEncoderOnnx&&) noexcept;

    bool load(const std::string& onnxPath, bool useGPU = true);
    bool isLoaded() const;

    /// Encode a motion clip to a 64D style embedding (L2 normalized).
    std::array<float, STYLE_DIM> encode(const std::vector<SkeletonFrame>& frames);

    /// Prepare the 201D-per-frame feature matrix from skeleton frames.
    static std::vector<std::array<float, STYLE_INPUT_DIM>>
    prepareInput(const std::vector<SkeletonFrame>& frames);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// =========================================================================
// LibTorch training version — used by hm_style CLI
// =========================================================================

#ifdef HM_HAS_TORCH
#include <torch/torch.h>

/// 1D residual block for the style encoder.
struct ResBlock1DImpl : torch::nn::Module {
    ResBlock1DImpl(int inChannels, int outChannels);
    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Conv1d conv1_{nullptr}, conv2_{nullptr};
    torch::nn::BatchNorm1d bn1_{nullptr}, bn2_{nullptr};
    torch::nn::Conv1d downsample_{nullptr};
    bool needsDownsample_ = false;
};

TORCH_MODULE(ResBlock1D);

/// Style encoder (~1.9M parameters).
///
/// Conv1D(201,128) -> 4 ResBlocks (128->128->256->256->512)
/// -> GAP -> Linear(512,256) -> ReLU -> Linear(256,64) -> L2 norm
struct StyleEncoderImpl : torch::nn::Module {
    StyleEncoderImpl(int inputDim = STYLE_INPUT_DIM, int styleDim = STYLE_DIM);

    /// @param x [B, inputDim, T]  (Conv1D format)
    /// @return  [B, styleDim]  L2-normalized embedding
    torch::Tensor forward(torch::Tensor x);

    /// Build a tensor from skeleton frames: returns [1, STYLE_INPUT_DIM, T].
    static torch::Tensor prepareInput(const std::vector<SkeletonFrame>& frames);

private:
    torch::nn::Conv1d inputConv_{nullptr};
    torch::nn::BatchNorm1d inputBN_{nullptr};
    torch::nn::ModuleList resBlocks_{nullptr};
    torch::nn::Linear fc1_{nullptr}, fc2_{nullptr};
};

TORCH_MODULE(StyleEncoder);

#endif  // HM_HAS_TORCH

} // namespace hm::style
