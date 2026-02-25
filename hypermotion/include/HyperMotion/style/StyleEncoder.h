#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/ml/OnnxInference.h"
#include <array>
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace hm::style {

static constexpr int STYLE_INPUT_DIM = 201;  // 132 + 3 + 66

// =========================================================================
// ONNX inference version -- used at runtime
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
// LibTorch training version -- used by hm_style CLI
// =========================================================================

#ifdef HM_HAS_TORCH
#include <torch/torch.h>

/// Per-layer summary entry for model introspection.
struct LayerSummary {
    std::string name;
    std::string type;
    int64_t numParameters = 0;
    int64_t numTrainable = 0;
    std::vector<int64_t> outputShape;  ///< Approximate output shape (may be empty).
};

/// Complete model summary with parameter counts and layer info.
struct ModelSummary {
    std::vector<LayerSummary> layers;
    int64_t totalParameters = 0;
    int64_t trainableParameters = 0;
    int64_t nonTrainableParameters = 0;

    /// Format as a human-readable string table.
    std::string toString() const;
};

/// Intermediate feature maps extracted during a forward pass, useful for
/// visualization and analysis of learned representations.
struct IntermediateFeatures {
    torch::Tensor afterInputConv;    ///< [B, 128, T] after initial conv + BN + ReLU
    torch::Tensor afterResBlock0;    ///< [B, 128, T]
    torch::Tensor afterResBlock1;    ///< [B, 256, T]
    torch::Tensor afterResBlock2;    ///< [B, 256, T]
    torch::Tensor afterResBlock3;    ///< [B, 512, T]
    torch::Tensor afterGAP;          ///< [B, 512] after global average pooling
    torch::Tensor afterFC1;          ///< [B, 256] after fc1 + ReLU
    torch::Tensor finalEmbedding;    ///< [B, 64] L2-normalized output
};

/// Weight initialization strategy for the encoder.
enum class WeightInitStrategy {
    KaimingNormal,    ///< Kaiming He (fan_out, relu) -- default for conv layers.
    KaimingUniform,   ///< Kaiming He (uniform variant).
    XavierNormal,     ///< Glorot normal (suitable for linear layers without ReLU).
    XavierUniform,    ///< Glorot uniform.
    Default           ///< PyTorch default initialization (no explicit init).
};

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

    /// Forward pass that also captures intermediate activations.
    /// @param x [B, inputDim, T]
    /// @param[out] features  filled with activations from each layer
    /// @return  [B, styleDim]  L2-normalized embedding
    torch::Tensor forwardWithIntermediates(torch::Tensor x, IntermediateFeatures& features);

    /// Build a tensor from skeleton frames: returns [1, STYLE_INPUT_DIM, T].
    static torch::Tensor prepareInput(const std::vector<SkeletonFrame>& frames);

    /// Apply weight initialization strategy to all layers.
    void initializeWeights(WeightInitStrategy strategy = WeightInitStrategy::KaimingNormal);

    /// Set the encoder into inference mode (eval + no_grad-friendly).
    /// Freezes batch norm running stats, disables dropout.
    void setInferenceMode(bool enabled);

    /// Check if the encoder is currently in inference mode.
    bool isInferenceMode() const { return inferenceMode_; }

    /// Compute and return a structured summary of all layers and parameter counts.
    ModelSummary summary() const;

    /// Convenience: print summary to the logger.
    void printSummary() const;

    /// Get the total number of trainable parameters.
    int64_t countTrainableParameters() const;

    /// Get the total number of parameters (trainable + frozen).
    int64_t countTotalParameters() const;

    /// Compute per-channel input feature normalization statistics from a dataset.
    /// @param dataLoader  iterable that yields [B, inputDim, T] tensors
    /// @param maxBatches  maximum number of batches to sample (0 = all)
    void computeInputNormalization(const std::vector<torch::Tensor>& samples,
                                   int maxBatches = 0);

    /// Apply precomputed input normalization to a tensor.
    /// @param x [B, inputDim, T]
    /// @return normalized tensor
    torch::Tensor normalizeInput(torch::Tensor x) const;

    /// Check whether input normalization stats have been computed.
    bool hasInputNormalization() const { return hasInputNorm_; }

    /// Export per-channel mean and std (for use in ONNX export preprocessing).
    std::pair<std::vector<float>, std::vector<float>> getInputNormStats() const;

private:
    torch::nn::Conv1d inputConv_{nullptr};
    torch::nn::BatchNorm1d inputBN_{nullptr};
    torch::nn::ModuleList resBlocks_{nullptr};
    torch::nn::Linear fc1_{nullptr}, fc2_{nullptr};

    bool inferenceMode_ = false;

    // Per-channel input normalization: mean and std, each [inputDim]
    torch::Tensor inputMean_;
    torch::Tensor inputStd_;
    bool hasInputNorm_ = false;
    int inputDim_ = STYLE_INPUT_DIM;
    int styleDim_ = STYLE_DIM;
};

TORCH_MODULE(StyleEncoder);

#endif  // HM_HAS_TORCH

} // namespace hm::style
