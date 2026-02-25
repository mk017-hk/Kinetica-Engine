#pragma once

#include "HyperMotion/core/Types.h"

#ifdef HM_HAS_TORCH
#include <torch/torch.h>

namespace hm::ml {

/// Configuration for the ConditionEncoder network.
struct ConditionEncoderConfig {
    int inputDim = MotionCondition::DIM;   // 78
    int hiddenDim = 512;
    int outputDim = 256;
    float dropoutRate = 0.1f;
    bool useBatchNorm = true;
    bool useLayerNorm = false;              // Alternative to BN for small batches

    enum class InitMethod {
        Xavier,
        Kaiming,
        Orthogonal
    };
    InitMethod initMethod = InitMethod::Kaiming;
};

/// Condition encoder: 78D MotionCondition -> 256D latent.
/// Architecture: Linear(78,512) -> [BN] -> ReLU -> Dropout ->
///               Linear(512,512) -> [BN] -> ReLU -> Dropout ->
///               Linear(512,256)
///
/// Features:
///   - Optional batch normalisation between layers
///   - Dropout for regularisation during training
///   - Configurable weight initialisation (Xavier, Kaiming, Orthogonal)
///   - Factory method with config validation
struct ConditionEncoderImpl : torch::nn::Module {
    /// Construct with full configuration.
    explicit ConditionEncoderImpl(const ConditionEncoderConfig& config = {});

    /// Legacy constructor for backward compatibility.
    ConditionEncoderImpl(int inputDim, int hiddenDim, int outputDim);

    /// Forward pass.  Input: [B, inputDim], Output: [B, outputDim].
    torch::Tensor forward(torch::Tensor x);

    /// Forward with explicit training flag (useful when BN is enabled).
    torch::Tensor forward(torch::Tensor x, bool isTrain);

    /// Factory: create a validated ConditionEncoder from config.
    /// Returns nullptr on validation failure (logs error).
    static ConditionEncoder create(const ConditionEncoderConfig& config);

    /// Return total trainable parameter count.
    int64_t parameterCount() const;

    /// Return the active configuration.
    const ConditionEncoderConfig& config() const { return config_; }

private:
    void initWeights();
    void buildLayers();

    ConditionEncoderConfig config_;

    torch::nn::Linear fc1_{nullptr}, fc2_{nullptr}, fc3_{nullptr};
    torch::nn::BatchNorm1d bn1_{nullptr}, bn2_{nullptr};
    torch::nn::LayerNorm ln1_{nullptr}, ln2_{nullptr};
    torch::nn::Dropout drop1_{nullptr}, drop2_{nullptr};
};

TORCH_MODULE(ConditionEncoder);

} // namespace hm::ml

#else

namespace hm::ml {
// ConditionEncoder requires LibTorch for training.
// At inference time it is fused into the ONNX denoiser graph.
} // namespace hm::ml

#endif
