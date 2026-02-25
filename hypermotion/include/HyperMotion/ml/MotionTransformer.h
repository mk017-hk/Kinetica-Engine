#pragma once

#include "HyperMotion/core/Types.h"

#ifdef HM_HAS_TORCH
#include <torch/torch.h>

namespace hm::ml {

/// Configuration for the MotionTransformer.
struct MotionTransformerConfig {
    int motionDim = FRAME_DIM;   // 132
    int condDim = 256;           // Encoded condition dimension (output of ConditionEncoder)
    int modelDim = 512;          // Transformer hidden dimension
    int numHeads = 8;            // Number of attention heads
    int numLayers = 8;           // Number of transformer layers
    int ffnDim = 2048;           // Feed-forward network dimension
    float dropout = 0.1f;        // Dropout rate
    int maxSeqLen = 256;         // Maximum sequence length for positional encoding
    bool useGradientCheckpointing = false;  // Memory-efficient training via checkpointing
    bool useLearnedPosEncoding = false;     // Learned vs sinusoidal positional encoding
};

/// Pre-norm transformer encoder block with pre-LayerNorm and residual connections.
struct TransformerBlockImpl : torch::nn::Module {
    TransformerBlockImpl(int modelDim, int numHeads, int ffnDim, float dropout);

    torch::Tensor forward(torch::Tensor x);

    /// Forward with optional attention mask.
    torch::Tensor forward(torch::Tensor x, const torch::Tensor& attnMask);

private:
    torch::nn::LayerNorm ln1_{nullptr}, ln2_{nullptr};
    torch::nn::MultiheadAttention attn_{nullptr};
    torch::nn::Linear ffn1_{nullptr}, ffn2_{nullptr};
    torch::nn::Dropout drop1_{nullptr}, drop2_{nullptr};
};

TORCH_MODULE(TransformerBlock);

/// Motion denoiser transformer (~17.6M parameters).
///
/// Architecture:
///   Input:     Linear(motionDim, modelDim)
///   PosEnc:    Sinusoidal or learned positional encoding [maxSeqLen, modelDim]
///   Time:      Sinusoidal PE(modelDim) -> MLP(modelDim, modelDim)
///   Condition: Linear(condDim, modelDim)
///   Body:      8x pre-norm TransformerEncoder (8 heads, 512D, FFN 2048, GELU, dropout 0.1)
///   Output:    LayerNorm -> Linear(modelDim, motionDim)
///
/// Features:
///   - Sinusoidal or learned sequence positional encoding
///   - Gradient checkpointing support for memory-efficient training
///   - Separate encode/decode stages
///   - Parameter counting utility
struct MotionTransformerImpl : torch::nn::Module {
    /// Construct with full config.
    explicit MotionTransformerImpl(const MotionTransformerConfig& config = {});

    /// Legacy constructor for backward compatibility.
    MotionTransformerImpl(int motionDim, int condDim, int modelDim,
                           int numHeads, int numLayers, int ffnDim,
                           float dropout);

    /// Full forward: noisy motion + timestep + condition -> predicted noise.
    /// @param x    Noisy motion [B, seqLen, motionDim]
    /// @param t    Diffusion timestep [B] (int64)
    /// @param cond Encoded condition [B, condDim]
    /// @return     Predicted noise [B, seqLen, motionDim]
    torch::Tensor forward(torch::Tensor x, torch::Tensor t, torch::Tensor cond);

    /// Encode stage only: projects motion + adds time/cond embeddings,
    /// runs through transformer blocks.  Returns [B, seqLen, modelDim].
    torch::Tensor encode(torch::Tensor x, torch::Tensor t, torch::Tensor cond);

    /// Decode stage: final norm + output projection.
    /// Input: [B, seqLen, modelDim], Output: [B, seqLen, motionDim].
    torch::Tensor decode(torch::Tensor h);

    /// Total trainable parameter count.
    int64_t parameterCount() const;

    /// Detailed per-component parameter breakdown (for logging).
    std::string parameterBreakdown() const;

    /// Enable or disable gradient checkpointing at runtime.
    void setGradientCheckpointing(bool enable);
    bool gradientCheckpointingEnabled() const;

    /// Return the active config.
    const MotionTransformerConfig& config() const { return config_; }

private:
    torch::Tensor sinusoidalEmbedding(torch::Tensor t, int dim);
    torch::Tensor getPositionalEncoding(int seqLen, torch::Device device);

    MotionTransformerConfig config_;
    bool useGradCheckpoint_ = false;

    torch::nn::Linear inputProj_{nullptr};
    torch::nn::Linear timeMLPfc1_{nullptr}, timeMLPfc2_{nullptr};
    torch::nn::Linear condProj_{nullptr};
    torch::nn::ModuleList blocks_{nullptr};
    torch::nn::LayerNorm finalNorm_{nullptr};
    torch::nn::Linear outputProj_{nullptr};

    // Positional encoding (sinusoidal: buffer, learned: parameter)
    torch::Tensor posEncBuffer_;                     // Sinusoidal cache
    torch::nn::Embedding learnedPosEnc_{nullptr};    // Learned alternative
};

TORCH_MODULE(MotionTransformer);

} // namespace hm::ml

#else

namespace hm::ml {
// MotionTransformer requires LibTorch for training.
// At inference time it is fused into the ONNX denoiser graph.
} // namespace hm::ml

#endif
