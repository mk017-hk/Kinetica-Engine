#pragma once

#include "HyperMotion/core/Types.h"

#ifdef HM_HAS_TORCH
#include <torch/torch.h>

namespace hm::ml {

/// Pre-norm transformer encoder block.
struct TransformerBlockImpl : torch::nn::Module {
    TransformerBlockImpl(int modelDim, int numHeads, int ffnDim, float dropout);
    torch::Tensor forward(torch::Tensor x);

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
///   Input:  Linear(motionDim, modelDim)
///   Time:   sinusoidal PE(modelDim) -> MLP(modelDim, modelDim)
///   Cond:   Linear(condDim, modelDim)
///   Body:   8x pre-norm TransformerEncoder (8 heads, 512D, FFN 2048, GELU, dropout 0.1)
///   Output: LayerNorm -> Linear(modelDim, motionDim)
struct MotionTransformerImpl : torch::nn::Module {
    MotionTransformerImpl(int motionDim = FRAME_DIM,
                           int condDim = 256,
                           int modelDim = 512,
                           int numHeads = 8,
                           int numLayers = 8,
                           int ffnDim = 2048,
                           float dropout = 0.1f);

    /// @param x    Noisy motion [B, seqLen, motionDim]
    /// @param t    Diffusion timestep [B] (int64)
    /// @param cond Encoded condition [B, condDim]
    /// @return Predicted noise [B, seqLen, motionDim]
    torch::Tensor forward(torch::Tensor x, torch::Tensor t, torch::Tensor cond);

private:
    torch::Tensor sinusoidalEmbedding(torch::Tensor t, int dim);

    int modelDim_;
    torch::nn::Linear inputProj_{nullptr};
    torch::nn::Linear timeMLPfc1_{nullptr}, timeMLPfc2_{nullptr};
    torch::nn::Linear condProj_{nullptr};
    torch::nn::ModuleList blocks_{nullptr};
    torch::nn::LayerNorm finalNorm_{nullptr};
    torch::nn::Linear outputProj_{nullptr};
};

TORCH_MODULE(MotionTransformer);

} // namespace hm::ml

#else

namespace hm::ml {
// MotionTransformer requires LibTorch for training.
// At inference time it is fused into the ONNX denoiser graph.
} // namespace hm::ml

#endif
