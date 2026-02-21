#pragma once

#include "HyperMotion/core/Types.h"
#include <torch/torch.h>

namespace hm::ml {

// Motion Transformer for diffusion denoising
// Input: Linear(132, 512), timestep: sinusoidal PE(512) + MLP, condition: Linear(256, 512)
// 8-layer transformer encoder, 8 heads, 512D, FFN 2048, pre-norm, GELU, dropout 0.1
// Output: Linear(512, 132)
// ~17.6M parameters

struct SinusoidalPEImpl : torch::nn::Module {
    int dim;
    SinusoidalPEImpl(int dim);
    torch::Tensor forward(torch::Tensor timesteps);  // [batch] -> [batch, dim]
};
TORCH_MODULE(SinusoidalPE);

struct TimestepEmbedderImpl : torch::nn::Module {
    SinusoidalPE sinPE{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};

    TimestepEmbedderImpl(int dim = 512);
    torch::Tensor forward(torch::Tensor timesteps);  // [batch] -> [batch, dim]
};
TORCH_MODULE(TimestepEmbedder);

struct PreNormTransformerLayerImpl : torch::nn::Module {
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    torch::nn::MultiheadAttention attn{nullptr};
    torch::nn::Linear ff1{nullptr}, ff2{nullptr};
    torch::nn::Dropout dropout{nullptr};

    PreNormTransformerLayerImpl(int dim = 512, int nHeads = 8,
                                 int ffDim = 2048, float dropoutRate = 0.1f);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(PreNormTransformerLayer);

struct MotionTransformerImpl : torch::nn::Module {
    torch::nn::Linear input_proj{nullptr};
    torch::nn::Linear cond_proj{nullptr};
    torch::nn::Linear output_proj{nullptr};
    TimestepEmbedder timestep_embedder{nullptr};
    std::vector<PreNormTransformerLayer> layers;
    torch::nn::LayerNorm final_norm{nullptr};

    MotionTransformerImpl(int motionDim = FRAME_DIM,
                           int condDim = 256,
                           int modelDim = 512,
                           int nHeads = 8,
                           int nLayers = 8,
                           int ffDim = 2048,
                           float dropout = 0.1f);

    // x: [batch, seq_len, 132], t: [batch], cond: [batch, 256]
    // Returns: [batch, seq_len, 132]
    torch::Tensor forward(torch::Tensor x, torch::Tensor t, torch::Tensor cond);
};
TORCH_MODULE(MotionTransformer);

} // namespace hm::ml
