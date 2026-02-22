#include "HyperMotion/ml/MotionTransformer.h"

#ifdef HM_HAS_TORCH

#include <cmath>

namespace hm::ml {

// -----------------------------------------------------------------------
// TransformerBlock
// -----------------------------------------------------------------------

TransformerBlockImpl::TransformerBlockImpl(int modelDim, int numHeads,
                                            int ffnDim, float dropout) {
    ln1_ = register_module("ln1", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({modelDim})));
    ln2_ = register_module("ln2", torch::nn::LayerNorm(
        torch::nn::LayerNormOptions({modelDim})));

    attn_ = register_module("attn", torch::nn::MultiheadAttention(
        torch::nn::MultiheadAttentionOptions(modelDim, numHeads)
            .dropout(dropout)
            .batch_first(true)));

    ffn1_ = register_module("ffn1", torch::nn::Linear(modelDim, ffnDim));
    ffn2_ = register_module("ffn2", torch::nn::Linear(ffnDim, modelDim));
    drop1_ = register_module("drop1", torch::nn::Dropout(dropout));
    drop2_ = register_module("drop2", torch::nn::Dropout(dropout));
}

torch::Tensor TransformerBlockImpl::forward(torch::Tensor x) {
    // Pre-norm self-attention
    auto normed = ln1_->forward(x);
    auto [attnOut, _] = attn_->forward(normed, normed, normed);
    x = x + drop1_->forward(attnOut);

    // Pre-norm feed-forward (GELU activation)
    normed = ln2_->forward(x);
    auto ff = drop2_->forward(ffn2_->forward(torch::gelu(ffn1_->forward(normed))));
    x = x + ff;

    return x;
}

// -----------------------------------------------------------------------
// MotionTransformer
// -----------------------------------------------------------------------

MotionTransformerImpl::MotionTransformerImpl(int motionDim, int condDim,
                                              int modelDim, int numHeads,
                                              int numLayers, int ffnDim,
                                              float dropout)
    : modelDim_(modelDim) {
    inputProj_ = register_module("input_proj",
        torch::nn::Linear(motionDim, modelDim));

    // Timestep embedding MLP: sinusoidal(modelDim) -> modelDim
    timeMLPfc1_ = register_module("time_mlp_fc1",
        torch::nn::Linear(modelDim, modelDim));
    timeMLPfc2_ = register_module("time_mlp_fc2",
        torch::nn::Linear(modelDim, modelDim));

    condProj_ = register_module("cond_proj",
        torch::nn::Linear(condDim, modelDim));

    blocks_ = register_module("blocks", torch::nn::ModuleList());
    for (int i = 0; i < numLayers; ++i) {
        blocks_->push_back(TransformerBlock(modelDim, numHeads, ffnDim, dropout));
    }

    finalNorm_ = register_module("final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({modelDim})));
    outputProj_ = register_module("output_proj",
        torch::nn::Linear(modelDim, motionDim));
}

torch::Tensor MotionTransformerImpl::sinusoidalEmbedding(torch::Tensor t, int dim) {
    int halfDim = dim / 2;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(t.device());
    auto freqs = torch::exp(
        -std::log(10000.0) * torch::arange(halfDim, opts) /
        static_cast<float>(halfDim));

    // t: [B] -> [B,1], freqs: [halfDim] -> [1,halfDim]
    auto angles = t.unsqueeze(-1).to(torch::kFloat32) * freqs.unsqueeze(0);
    return torch::cat({torch::sin(angles), torch::cos(angles)}, /*dim=*/-1);  // [B, dim]
}

torch::Tensor MotionTransformerImpl::forward(torch::Tensor x,
                                              torch::Tensor t,
                                              torch::Tensor cond) {
    // x:    [B, S, motionDim]
    // t:    [B]  (int64 timesteps)
    // cond: [B, condDim]

    // Project input motion
    auto h = inputProj_->forward(x);  // [B, S, modelDim]

    // Timestep embedding: sinusoidal -> MLP
    auto tEmb = sinusoidalEmbedding(t, modelDim_);  // [B, modelDim]
    tEmb = torch::gelu(timeMLPfc1_->forward(tEmb));
    tEmb = timeMLPfc2_->forward(tEmb);              // [B, modelDim]

    // Condition projection
    auto cEmb = condProj_->forward(cond);            // [B, modelDim]

    // Add time + condition as bias to every token
    h = h + tEmb.unsqueeze(1) + cEmb.unsqueeze(1);  // [B, S, modelDim]

    // Transformer blocks
    for (const auto& block : *blocks_) {
        h = block->as<TransformerBlockImpl>()->forward(h);
    }

    // Final norm + output projection
    h = finalNorm_->forward(h);
    return outputProj_->forward(h);  // [B, S, motionDim]
}

} // namespace hm::ml

#endif
