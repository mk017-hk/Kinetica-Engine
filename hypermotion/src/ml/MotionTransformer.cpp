#include "HyperMotion/ml/MotionTransformer.h"

#include <cmath>

namespace hm::ml {

// -------------------------------------------------------------------
// Sinusoidal Positional Encoding
// -------------------------------------------------------------------

SinusoidalPEImpl::SinusoidalPEImpl(int dim) : dim(dim) {}

torch::Tensor SinusoidalPEImpl::forward(torch::Tensor timesteps) {
    int halfDim = dim / 2;
    float logFactor = std::log(10000.0f) / (halfDim - 1);

    auto freqs = torch::exp(torch::arange(halfDim, torch::kFloat32) * (-logFactor));
    freqs = freqs.to(timesteps.device());

    // [batch, 1] * [halfDim] -> [batch, halfDim]
    auto args = timesteps.unsqueeze(-1).to(torch::kFloat32) * freqs.unsqueeze(0);

    auto sinPE = torch::sin(args);
    auto cosPE = torch::cos(args);

    return torch::cat({sinPE, cosPE}, -1);  // [batch, dim]
}

// -------------------------------------------------------------------
// Timestep Embedder
// -------------------------------------------------------------------

TimestepEmbedderImpl::TimestepEmbedderImpl(int dim) {
    sinPE = register_module("sinPE", SinusoidalPE(dim));
    fc1 = register_module("fc1", torch::nn::Linear(dim, dim));
    fc2 = register_module("fc2", torch::nn::Linear(dim, dim));
}

torch::Tensor TimestepEmbedderImpl::forward(torch::Tensor timesteps) {
    auto pe = sinPE(timesteps);
    pe = torch::gelu(fc1(pe));
    pe = fc2(pe);
    return pe;  // [batch, dim]
}

// -------------------------------------------------------------------
// Pre-Norm Transformer Layer
// -------------------------------------------------------------------

PreNormTransformerLayerImpl::PreNormTransformerLayerImpl(
    int dim, int nHeads, int ffDim, float dropoutRate) {

    norm1 = register_module("norm1",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    norm2 = register_module("norm2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));

    attn = register_module("attn",
        torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(dim, nHeads).dropout(dropoutRate)));

    ff1 = register_module("ff1", torch::nn::Linear(dim, ffDim));
    ff2 = register_module("ff2", torch::nn::Linear(ffDim, dim));
    dropout = register_module("dropout", torch::nn::Dropout(dropoutRate));
}

torch::Tensor PreNormTransformerLayerImpl::forward(torch::Tensor x) {
    // Pre-norm self-attention
    auto normed = norm1(x);
    // MultiheadAttention expects [seq_len, batch, dim]
    auto transposed = normed.transpose(0, 1);
    auto [attnOut, attnWeights] = attn(transposed, transposed, transposed);
    auto attnResult = attnOut.transpose(0, 1);
    x = x + dropout(attnResult);

    // Pre-norm FFN with GELU
    normed = norm2(x);
    auto ffOut = ff2(dropout(torch::gelu(ff1(normed))));
    x = x + dropout(ffOut);

    return x;
}

// -------------------------------------------------------------------
// Motion Transformer
// -------------------------------------------------------------------

MotionTransformerImpl::MotionTransformerImpl(
    int motionDim, int condDim, int modelDim,
    int nHeads, int nLayers, int ffDim, float dropout) {

    input_proj = register_module("input_proj",
        torch::nn::Linear(motionDim, modelDim));
    cond_proj = register_module("cond_proj",
        torch::nn::Linear(condDim, modelDim));
    output_proj = register_module("output_proj",
        torch::nn::Linear(modelDim, motionDim));

    timestep_embedder = register_module("timestep_embedder",
        TimestepEmbedder(modelDim));

    for (int i = 0; i < nLayers; ++i) {
        auto layer = PreNormTransformerLayer(modelDim, nHeads, ffDim, dropout);
        layers.push_back(register_module("layer_" + std::to_string(i), layer));
    }

    final_norm = register_module("final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({modelDim})));
}

torch::Tensor MotionTransformerImpl::forward(
    torch::Tensor x, torch::Tensor t, torch::Tensor cond) {
    // x: [batch, seq_len, 132]
    // t: [batch] (integer timesteps)
    // cond: [batch, 256] (encoded condition)

    int batchSize = x.size(0);
    int seqLen = x.size(1);

    // Project input to model dimension
    auto h = input_proj(x);  // [batch, seq_len, 512]

    // Timestep embedding
    auto tEmb = timestep_embedder(t);  // [batch, 512]
    h = h + tEmb.unsqueeze(1);  // Broadcast over sequence

    // Condition embedding
    auto condEmb = cond_proj(cond);  // [batch, 512]
    h = h + condEmb.unsqueeze(1);  // Broadcast over sequence

    // Transformer layers
    for (auto& layer : layers) {
        h = layer(h);
    }

    h = final_norm(h);

    // Project back to motion dimension
    auto output = output_proj(h);  // [batch, seq_len, 132]
    return output;
}

} // namespace hm::ml
