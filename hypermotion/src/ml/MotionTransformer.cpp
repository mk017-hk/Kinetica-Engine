#include "HyperMotion/ml/MotionTransformer.h"
#include "HyperMotion/core/Logger.h"

#ifdef HM_HAS_TORCH

#include <cmath>
#include <sstream>

namespace hm::ml {

static constexpr const char* TAG = "MotionTransformer";

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
    return forward(x, /*attnMask=*/{});
}

torch::Tensor TransformerBlockImpl::forward(torch::Tensor x, const torch::Tensor& attnMask) {
    // Pre-norm self-attention
    auto normed = ln1_->forward(x);
    auto [attnOut, _] = attn_->forward(normed, normed, normed,
                                         /*key_padding_mask=*/{},
                                         /*need_weights=*/false,
                                         /*attn_mask=*/attnMask);
    x = x + drop1_->forward(attnOut);

    // Pre-norm feed-forward (GELU activation)
    normed = ln2_->forward(x);
    auto ff = drop2_->forward(ffn2_->forward(torch::gelu(ffn1_->forward(normed))));
    x = x + ff;

    return x;
}

// -----------------------------------------------------------------------
// MotionTransformer — positional encoding helpers
// -----------------------------------------------------------------------

torch::Tensor MotionTransformerImpl::sinusoidalEmbedding(torch::Tensor t, int dim) {
    // t: [B] integer timesteps -> [B, dim] float embeddings
    int halfDim = dim / 2;
    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(t.device());
    auto freqs = torch::exp(
        -std::log(10000.0) * torch::arange(halfDim, opts) /
        static_cast<float>(halfDim));

    // t: [B] -> [B,1], freqs: [halfDim] -> [1,halfDim]
    auto angles = t.unsqueeze(-1).to(torch::kFloat32) * freqs.unsqueeze(0);
    return torch::cat({torch::sin(angles), torch::cos(angles)}, /*dim=*/-1);  // [B, dim]
}

torch::Tensor MotionTransformerImpl::getPositionalEncoding(int seqLen, torch::Device device) {
    if (config_.useLearnedPosEncoding) {
        // Learned positional encoding
        auto positions = torch::arange(seqLen,
            torch::TensorOptions().dtype(torch::kLong).device(device));
        return learnedPosEnc_->forward(positions);  // [seqLen, modelDim]
    }

    // Sinusoidal positional encoding — compute or retrieve from cache
    if (posEncBuffer_.defined() &&
        posEncBuffer_.size(0) >= seqLen &&
        posEncBuffer_.device() == device) {
        return posEncBuffer_.slice(0, 0, seqLen);
    }

    // Build sinusoidal PE: [maxSeqLen, modelDim]
    int maxLen = std::max(seqLen, config_.maxSeqLen);
    int dim = config_.modelDim;
    int halfDim = dim / 2;

    auto opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto positions = torch::arange(maxLen, opts).unsqueeze(1);   // [maxLen, 1]
    auto divTerm = torch::exp(
        torch::arange(halfDim, opts) * (-std::log(10000.0) / static_cast<float>(halfDim)));  // [halfDim]

    auto angles = positions * divTerm.unsqueeze(0);  // [maxLen, halfDim]
    auto pe = torch::zeros({maxLen, dim}, opts);
    pe.slice(1, 0, halfDim) = torch::sin(angles);
    pe.slice(1, halfDim, dim) = torch::cos(angles);

    posEncBuffer_ = pe;
    return pe.slice(0, 0, seqLen);
}

// -----------------------------------------------------------------------
// MotionTransformer — construction
// -----------------------------------------------------------------------

MotionTransformerImpl::MotionTransformerImpl(const MotionTransformerConfig& config)
    : config_(config), useGradCheckpoint_(config.useGradientCheckpointing) {

    inputProj_ = register_module("input_proj",
        torch::nn::Linear(config_.motionDim, config_.modelDim));

    // Timestep embedding MLP: sinusoidal(modelDim) -> modelDim
    timeMLPfc1_ = register_module("time_mlp_fc1",
        torch::nn::Linear(config_.modelDim, config_.modelDim));
    timeMLPfc2_ = register_module("time_mlp_fc2",
        torch::nn::Linear(config_.modelDim, config_.modelDim));

    condProj_ = register_module("cond_proj",
        torch::nn::Linear(config_.condDim, config_.modelDim));

    // Transformer blocks
    blocks_ = register_module("blocks", torch::nn::ModuleList());
    for (int i = 0; i < config_.numLayers; ++i) {
        blocks_->push_back(TransformerBlock(
            config_.modelDim, config_.numHeads,
            config_.ffnDim, config_.dropout));
    }

    finalNorm_ = register_module("final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({config_.modelDim})));
    outputProj_ = register_module("output_proj",
        torch::nn::Linear(config_.modelDim, config_.motionDim));

    // Positional encoding
    if (config_.useLearnedPosEncoding) {
        learnedPosEnc_ = register_module("pos_enc",
            torch::nn::Embedding(config_.maxSeqLen, config_.modelDim));
    }

    HM_LOG_INFO(TAG, "Created MotionTransformer: " + parameterBreakdown());
}

MotionTransformerImpl::MotionTransformerImpl(int motionDim, int condDim,
                                              int modelDim, int numHeads,
                                              int numLayers, int ffnDim,
                                              float dropout) {
    config_.motionDim = motionDim;
    config_.condDim = condDim;
    config_.modelDim = modelDim;
    config_.numHeads = numHeads;
    config_.numLayers = numLayers;
    config_.ffnDim = ffnDim;
    config_.dropout = dropout;

    inputProj_ = register_module("input_proj",
        torch::nn::Linear(motionDim, modelDim));

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

// -----------------------------------------------------------------------
// Forward pass — full pipeline
// -----------------------------------------------------------------------

torch::Tensor MotionTransformerImpl::forward(torch::Tensor x,
                                              torch::Tensor t,
                                              torch::Tensor cond) {
    auto h = encode(x, t, cond);
    return decode(h);
}

// -----------------------------------------------------------------------
// Encode stage
// -----------------------------------------------------------------------

torch::Tensor MotionTransformerImpl::encode(torch::Tensor x,
                                             torch::Tensor t,
                                             torch::Tensor cond) {
    // x:    [B, S, motionDim]
    // t:    [B]  (int64 timesteps)
    // cond: [B, condDim]

    auto seqLen = x.size(1);

    // Project input motion
    auto h = inputProj_->forward(x);  // [B, S, modelDim]

    // Add positional encoding to sequence tokens
    auto posEnc = getPositionalEncoding(static_cast<int>(seqLen), x.device());  // [S, modelDim]
    h = h + posEnc.unsqueeze(0);  // broadcast over batch

    // Timestep embedding: sinusoidal -> MLP
    auto tEmb = sinusoidalEmbedding(t, config_.modelDim);  // [B, modelDim]
    tEmb = torch::gelu(timeMLPfc1_->forward(tEmb));
    tEmb = timeMLPfc2_->forward(tEmb);                     // [B, modelDim]

    // Condition projection
    auto cEmb = condProj_->forward(cond);                   // [B, modelDim]

    // Add time + condition as bias to every token
    h = h + tEmb.unsqueeze(1) + cEmb.unsqueeze(1);         // [B, S, modelDim]

    // Transformer blocks
    if (useGradCheckpoint_ && is_training()) {
        // Gradient checkpointing: trade compute for memory
        for (const auto& block : *blocks_) {
            auto blockPtr = block->as<TransformerBlockImpl>();
            // Use torch::utils::checkpoint to avoid storing intermediate activations.
            // Wrap forward in a lambda that torch checkpoint can call.
            h = torch::checkpoint(
                [blockPtr](torch::Tensor input) -> torch::Tensor {
                    return blockPtr->forward(input);
                },
                h);
        }
    } else {
        for (const auto& block : *blocks_) {
            h = block->as<TransformerBlockImpl>()->forward(h);
        }
    }

    return h;  // [B, S, modelDim]
}

// -----------------------------------------------------------------------
// Decode stage
// -----------------------------------------------------------------------

torch::Tensor MotionTransformerImpl::decode(torch::Tensor h) {
    h = finalNorm_->forward(h);
    return outputProj_->forward(h);  // [B, S, motionDim]
}

// -----------------------------------------------------------------------
// Parameter counting
// -----------------------------------------------------------------------

int64_t MotionTransformerImpl::parameterCount() const {
    int64_t count = 0;
    for (const auto& p : parameters()) {
        count += p.numel();
    }
    return count;
}

std::string MotionTransformerImpl::parameterBreakdown() const {
    auto countModule = [](const torch::nn::Module& mod) -> int64_t {
        int64_t count = 0;
        for (const auto& p : mod.parameters()) {
            count += p.numel();
        }
        return count;
    };

    int64_t inputProjParams = countModule(*inputProj_);
    int64_t timeMlpParams = countModule(*timeMLPfc1_) + countModule(*timeMLPfc2_);
    int64_t condProjParams = countModule(*condProj_);

    int64_t blockParams = 0;
    for (const auto& block : *blocks_) {
        blockParams += countModule(*block);
    }

    int64_t normParams = countModule(*finalNorm_);
    int64_t outputProjParams = countModule(*outputProj_);
    int64_t posEncParams = 0;
    if (learnedPosEnc_) {
        posEncParams = countModule(*learnedPosEnc_);
    }

    int64_t total = inputProjParams + timeMlpParams + condProjParams +
                    blockParams + normParams + outputProjParams + posEncParams;

    std::ostringstream oss;
    oss << total << " params total ("
        << "input=" << inputProjParams
        << ", time_mlp=" << timeMlpParams
        << ", cond=" << condProjParams
        << ", blocks=" << blockParams << " (" << config_.numLayers << " layers)"
        << ", norm=" << normParams
        << ", output=" << outputProjParams;
    if (posEncParams > 0) {
        oss << ", pos_enc=" << posEncParams;
    }
    oss << ")";

    return oss.str();
}

// -----------------------------------------------------------------------
// Gradient checkpointing
// -----------------------------------------------------------------------

void MotionTransformerImpl::setGradientCheckpointing(bool enable) {
    useGradCheckpoint_ = enable;
    if (enable) {
        HM_LOG_INFO(TAG, "Gradient checkpointing enabled");
    } else {
        HM_LOG_INFO(TAG, "Gradient checkpointing disabled");
    }
}

bool MotionTransformerImpl::gradientCheckpointingEnabled() const {
    return useGradCheckpoint_;
}

} // namespace hm::ml

#endif
