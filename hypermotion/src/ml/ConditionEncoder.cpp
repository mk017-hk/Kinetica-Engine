#include "HyperMotion/ml/ConditionEncoder.h"
#include "HyperMotion/core/Logger.h"

#ifdef HM_HAS_TORCH

#include <cmath>
#include <stdexcept>

namespace hm::ml {

static constexpr const char* TAG = "ConditionEncoder";

// ---------------------------------------------------------------------------
// Weight initialisation helpers
// ---------------------------------------------------------------------------

namespace {

void applyXavierInit(torch::nn::Linear& layer) {
    torch::nn::init::xavier_uniform_(layer->weight);
    if (layer->bias.defined()) {
        torch::nn::init::zeros_(layer->bias);
    }
}

void applyKaimingInit(torch::nn::Linear& layer) {
    torch::nn::init::kaiming_uniform_(layer->weight, /*a=*/std::sqrt(5.0));
    if (layer->bias.defined()) {
        // Fan-in based uniform bias init (PyTorch default)
        auto fanIn = layer->weight.size(1);
        float bound = 1.0f / std::sqrt(static_cast<float>(fanIn));
        torch::nn::init::uniform_(layer->bias, -bound, bound);
    }
}

void applyOrthogonalInit(torch::nn::Linear& layer) {
    torch::nn::init::orthogonal_(layer->weight);
    if (layer->bias.defined()) {
        torch::nn::init::zeros_(layer->bias);
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

ConditionEncoderImpl::ConditionEncoderImpl(const ConditionEncoderConfig& config)
    : config_(config) {
    buildLayers();
    initWeights();
}

ConditionEncoderImpl::ConditionEncoderImpl(int inputDim, int hiddenDim, int outputDim) {
    config_.inputDim = inputDim;
    config_.hiddenDim = hiddenDim;
    config_.outputDim = outputDim;
    config_.dropoutRate = 0.1f;
    config_.useBatchNorm = false;
    config_.useLayerNorm = false;
    config_.initMethod = ConditionEncoderConfig::InitMethod::Kaiming;

    buildLayers();
    initWeights();
}

void ConditionEncoderImpl::buildLayers() {
    // Core linear layers: 78 -> 512 -> 512 -> 256
    fc1_ = register_module("fc1",
        torch::nn::Linear(config_.inputDim, config_.hiddenDim));
    fc2_ = register_module("fc2",
        torch::nn::Linear(config_.hiddenDim, config_.hiddenDim));
    fc3_ = register_module("fc3",
        torch::nn::Linear(config_.hiddenDim, config_.outputDim));

    // Optional normalisation layers after fc1 and fc2
    if (config_.useBatchNorm) {
        bn1_ = register_module("bn1",
            torch::nn::BatchNorm1d(config_.hiddenDim));
        bn2_ = register_module("bn2",
            torch::nn::BatchNorm1d(config_.hiddenDim));
    }

    if (config_.useLayerNorm) {
        ln1_ = register_module("ln1",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({config_.hiddenDim})));
        ln2_ = register_module("ln2",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({config_.hiddenDim})));
    }

    // Dropout between hidden layers
    if (config_.dropoutRate > 0.0f) {
        drop1_ = register_module("drop1",
            torch::nn::Dropout(config_.dropoutRate));
        drop2_ = register_module("drop2",
            torch::nn::Dropout(config_.dropoutRate));
    }
}

void ConditionEncoderImpl::initWeights() {
    auto initLinear = [&](torch::nn::Linear& layer) {
        switch (config_.initMethod) {
            case ConditionEncoderConfig::InitMethod::Xavier:
                applyXavierInit(layer);
                break;
            case ConditionEncoderConfig::InitMethod::Kaiming:
                applyKaimingInit(layer);
                break;
            case ConditionEncoderConfig::InitMethod::Orthogonal:
                applyOrthogonalInit(layer);
                break;
        }
    };

    initLinear(fc1_);
    initLinear(fc2_);

    // Output layer uses Xavier regardless to keep initial outputs small
    applyXavierInit(fc3_);

    HM_LOG_DEBUG(TAG, "Weights initialized (" +
        std::string(config_.initMethod == ConditionEncoderConfig::InitMethod::Xavier ? "Xavier" :
                    config_.initMethod == ConditionEncoderConfig::InitMethod::Kaiming ? "Kaiming" :
                    "Orthogonal") + "), " +
        std::to_string(parameterCount()) + " parameters");
}

// ---------------------------------------------------------------------------
// Forward pass
// ---------------------------------------------------------------------------

torch::Tensor ConditionEncoderImpl::forward(torch::Tensor x) {
    return forward(x, is_training());
}

torch::Tensor ConditionEncoderImpl::forward(torch::Tensor x, bool isTrain) {
    // Layer 1: Linear -> [Norm] -> ReLU -> [Dropout]
    x = fc1_->forward(x);
    if (config_.useBatchNorm && bn1_) {
        // BatchNorm1d expects [B, C] or [B, C, L]; our input is [B, C]
        x = bn1_->forward(x);
    }
    if (config_.useLayerNorm && ln1_) {
        x = ln1_->forward(x);
    }
    x = torch::relu(x);
    if (drop1_ && isTrain) {
        x = drop1_->forward(x);
    }

    // Layer 2: Linear -> [Norm] -> ReLU -> [Dropout]
    x = fc2_->forward(x);
    if (config_.useBatchNorm && bn2_) {
        x = bn2_->forward(x);
    }
    if (config_.useLayerNorm && ln2_) {
        x = ln2_->forward(x);
    }
    x = torch::relu(x);
    if (drop2_ && isTrain) {
        x = drop2_->forward(x);
    }

    // Layer 3: Linear (no activation — raw latent)
    x = fc3_->forward(x);

    return x;
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

ConditionEncoder ConditionEncoderImpl::create(const ConditionEncoderConfig& config) {
    // Validate configuration
    if (config.inputDim <= 0) {
        HM_LOG_ERROR(TAG, "Invalid inputDim: " + std::to_string(config.inputDim));
        return {nullptr};
    }
    if (config.hiddenDim <= 0) {
        HM_LOG_ERROR(TAG, "Invalid hiddenDim: " + std::to_string(config.hiddenDim));
        return {nullptr};
    }
    if (config.outputDim <= 0) {
        HM_LOG_ERROR(TAG, "Invalid outputDim: " + std::to_string(config.outputDim));
        return {nullptr};
    }
    if (config.dropoutRate < 0.0f || config.dropoutRate >= 1.0f) {
        HM_LOG_ERROR(TAG, "Invalid dropoutRate: " + std::to_string(config.dropoutRate) +
                     " (must be in [0, 1))");
        return {nullptr};
    }
    if (config.useBatchNorm && config.useLayerNorm) {
        HM_LOG_WARN(TAG, "Both BatchNorm and LayerNorm enabled; both will be applied. "
                    "This is unusual — consider enabling only one.");
    }

    ConditionEncoder encoder(config);

    HM_LOG_INFO(TAG, "Created ConditionEncoder: " +
        std::to_string(config.inputDim) + " -> " +
        std::to_string(config.hiddenDim) + " -> " +
        std::to_string(config.hiddenDim) + " -> " +
        std::to_string(config.outputDim) +
        " (dropout=" + std::to_string(config.dropoutRate) +
        ", bn=" + (config.useBatchNorm ? "true" : "false") +
        ", ln=" + (config.useLayerNorm ? "true" : "false") + ")");

    return encoder;
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

int64_t ConditionEncoderImpl::parameterCount() const {
    int64_t count = 0;
    for (const auto& p : parameters()) {
        count += p.numel();
    }
    return count;
}

} // namespace hm::ml

#endif
