#include "HyperMotion/style/StyleEncoder.h"
#include "HyperMotion/core/Logger.h"

#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

namespace hm::style {

static constexpr const char* TAG = "StyleEncoder";

// ===================================================================
// ONNX inference version (StyleEncoderOnnx)
// ===================================================================

struct StyleEncoderOnnx::Impl {
    hm::ml::OnnxInference onnx;
    bool loaded = false;
};

StyleEncoderOnnx::StyleEncoderOnnx() : impl_(std::make_unique<Impl>()) {}
StyleEncoderOnnx::~StyleEncoderOnnx() = default;
StyleEncoderOnnx::StyleEncoderOnnx(StyleEncoderOnnx&&) noexcept = default;
StyleEncoderOnnx& StyleEncoderOnnx::operator=(StyleEncoderOnnx&&) noexcept = default;

bool StyleEncoderOnnx::load(const std::string& onnxPath, bool useGPU) {
    impl_->loaded = impl_->onnx.load(onnxPath, useGPU);
    return impl_->loaded;
}

bool StyleEncoderOnnx::isLoaded() const { return impl_->loaded; }

std::vector<std::array<float, STYLE_INPUT_DIM>>
StyleEncoderOnnx::prepareInput(const std::vector<SkeletonFrame>& frames) {
    int numFrames = static_cast<int>(frames.size());
    std::vector<std::array<float, STYLE_INPUT_DIM>> result(numFrames);

    for (int f = 0; f < numFrames; ++f) {
        int idx = 0;

        // 132D rotations (22 joints x 6D)
        for (int j = 0; j < JOINT_COUNT; ++j) {
            for (int d = 0; d < ROTATION_DIM; ++d) {
                result[f][idx++] = frames[f].joints[j].rotation6D[d];
            }
        }

        // 3D root velocity (normalized)
        result[f][idx++] = frames[f].rootVelocity.x / 800.0f;
        result[f][idx++] = frames[f].rootVelocity.y / 800.0f;
        result[f][idx++] = frames[f].rootVelocity.z / 800.0f;

        // 66D angular velocities (finite differences)
        if (f > 0) {
            for (int j = 0; j < JOINT_COUNT; ++j) {
                Vec3 prev = frames[f - 1].joints[j].localEulerDeg;
                Vec3 curr = frames[f].joints[j].localEulerDeg;
                result[f][idx++] = (curr.x - prev.x) / 360.0f;
                result[f][idx++] = (curr.y - prev.y) / 360.0f;
                result[f][idx++] = (curr.z - prev.z) / 360.0f;
            }
        } else {
            for (int k = 0; k < 66; ++k) result[f][idx++] = 0.0f;
        }
    }

    return result;
}

std::array<float, STYLE_DIM> StyleEncoderOnnx::encode(const std::vector<SkeletonFrame>& frames) {
    std::array<float, STYLE_DIM> embedding{};
    embedding.fill(0.0f);

    if (!impl_->loaded || frames.empty()) {
        HM_LOG_WARN(TAG, "Encoder not loaded or empty input");
        return embedding;
    }

    auto feats = prepareInput(frames);
    int numFrames = static_cast<int>(feats.size());

    // Flatten to contiguous buffer [1, numFrames, 201]
    std::vector<float> inputData(numFrames * STYLE_INPUT_DIM);
    for (int f = 0; f < numFrames; ++f) {
        std::copy(feats[f].begin(), feats[f].end(),
                  inputData.begin() + f * STYLE_INPUT_DIM);
    }

#ifdef HM_HAS_ONNXRUNTIME
    std::vector<int64_t> inputShape = {1, static_cast<int64_t>(numFrames),
                                        static_cast<int64_t>(STYLE_INPUT_DIM)};
    auto& memInfo = impl_->onnx.memoryInfo();

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memInfo, inputData.data(), inputData.size(),
        inputShape.data(), inputShape.size()));

    auto outputs = impl_->onnx.run(inputs);

    // Output: [1, 64]
    const float* embData = outputs[0].GetTensorData<float>();
    std::copy_n(embData, STYLE_DIM, embedding.begin());
#else
    (void)inputData;
    HM_LOG_WARN(TAG, "ONNX Runtime not available, returning zero embedding");
    return embedding;
#endif

    // L2 normalize
    float norm = 0.0f;
    for (float v : embedding) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
        for (float& v : embedding) v /= norm;
    }

    return embedding;
}

// ===================================================================
// LibTorch training version
// ===================================================================

#ifdef HM_HAS_TORCH

// -------------------------------------------------------------------
// ModelSummary::toString
// -------------------------------------------------------------------

std::string ModelSummary::toString() const {
    std::ostringstream oss;
    oss << "================================================================\n";
    oss << std::left << std::setw(32) << "Layer (name)"
        << std::setw(18) << "Type"
        << std::right << std::setw(14) << "Params"
        << std::setw(14) << "Trainable" << "\n";
    oss << "----------------------------------------------------------------\n";

    for (const auto& layer : layers) {
        oss << std::left << std::setw(32) << layer.name
            << std::setw(18) << layer.type
            << std::right << std::setw(14) << layer.numParameters
            << std::setw(14) << layer.numTrainable << "\n";
    }

    oss << "================================================================\n";
    oss << "Total params: " << totalParameters << "\n";
    oss << "Trainable params: " << trainableParameters << "\n";
    oss << "Non-trainable params: " << nonTrainableParameters << "\n";
    oss << "================================================================\n";

    return oss.str();
}

// -------------------------------------------------------------------
// ResBlock1D
// -------------------------------------------------------------------

ResBlock1DImpl::ResBlock1DImpl(int inChannels, int outChannels) {
    conv1_ = register_module("conv1", torch::nn::Conv1d(
        torch::nn::Conv1dOptions(inChannels, outChannels, 3).padding(1)));
    bn1_ = register_module("bn1", torch::nn::BatchNorm1d(outChannels));
    conv2_ = register_module("conv2", torch::nn::Conv1d(
        torch::nn::Conv1dOptions(outChannels, outChannels, 3).padding(1)));
    bn2_ = register_module("bn2", torch::nn::BatchNorm1d(outChannels));

    needsDownsample_ = (inChannels != outChannels);
    if (needsDownsample_) {
        downsample_ = register_module("downsample", torch::nn::Conv1d(
            torch::nn::Conv1dOptions(inChannels, outChannels, 1)));
    }
}

torch::Tensor ResBlock1DImpl::forward(torch::Tensor x) {
    auto residual = needsDownsample_ ? downsample_->forward(x) : x;

    auto h = torch::relu(bn1_->forward(conv1_->forward(x)));
    h = bn2_->forward(conv2_->forward(h));
    return torch::relu(h + residual);
}

// -------------------------------------------------------------------
// StyleEncoder (training) -- Construction
// -------------------------------------------------------------------

StyleEncoderImpl::StyleEncoderImpl(int inputDim, int styleDim)
    : inputDim_(inputDim), styleDim_(styleDim) {

    inputConv_ = register_module("input_conv", torch::nn::Conv1d(
        torch::nn::Conv1dOptions(inputDim, 128, 3).padding(1)));
    inputBN_ = register_module("input_bn", torch::nn::BatchNorm1d(128));

    // 4 ResBlocks: 128->128, 128->256, 256->256, 256->512
    resBlocks_ = register_module("res_blocks", torch::nn::ModuleList());
    resBlocks_->push_back(ResBlock1D(128, 128));
    resBlocks_->push_back(ResBlock1D(128, 256));
    resBlocks_->push_back(ResBlock1D(256, 256));
    resBlocks_->push_back(ResBlock1D(256, 512));

    fc1_ = register_module("fc1", torch::nn::Linear(512, 256));
    fc2_ = register_module("fc2", torch::nn::Linear(256, styleDim));

    // Apply default Kaiming initialization
    initializeWeights(WeightInitStrategy::KaimingNormal);
}

// -------------------------------------------------------------------
// Forward pass
// -------------------------------------------------------------------

torch::Tensor StyleEncoderImpl::forward(torch::Tensor x) {
    // x: [B, inputDim, T]

    // Optional input normalization
    if (hasInputNorm_) {
        x = normalizeInput(x);
    }

    auto h = torch::relu(inputBN_->forward(inputConv_->forward(x)));

    for (const auto& block : *resBlocks_) {
        h = block->as<ResBlock1DImpl>()->forward(h);
    }

    // Global Average Pooling: [B, 512, T] -> [B, 512]
    h = h.mean(/*dim=*/2);

    h = torch::relu(fc1_->forward(h));
    h = fc2_->forward(h);

    // L2 normalize
    h = torch::nn::functional::normalize(h,
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));

    return h;  // [B, styleDim]
}

// -------------------------------------------------------------------
// Forward with intermediate feature extraction
// -------------------------------------------------------------------

torch::Tensor StyleEncoderImpl::forwardWithIntermediates(
    torch::Tensor x, IntermediateFeatures& features) {
    // x: [B, inputDim, T]

    // Optional input normalization
    if (hasInputNorm_) {
        x = normalizeInput(x);
    }

    // Input conv + BN + ReLU
    auto h = torch::relu(inputBN_->forward(inputConv_->forward(x)));
    features.afterInputConv = h.detach().clone();

    // ResBlocks
    h = resBlocks_[0]->as<ResBlock1DImpl>()->forward(h);
    features.afterResBlock0 = h.detach().clone();

    h = resBlocks_[1]->as<ResBlock1DImpl>()->forward(h);
    features.afterResBlock1 = h.detach().clone();

    h = resBlocks_[2]->as<ResBlock1DImpl>()->forward(h);
    features.afterResBlock2 = h.detach().clone();

    h = resBlocks_[3]->as<ResBlock1DImpl>()->forward(h);
    features.afterResBlock3 = h.detach().clone();

    // Global Average Pooling
    h = h.mean(/*dim=*/2);
    features.afterGAP = h.detach().clone();

    // FC layers
    h = torch::relu(fc1_->forward(h));
    features.afterFC1 = h.detach().clone();

    h = fc2_->forward(h);

    // L2 normalize
    h = torch::nn::functional::normalize(h,
        torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
    features.finalEmbedding = h.detach().clone();

    return h;
}

// -------------------------------------------------------------------
// Weight initialization
// -------------------------------------------------------------------

static void initConvWeights(torch::nn::Conv1d& conv, WeightInitStrategy strategy) {
    if (!conv.get()) return;
    switch (strategy) {
        case WeightInitStrategy::KaimingNormal:
            torch::nn::init::kaiming_normal_(
                conv->weight,
                /*a=*/0.0,
                torch::kFanOut,
                torch::kReLU);
            break;
        case WeightInitStrategy::KaimingUniform:
            torch::nn::init::kaiming_uniform_(
                conv->weight,
                /*a=*/0.0,
                torch::kFanOut);
            break;
        case WeightInitStrategy::XavierNormal:
            torch::nn::init::xavier_normal_(conv->weight);
            break;
        case WeightInitStrategy::XavierUniform:
            torch::nn::init::xavier_uniform_(conv->weight);
            break;
        case WeightInitStrategy::Default:
            // Do nothing; use PyTorch default
            break;
    }
    if (conv->bias.defined()) {
        torch::nn::init::zeros_(conv->bias);
    }
}

static void initLinearWeights(torch::nn::Linear& linear, WeightInitStrategy strategy) {
    if (!linear.get()) return;
    switch (strategy) {
        case WeightInitStrategy::KaimingNormal:
            // For linear layers after ReLU, Kaiming is appropriate
            torch::nn::init::kaiming_normal_(
                linear->weight,
                /*a=*/0.0,
                torch::kFanOut,
                torch::kReLU);
            break;
        case WeightInitStrategy::KaimingUniform:
            torch::nn::init::kaiming_uniform_(
                linear->weight,
                /*a=*/0.0,
                torch::kFanOut);
            break;
        case WeightInitStrategy::XavierNormal:
            torch::nn::init::xavier_normal_(linear->weight);
            break;
        case WeightInitStrategy::XavierUniform:
            torch::nn::init::xavier_uniform_(linear->weight);
            break;
        case WeightInitStrategy::Default:
            break;
    }
    if (linear->bias.defined()) {
        torch::nn::init::zeros_(linear->bias);
    }
}

static void initBatchNormWeights(torch::nn::BatchNorm1d& bn) {
    if (!bn.get()) return;
    if (bn->weight.defined()) {
        torch::nn::init::ones_(bn->weight);
    }
    if (bn->bias.defined()) {
        torch::nn::init::zeros_(bn->bias);
    }
}

void StyleEncoderImpl::initializeWeights(WeightInitStrategy strategy) {
    if (strategy == WeightInitStrategy::Default) {
        HM_LOG_DEBUG(TAG, "Using default PyTorch initialization (no explicit init)");
        return;
    }

    HM_LOG_DEBUG(TAG, "Initializing weights with strategy: " +
                 std::to_string(static_cast<int>(strategy)));

    torch::NoGradGuard noGrad;

    // Input conv + BN
    initConvWeights(inputConv_, strategy);
    initBatchNormWeights(inputBN_);

    // ResBlocks: iterate through each block's sub-modules
    for (const auto& block : *resBlocks_) {
        auto* resBlock = block->as<ResBlock1DImpl>();
        // Access named parameters to initialize the conv layers inside ResBlocks
        for (auto& pair : resBlock->named_modules(/*memo=*/std::string{}, /*include_self=*/false)) {
            const auto& name = pair.key();
            auto& mod = pair.value();

            // Try Conv1d
            auto conv1d = std::dynamic_pointer_cast<torch::nn::Conv1dImpl>(mod);
            if (conv1d) {
                switch (strategy) {
                    case WeightInitStrategy::KaimingNormal:
                        torch::nn::init::kaiming_normal_(conv1d->weight, 0.0, torch::kFanOut, torch::kReLU);
                        break;
                    case WeightInitStrategy::KaimingUniform:
                        torch::nn::init::kaiming_uniform_(conv1d->weight, 0.0, torch::kFanOut);
                        break;
                    case WeightInitStrategy::XavierNormal:
                        torch::nn::init::xavier_normal_(conv1d->weight);
                        break;
                    case WeightInitStrategy::XavierUniform:
                        torch::nn::init::xavier_uniform_(conv1d->weight);
                        break;
                    default:
                        break;
                }
                if (conv1d->bias.defined()) {
                    torch::nn::init::zeros_(conv1d->bias);
                }
                continue;
            }

            // Try BatchNorm1d
            auto bn1d = std::dynamic_pointer_cast<torch::nn::BatchNorm1dImpl>(mod);
            if (bn1d) {
                if (bn1d->weight.defined()) torch::nn::init::ones_(bn1d->weight);
                if (bn1d->bias.defined()) torch::nn::init::zeros_(bn1d->bias);
                continue;
            }
        }
    }

    // FC layers: use Xavier for the final projection since it feeds into L2 norm,
    // but Kaiming for fc1 which is followed by ReLU
    initLinearWeights(fc1_, strategy);

    // Final layer (fc2) benefits from Xavier since its output is L2-normalized
    // rather than fed through ReLU
    WeightInitStrategy finalStrategy = strategy;
    if (strategy == WeightInitStrategy::KaimingNormal) {
        finalStrategy = WeightInitStrategy::XavierNormal;
    } else if (strategy == WeightInitStrategy::KaimingUniform) {
        finalStrategy = WeightInitStrategy::XavierUniform;
    }
    initLinearWeights(fc2_, finalStrategy);

    HM_LOG_INFO(TAG, "Weight initialization complete. Total params: " +
                std::to_string(countTotalParameters()));
}

// -------------------------------------------------------------------
// Inference mode
// -------------------------------------------------------------------

void StyleEncoderImpl::setInferenceMode(bool enabled) {
    inferenceMode_ = enabled;
    if (enabled) {
        this->eval();
        // Freeze all parameters for inference
        for (auto& param : this->parameters()) {
            param.set_requires_grad(false);
        }
        HM_LOG_DEBUG(TAG, "Inference mode enabled (eval + params frozen)");
    } else {
        this->train();
        // Unfreeze all parameters for training
        for (auto& param : this->parameters()) {
            param.set_requires_grad(true);
        }
        HM_LOG_DEBUG(TAG, "Training mode restored (train + params unfrozen)");
    }
}

// -------------------------------------------------------------------
// Model summary and parameter counting
// -------------------------------------------------------------------

int64_t StyleEncoderImpl::countTrainableParameters() const {
    int64_t count = 0;
    for (const auto& param : this->parameters()) {
        if (param.requires_grad()) {
            count += param.numel();
        }
    }
    return count;
}

int64_t StyleEncoderImpl::countTotalParameters() const {
    int64_t count = 0;
    for (const auto& param : this->parameters()) {
        count += param.numel();
    }
    return count;
}

ModelSummary StyleEncoderImpl::summary() const {
    ModelSummary result;

    for (const auto& pair : this->named_modules(/*memo=*/std::string{}, /*include_self=*/false)) {
        const auto& name = pair.key();
        auto& mod = pair.value();

        LayerSummary layer;
        layer.name = name;

        // Determine type string
        auto conv1d = std::dynamic_pointer_cast<torch::nn::Conv1dImpl>(mod);
        auto bn1d = std::dynamic_pointer_cast<torch::nn::BatchNorm1dImpl>(mod);
        auto linear = std::dynamic_pointer_cast<torch::nn::LinearImpl>(mod);

        if (conv1d) {
            layer.type = "Conv1d";
        } else if (bn1d) {
            layer.type = "BatchNorm1d";
        } else if (linear) {
            layer.type = "Linear";
        } else {
            layer.type = "Module";
        }

        // Count parameters in this module (direct, not recursive)
        int64_t numParams = 0;
        int64_t numTrainable = 0;
        for (const auto& p : mod->parameters(/*recurse=*/false)) {
            numParams += p.numel();
            if (p.requires_grad()) {
                numTrainable += p.numel();
            }
        }
        // Also count buffers (e.g., running_mean, running_var in BN)
        for (const auto& b : mod->buffers(/*recurse=*/false)) {
            numParams += b.numel();
        }

        layer.numParameters = numParams;
        layer.numTrainable = numTrainable;

        // Only add leaf modules with parameters (skip container wrappers)
        if (numParams > 0) {
            result.layers.push_back(layer);
        }
    }

    // Compute totals
    result.totalParameters = countTotalParameters();
    result.trainableParameters = countTrainableParameters();
    result.nonTrainableParameters = result.totalParameters - result.trainableParameters;

    // Add non-parameter buffers to total (BN running stats etc.)
    for (const auto& b : this->buffers()) {
        result.totalParameters += b.numel();
    }
    result.nonTrainableParameters = result.totalParameters - result.trainableParameters;

    return result;
}

void StyleEncoderImpl::printSummary() const {
    auto s = summary();
    HM_LOG_INFO(TAG, "\n" + s.toString());
}

// -------------------------------------------------------------------
// Input normalization
// -------------------------------------------------------------------

void StyleEncoderImpl::computeInputNormalization(
    const std::vector<torch::Tensor>& samples,
    int maxBatches) {

    if (samples.empty()) {
        HM_LOG_WARN(TAG, "No samples provided for input normalization");
        return;
    }

    HM_LOG_INFO(TAG, "Computing input normalization statistics from " +
                std::to_string(samples.size()) + " samples");

    torch::NoGradGuard noGrad;

    // Welford's online algorithm for numerically stable mean/variance
    auto runningMean = torch::zeros({inputDim_});
    auto runningM2 = torch::zeros({inputDim_});
    int64_t totalFrames = 0;

    int batchCount = 0;
    for (const auto& sample : samples) {
        if (maxBatches > 0 && batchCount >= maxBatches) break;

        // sample: [B, inputDim, T] or [inputDim, T]
        torch::Tensor data;
        if (sample.dim() == 3) {
            // [B, inputDim, T] -> [inputDim, B*T]
            data = sample.permute({1, 0, 2}).reshape({inputDim_, -1});
        } else if (sample.dim() == 2) {
            // [inputDim, T] -> already fine
            data = sample;
        } else {
            continue;
        }

        int64_t numFrames = data.size(1);
        for (int64_t f = 0; f < numFrames; ++f) {
            totalFrames++;
            auto frame = data.select(1, f);  // [inputDim]
            auto delta = frame - runningMean;
            runningMean += delta / static_cast<float>(totalFrames);
            auto delta2 = frame - runningMean;
            runningM2 += delta * delta2;
        }

        batchCount++;
    }

    if (totalFrames < 2) {
        HM_LOG_WARN(TAG, "Insufficient data for normalization (need >= 2 frames)");
        return;
    }

    auto variance = runningM2 / static_cast<float>(totalFrames - 1);
    auto stddev = torch::sqrt(variance + 1e-8f);

    inputMean_ = runningMean;
    inputStd_ = stddev;
    hasInputNorm_ = true;

    HM_LOG_INFO(TAG, "Input normalization computed from " +
                std::to_string(totalFrames) + " frames across " +
                std::to_string(batchCount) + " batches");

    // Log some statistics
    HM_LOG_DEBUG(TAG, "Mean range: [" +
                 std::to_string(inputMean_.min().item<float>()) + ", " +
                 std::to_string(inputMean_.max().item<float>()) + "]");
    HM_LOG_DEBUG(TAG, "Std range: [" +
                 std::to_string(inputStd_.min().item<float>()) + ", " +
                 std::to_string(inputStd_.max().item<float>()) + "]");
}

torch::Tensor StyleEncoderImpl::normalizeInput(torch::Tensor x) const {
    if (!hasInputNorm_) {
        return x;
    }

    // x: [B, inputDim, T]
    // inputMean_ and inputStd_: [inputDim]
    // Reshape for broadcasting: [1, inputDim, 1]
    auto mean = inputMean_.unsqueeze(0).unsqueeze(2).to(x.device());
    auto std = inputStd_.unsqueeze(0).unsqueeze(2).to(x.device());

    return (x - mean) / std;
}

std::pair<std::vector<float>, std::vector<float>>
StyleEncoderImpl::getInputNormStats() const {
    std::vector<float> mean(inputDim_, 0.0f);
    std::vector<float> std(inputDim_, 1.0f);

    if (hasInputNorm_) {
        auto meanAcc = inputMean_.accessor<float, 1>();
        auto stdAcc = inputStd_.accessor<float, 1>();
        for (int i = 0; i < inputDim_; ++i) {
            mean[i] = meanAcc[i];
            std[i] = stdAcc[i];
        }
    }

    return {mean, std};
}

// -------------------------------------------------------------------
// prepareInput (static utility)
// -------------------------------------------------------------------

torch::Tensor StyleEncoderImpl::prepareInput(const std::vector<SkeletonFrame>& frames) {
    int numFrames = static_cast<int>(frames.size());
    auto tensor = torch::zeros({1, STYLE_INPUT_DIM, numFrames});
    auto acc = tensor.accessor<float, 3>();

    for (int f = 0; f < numFrames; ++f) {
        int idx = 0;

        // 132D rotations (22 joints x 6D)
        for (int j = 0; j < JOINT_COUNT; ++j) {
            for (int d = 0; d < ROTATION_DIM; ++d) {
                acc[0][idx++][f] = frames[f].joints[j].rotation6D[d];
            }
        }

        // 3D root velocity (normalized)
        acc[0][idx++][f] = frames[f].rootVelocity.x / 800.0f;
        acc[0][idx++][f] = frames[f].rootVelocity.y / 800.0f;
        acc[0][idx++][f] = frames[f].rootVelocity.z / 800.0f;

        // 66D angular velocities (finite differences)
        if (f > 0) {
            for (int j = 0; j < JOINT_COUNT; ++j) {
                Vec3 prev = frames[f - 1].joints[j].localEulerDeg;
                Vec3 curr = frames[f].joints[j].localEulerDeg;
                acc[0][idx++][f] = (curr.x - prev.x) / 360.0f;
                acc[0][idx++][f] = (curr.y - prev.y) / 360.0f;
                acc[0][idx++][f] = (curr.z - prev.z) / 360.0f;
            }
        } else {
            for (int k = 0; k < 66; ++k) acc[0][idx++][f] = 0.0f;
        }
    }

    return tensor;  // [1, STYLE_INPUT_DIM, T]
}

#endif  // HM_HAS_TORCH

} // namespace hm::style
