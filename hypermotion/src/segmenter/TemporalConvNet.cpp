#include "HyperMotion/segmenter/TemporalConvNet.h"
#include "HyperMotion/core/Logger.h"

namespace hm::segmenter {

static constexpr const char* TAG = "TemporalConvNet";

// ===================================================================
// ONNX inference version (TemporalConvNetOnnx)
// ===================================================================

struct TemporalConvNetOnnx::Impl {
    hm::ml::OnnxInference onnx;
    bool loaded = false;
};

TemporalConvNetOnnx::TemporalConvNetOnnx() : impl_(std::make_unique<Impl>()) {}
TemporalConvNetOnnx::~TemporalConvNetOnnx() = default;
TemporalConvNetOnnx::TemporalConvNetOnnx(TemporalConvNetOnnx&&) noexcept = default;
TemporalConvNetOnnx& TemporalConvNetOnnx::operator=(TemporalConvNetOnnx&&) noexcept = default;

bool TemporalConvNetOnnx::load(const std::string& onnxPath, bool useGPU) {
    impl_->loaded = impl_->onnx.load(onnxPath, useGPU);
    return impl_->loaded;
}

bool TemporalConvNetOnnx::isLoaded() const { return impl_->loaded; }

std::vector<std::array<float, MOTION_TYPE_COUNT>> TemporalConvNetOnnx::classify(
    const std::vector<std::array<float, 70>>& features) {

    int numFrames = static_cast<int>(features.size());
    std::vector<std::array<float, MOTION_TYPE_COUNT>> result(numFrames);

    if (!impl_->loaded || numFrames == 0) {
        for (auto& r : result) {
            r.fill(0.0f);
            r[static_cast<int>(MotionType::Unknown)] = 1.0f;
        }
        return result;
    }

    // Flatten to [1, numFrames, 70]
    std::vector<float> inputData(numFrames * 70);
    for (int f = 0; f < numFrames; ++f) {
        std::copy(features[f].begin(), features[f].end(),
                  inputData.begin() + f * 70);
    }

    std::vector<int64_t> inputShape = {1, static_cast<int64_t>(numFrames), 70};
    auto& memInfo = impl_->onnx.memoryInfo();

    std::vector<Ort::Value> inputs;
    inputs.push_back(Ort::Value::CreateTensor<float>(
        memInfo, inputData.data(), inputData.size(),
        inputShape.data(), inputShape.size()));

    auto outputs = impl_->onnx.run(inputs);

    // Output: [1, numFrames, MOTION_TYPE_COUNT]
    const float* logits = outputs[0].GetTensorData<float>();

    for (int f = 0; f < numFrames; ++f) {
        const float* frameLogits = logits + f * MOTION_TYPE_COUNT;
        std::copy_n(frameLogits, MOTION_TYPE_COUNT, result[f].begin());
    }

    return result;
}

// ===================================================================
// LibTorch training version
// ===================================================================

#ifdef HM_HAS_TORCH

// -----------------------------------------------------------------------
// TemporalBlock
// -----------------------------------------------------------------------

TemporalBlockImpl::TemporalBlockImpl(int inChannels, int outChannels,
                                       int kernelSize, int dilation,
                                       float dropout) {
    // Causal padding: pad left by (kernel-1)*dilation
    padding_ = (kernelSize - 1) * dilation;

    conv1_ = register_module("conv1", torch::nn::Conv1d(
        torch::nn::Conv1dOptions(inChannels, outChannels, kernelSize)
            .dilation(dilation).padding(0)));
    bn1_ = register_module("bn1", torch::nn::BatchNorm1d(outChannels));

    conv2_ = register_module("conv2", torch::nn::Conv1d(
        torch::nn::Conv1dOptions(outChannels, outChannels, kernelSize)
            .dilation(dilation).padding(0)));
    bn2_ = register_module("bn2", torch::nn::BatchNorm1d(outChannels));

    drop_ = register_module("drop", torch::nn::Dropout(dropout));

    needsResidual_ = (inChannels != outChannels);
    if (needsResidual_) {
        residual_ = register_module("residual", torch::nn::Conv1d(
            torch::nn::Conv1dOptions(inChannels, outChannels, 1)));
    }
}

torch::Tensor TemporalBlockImpl::forward(torch::Tensor x) {
    // x: [B, C, T]
    auto residual = needsResidual_ ? residual_->forward(x) : x;

    // Causal conv1: pad left, then trim right
    auto h = torch::nn::functional::pad(x,
        torch::nn::functional::PadFuncOptions({padding_, 0}));
    h = torch::relu(bn1_->forward(conv1_->forward(h)));
    h = drop_->forward(h);

    // Causal conv2
    h = torch::nn::functional::pad(h,
        torch::nn::functional::PadFuncOptions({padding_, 0}));
    h = torch::relu(bn2_->forward(conv2_->forward(h)));
    h = drop_->forward(h);

    return torch::relu(h + residual);
}

// -----------------------------------------------------------------------
// TemporalConvNet (training module)
// -----------------------------------------------------------------------

TemporalConvNetImpl::TemporalConvNetImpl(int inputDim, int hiddenDim,
                                           int numClasses) {
    blocks_ = register_module("blocks", torch::nn::ModuleList());

    // 6 dilated causal blocks: dilation = 1, 2, 4, 8, 16, 32
    int dilations[] = {1, 2, 4, 8, 16, 32};
    int inCh = inputDim;
    for (int d : dilations) {
        blocks_->push_back(TemporalBlock(inCh, hiddenDim, /*kernelSize=*/3,
                                          d, /*dropout=*/0.1f));
        inCh = hiddenDim;
    }

    output_ = register_module("output", torch::nn::Conv1d(
        torch::nn::Conv1dOptions(hiddenDim, numClasses, 1)));
}

torch::Tensor TemporalConvNetImpl::forward(torch::Tensor x) {
    // x: [B, inputDim, T]
    for (const auto& block : *blocks_) {
        x = block->as<TemporalBlockImpl>()->forward(x);
    }
    return output_->forward(x);  // [B, numClasses, T]
}

#endif  // HM_HAS_TORCH

} // namespace hm::segmenter
