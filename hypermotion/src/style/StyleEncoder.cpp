#include "HyperMotion/style/StyleEncoder.h"
#include "HyperMotion/core/Logger.h"

#include <cmath>
#include <numeric>

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

// -----------------------------------------------------------------------
// ResBlock1D
// -----------------------------------------------------------------------

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

// -----------------------------------------------------------------------
// StyleEncoder (training)
// -----------------------------------------------------------------------

StyleEncoderImpl::StyleEncoderImpl(int inputDim, int styleDim) {
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
}

torch::Tensor StyleEncoderImpl::forward(torch::Tensor x) {
    // x: [B, inputDim, T]
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
