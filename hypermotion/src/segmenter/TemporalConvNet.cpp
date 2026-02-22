#include "HyperMotion/segmenter/TemporalConvNet.h"
#include "HyperMotion/core/Logger.h"

namespace hm::segmenter {

static constexpr const char* TAG = "TemporalConvNet";

struct TemporalConvNet::Impl {
    hm::ml::OnnxInference onnx;
    bool loaded = false;
};

TemporalConvNet::TemporalConvNet() : impl_(std::make_unique<Impl>()) {}
TemporalConvNet::~TemporalConvNet() = default;
TemporalConvNet::TemporalConvNet(TemporalConvNet&&) noexcept = default;
TemporalConvNet& TemporalConvNet::operator=(TemporalConvNet&&) noexcept = default;

bool TemporalConvNet::load(const std::string& onnxPath, bool useGPU) {
    impl_->loaded = impl_->onnx.load(onnxPath, useGPU);
    return impl_->loaded;
}

bool TemporalConvNet::isLoaded() const { return impl_->loaded; }

std::vector<std::array<float, MOTION_TYPE_COUNT>> TemporalConvNet::classify(
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

} // namespace hm::segmenter
