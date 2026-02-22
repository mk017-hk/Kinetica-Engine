#include "HyperMotion/style/StyleEncoder.h"
#include "HyperMotion/core/Logger.h"

#include <cmath>
#include <numeric>

namespace hm::style {

static constexpr const char* TAG = "StyleEncoder";

struct StyleEncoder::Impl {
    hm::ml::OnnxInference onnx;
    bool loaded = false;
};

StyleEncoder::StyleEncoder() : impl_(std::make_unique<Impl>()) {}
StyleEncoder::~StyleEncoder() = default;
StyleEncoder::StyleEncoder(StyleEncoder&&) noexcept = default;
StyleEncoder& StyleEncoder::operator=(StyleEncoder&&) noexcept = default;

bool StyleEncoder::load(const std::string& onnxPath, bool useGPU) {
    impl_->loaded = impl_->onnx.load(onnxPath, useGPU);
    return impl_->loaded;
}

bool StyleEncoder::isLoaded() const { return impl_->loaded; }

std::vector<std::array<float, STYLE_INPUT_DIM>>
StyleEncoder::prepareInput(const std::vector<SkeletonFrame>& frames) {
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

std::array<float, STYLE_DIM> StyleEncoder::encode(const std::vector<SkeletonFrame>& frames) {
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

    // L2 normalize (should already be normalized by the model, but be safe)
    float norm = 0.0f;
    for (float v : embedding) norm += v * v;
    norm = std::sqrt(norm);
    if (norm > 1e-8f) {
        for (float& v : embedding) v /= norm;
    }

    return embedding;
}

} // namespace hm::style
