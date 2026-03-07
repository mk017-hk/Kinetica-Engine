#include "HyperMotion/analysis/MotionEmbedder.h"
#include "HyperMotion/core/Logger.h"

#include <cmath>
#include <algorithm>
#include <numeric>

namespace hm::analysis {

static constexpr const char* TAG = "MotionEmbedder";
static constexpr int JOINTS_DIM = JOINT_COUNT * 3;  // 66

struct MotionEmbedder::Impl {
    MotionEmbedderConfig config;
    ml::OnnxInference onnx;
    bool initialized = false;
};

MotionEmbedder::MotionEmbedder(const MotionEmbedderConfig& config)
    : impl_(std::make_unique<Impl>()) {
    impl_->config = config;
}

MotionEmbedder::~MotionEmbedder() = default;
MotionEmbedder::MotionEmbedder(MotionEmbedder&&) noexcept = default;
MotionEmbedder& MotionEmbedder::operator=(MotionEmbedder&&) noexcept = default;

bool MotionEmbedder::initialize() {
    if (impl_->config.onnxModelPath.empty()) {
        HM_LOG_WARN(TAG, "No ONNX model path specified — embedding will return zeros");
        impl_->initialized = true;
        return true;
    }

    bool loaded = impl_->onnx.load(impl_->config.onnxModelPath, impl_->config.useGPU);
    if (!loaded) {
        HM_LOG_WARN(TAG, "Failed to load ONNX model: " + impl_->config.onnxModelPath +
                    " — embedding will return zeros");
        impl_->initialized = true;
        return true;
    }

    impl_->initialized = true;
    HM_LOG_INFO(TAG, "Motion embedder loaded: " + impl_->config.onnxModelPath);
    return true;
}

bool MotionEmbedder::isInitialized() const {
    return impl_->initialized;
}

std::vector<float> MotionEmbedder::prepareInput(
    const std::vector<SkeletonFrame>& frames, int seqLen) {

    // Extract normalised joint world positions
    // Centre on hips, scale by skeleton height
    std::vector<float> input(seqLen * JOINTS_DIM, 0.0f);

    int numFrames = static_cast<int>(frames.size());
    if (numFrames == 0) return input;

    // Compute mean hip-to-head distance for scale normalisation
    float meanHeight = 0.0f;
    int headIdx = static_cast<int>(Joint::Head);
    int hipsIdx = static_cast<int>(Joint::Hips);

    for (const auto& f : frames) {
        Vec3 delta = f.joints[headIdx].worldPosition - f.joints[hipsIdx].worldPosition;
        meanHeight += delta.length();
    }
    meanHeight /= static_cast<float>(numFrames);
    if (meanHeight < 1e-6f) meanHeight = 1.0f;

    for (int t = 0; t < seqLen; ++t) {
        int srcFrame = (t < numFrames) ? t : numFrames - 1;  // pad with last frame
        const auto& frame = frames[srcFrame];
        Vec3 hips = frame.joints[hipsIdx].worldPosition;

        for (int j = 0; j < JOINT_COUNT; ++j) {
            Vec3 pos = frame.joints[j].worldPosition - hips;
            pos = pos * (1.0f / meanHeight);

            int offset = t * JOINTS_DIM + j * 3;
            input[offset + 0] = pos.x;
            input[offset + 1] = pos.y;
            input[offset + 2] = pos.z;
        }
    }

    return input;
}

std::array<float, MOTION_EMBEDDING_DIM> MotionEmbedder::embed(
    const std::vector<SkeletonFrame>& frames) const {

    std::array<float, MOTION_EMBEDDING_DIM> result{};

    if (!impl_->initialized || frames.empty()) return result;

    // Prepare input
    auto inputData = prepareInput(frames, impl_->config.seqLen);

    // Try ONNX inference
#ifdef HM_HAS_ONNXRUNTIME
    if (impl_->onnx.isLoaded()) {
        std::array<int64_t, 3> inputShape = {
            1,
            static_cast<int64_t>(impl_->config.seqLen),
            static_cast<int64_t>(JOINTS_DIM)
        };

        auto& memInfo = impl_->onnx.memoryInfo();
        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memInfo, inputData.data(),
            inputData.size(),
            inputShape.data(), inputShape.size()));

        auto outputs = impl_->onnx.run(inputs);
        if (!outputs.empty()) {
            const float* outputData = outputs[0].GetTensorData<float>();
            for (int i = 0; i < MOTION_EMBEDDING_DIM; ++i) {
                result[i] = outputData[i];
            }
            return result;
        }
    }
#endif

    // Fallback: compute a simple feature-based embedding
    // This is a deterministic fallback when no ONNX model is available
    if (frames.size() >= 2) {
        // Use averaged normalised positions as a simple embedding
        int seqLen = impl_->config.seqLen;
        int step = std::max(1, static_cast<int>(frames.size()) / seqLen);

        for (size_t f = 0; f < frames.size(); f += step) {
            const auto& frame = frames[f];
            Vec3 hips = frame.joints[static_cast<int>(Joint::Hips)].worldPosition;
            for (int j = 0; j < JOINT_COUNT && j * 3 < MOTION_EMBEDDING_DIM; ++j) {
                Vec3 pos = frame.joints[j].worldPosition - hips;
                int base = j * 3;
                if (base + 2 < MOTION_EMBEDDING_DIM) {
                    result[base] += pos.x;
                    result[base + 1] += pos.y;
                    result[base + 2] += pos.z;
                }
            }
        }

        // Add velocity information in remaining dimensions
        float avgSpeed = 0.0f;
        for (const auto& f : frames) {
            avgSpeed += f.rootVelocity.length();
        }
        avgSpeed /= static_cast<float>(frames.size());
        if (66 < MOTION_EMBEDDING_DIM) result[66] = avgSpeed / 500.0f;

        // L2 normalize
        float norm = 0.0f;
        for (float v : result) norm += v * v;
        norm = std::sqrt(norm);
        if (norm > 1e-8f) {
            for (float& v : result) v /= norm;
        }
    }

    return result;
}

std::array<float, MOTION_EMBEDDING_DIM> MotionEmbedder::embedClip(
    const AnimClip& clip) const {
    return embed(clip.frames);
}

std::vector<std::array<float, MOTION_EMBEDDING_DIM>> MotionEmbedder::embedBatch(
    const std::vector<AnimClip>& clips) const {

    std::vector<std::array<float, MOTION_EMBEDDING_DIM>> results;
    results.reserve(clips.size());
    for (const auto& clip : clips) {
        results.push_back(embedClip(clip));
    }
    return results;
}

} // namespace hm::analysis
