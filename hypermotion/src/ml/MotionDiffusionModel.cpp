#include "HyperMotion/ml/MotionDiffusionModel.h"
#include "HyperMotion/ml/NoiseScheduler.h"
#include "HyperMotion/core/Logger.h"
#include "HyperMotion/core/MathUtils.h"

#include <chrono>
#include <random>

namespace hm::ml {

static constexpr const char* TAG = "MotionDiffusionModel";

struct MotionDiffusionModel::Impl {
    MotionDiffusionConfig config;
    OnnxInference denoiser;
    NoiseScheduler scheduler;
    bool initialized = false;
    std::mt19937 rng{std::random_device{}()};

    Impl(const MotionDiffusionConfig& cfg)
        : config(cfg)
        , scheduler(cfg.numTimesteps, cfg.betaStart, cfg.betaEnd) {}
};

MotionDiffusionModel::MotionDiffusionModel(const MotionDiffusionConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

MotionDiffusionModel::~MotionDiffusionModel() = default;
MotionDiffusionModel::MotionDiffusionModel(MotionDiffusionModel&&) noexcept = default;
MotionDiffusionModel& MotionDiffusionModel::operator=(MotionDiffusionModel&&) noexcept = default;

bool MotionDiffusionModel::initialize() {
    if (impl_->config.onnxModelPath.empty()) {
        HM_LOG_ERROR(TAG, "No ONNX model path specified");
        return false;
    }

    if (!impl_->denoiser.load(impl_->config.onnxModelPath, impl_->config.useGPU)) {
        return false;
    }

    impl_->initialized = true;
    HM_LOG_INFO(TAG, "Diffusion model initialized (ONNX inference)");
    return true;
}

bool MotionDiffusionModel::isInitialized() const {
    return impl_->initialized;
}

std::vector<float> MotionDiffusionModel::generateRaw(const std::vector<float>& condition) {
    if (!impl_->initialized) {
        HM_LOG_ERROR(TAG, "Model not initialized");
        return {};
    }

    auto& cfg = impl_->config;
    const int totalElements = cfg.seqLen * cfg.motionDim;

    // Start from Gaussian noise
    std::vector<float> x(totalElements);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto& v : x) v = dist(impl_->rng);

    // Condition vector (pad/truncate to condDim)
    std::vector<float> cond(cfg.condDim, 0.0f);
    size_t copyLen = std::min(condition.size(), static_cast<size_t>(cfg.condDim));
    std::copy_n(condition.begin(), copyLen, cond.begin());

    // DDIM schedule
    auto schedule = impl_->scheduler.getDDIMSchedule(cfg.numInferenceSteps);
    std::vector<float> predictedNoise(totalElements);
    std::vector<float> xNext(totalElements);

    auto& memInfo = impl_->denoiser.memoryInfo();

    for (size_t i = 0; i < schedule.size(); ++i) {
        int currentStep = schedule[i];
        int nextStep = (i + 1 < schedule.size()) ? schedule[i + 1] : -1;

        // Build ONNX input tensors
        std::vector<int64_t> motionShape = {1, static_cast<int64_t>(cfg.seqLen),
                                             static_cast<int64_t>(cfg.motionDim)};
        std::vector<int64_t> tShape = {1};
        std::vector<int64_t> condShape = {1, static_cast<int64_t>(cfg.condDim)};

        int64_t tVal = static_cast<int64_t>(currentStep);

        std::vector<Ort::Value> inputs;
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memInfo, x.data(), totalElements, motionShape.data(), motionShape.size()));
        inputs.push_back(Ort::Value::CreateTensor<int64_t>(
            memInfo, &tVal, 1, tShape.data(), tShape.size()));
        inputs.push_back(Ort::Value::CreateTensor<float>(
            memInfo, cond.data(), cfg.condDim, condShape.data(), condShape.size()));

        // Run denoiser
        auto outputs = impl_->denoiser.run(inputs);

        // Copy predicted noise
        const float* noiseData = outputs[0].GetTensorData<float>();
        std::copy_n(noiseData, totalElements, predictedNoise.begin());

        // DDIM step
        impl_->scheduler.ddimStep(x.data(), predictedNoise.data(),
                                   totalElements, currentStep, nextStep, xNext.data());
        std::swap(x, xNext);
    }

    return x;  // [seqLen * motionDim] flat
}

std::vector<SkeletonFrame> MotionDiffusionModel::generate(const std::vector<float>& condition) {
    auto raw = generateRaw(condition);
    if (raw.empty()) return {};

    auto& cfg = impl_->config;
    std::vector<SkeletonFrame> frames(cfg.seqLen);

    for (int f = 0; f < cfg.seqLen; ++f) {
        frames[f].frameIndex = f;
        frames[f].timestamp = static_cast<double>(f) / 30.0;

        const float* frameData = raw.data() + f * FRAME_DIM;
        for (int j = 0; j < JOINT_COUNT; ++j) {
            Vec6 rot6d;
            for (int k = 0; k < ROTATION_DIM; ++k) {
                rot6d[k] = frameData[j * ROTATION_DIM + k];
            }
            frames[f].joints[j].rotation6D = rot6d;
            frames[f].joints[j].localRotation = MathUtils::rotation6DToQuat(rot6d);
            frames[f].joints[j].localEulerDeg = MathUtils::quatToEulerDeg(
                frames[f].joints[j].localRotation);
            frames[f].joints[j].confidence = 1.0f;
        }
    }

    return frames;
}

} // namespace hm::ml
