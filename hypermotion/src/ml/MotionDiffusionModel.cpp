#include "HyperMotion/ml/MotionDiffusionModel.h"
#include "HyperMotion/core/Logger.h"

#include <chrono>

namespace hm::ml {

static constexpr const char* TAG = "MotionDiffusionModel";

struct MotionDiffusionModel::Impl {
    MotionDiffusionConfig config;
    ConditionEncoder condEncoder;
    MotionTransformer transformer;
    NoiseScheduler scheduler;
    bool initialized = false;

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
    try {
        impl_->condEncoder = ConditionEncoder();
        impl_->transformer = MotionTransformer(
            impl_->config.motionDim, 256, 512, 8, 8, 2048, 0.1f);

        impl_->initialized = true;
        HM_LOG_INFO(TAG, "Motion diffusion model initialized");
        return true;
    } catch (const std::exception& e) {
        HM_LOG_ERROR(TAG, std::string("Initialization failed: ") + e.what());
        return false;
    }
}

bool MotionDiffusionModel::isInitialized() const {
    return impl_->initialized;
}

torch::Tensor MotionDiffusionModel::trainStep(
    const torch::Tensor& x0, const torch::Tensor& condition) {

    // x0: [batch, seqLen, 132], condition: [batch, 78]
    int batchSize = x0.size(0);

    // Sample random timesteps
    auto t = torch::randint(0, impl_->config.numTimesteps, {batchSize},
                            torch::TensorOptions().dtype(torch::kLong));

    // Add noise
    auto [noisedData, noise] = impl_->scheduler.addNoise(x0, t);

    // Encode condition
    auto condEncoded = impl_->condEncoder->forward(condition);

    // Predict noise
    auto predictedNoise = impl_->transformer->forward(noisedData, t.to(torch::kFloat32), condEncoded);

    // MSE loss
    auto loss = torch::mse_loss(predictedNoise, noise);

    return loss;
}

torch::Tensor MotionDiffusionModel::generate(const torch::Tensor& condition) {
    if (!impl_->initialized) {
        HM_LOG_ERROR(TAG, "Model not initialized");
        return {};
    }

    torch::NoGradGuard noGrad;
    impl_->condEncoder->eval();
    impl_->transformer->eval();

    int batchSize = condition.size(0);
    auto device = condition.device();

    // Encode condition
    auto condEncoded = impl_->condEncoder->forward(condition);

    // Start from pure noise
    auto xt = torch::randn({batchSize, impl_->config.seqLen, impl_->config.motionDim},
                           torch::TensorOptions().device(device));

    // DDIM sampling schedule
    auto schedule = impl_->scheduler.getDDIMSchedule(impl_->config.numInferenceSteps);

    for (size_t i = 0; i < schedule.size(); ++i) {
        int currentStep = schedule[i];
        int nextStep = (i + 1 < schedule.size()) ? schedule[i + 1] : -1;

        auto tTensor = torch::full({batchSize}, static_cast<float>(currentStep),
                                    torch::TensorOptions().device(device));

        auto predictedNoise = impl_->transformer->forward(xt, tTensor, condEncoded);

        xt = impl_->scheduler.ddimStep(xt, predictedNoise, currentStep, nextStep);
    }

    return xt;  // [batch, seqLen, 132]
}

void MotionDiffusionModel::save(const std::string& path) {
    torch::serialize::OutputArchive archive;
    impl_->condEncoder->save(archive);
    impl_->transformer->save(archive);
    archive.save_to(path);
    HM_LOG_INFO(TAG, "Model saved to: " + path);
}

void MotionDiffusionModel::load(const std::string& path) {
    try {
        torch::serialize::InputArchive archive;
        archive.load_from(path);
        impl_->condEncoder->load(archive);
        impl_->transformer->load(archive);
        HM_LOG_INFO(TAG, "Model loaded from: " + path);
    } catch (const std::exception& e) {
        HM_LOG_ERROR(TAG, std::string("Failed to load model: ") + e.what());
    }
}

MotionTransformer& MotionDiffusionModel::transformer() {
    return impl_->transformer;
}

ConditionEncoder& MotionDiffusionModel::condEncoder() {
    return impl_->condEncoder;
}

} // namespace hm::ml
