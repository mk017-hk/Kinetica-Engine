#include "HyperMotion/ml/NoiseScheduler.h"

#include <cmath>
#include <algorithm>

namespace hm::ml {

NoiseScheduler::NoiseScheduler(int numTimesteps, float betaStart, float betaEnd)
    : numTimesteps_(numTimesteps) {
    // Linear beta schedule
    betas_ = torch::linspace(betaStart, betaEnd, numTimesteps);
    alphas_ = 1.0f - betas_;
    alphasCumprod_ = torch::cumprod(alphas_, 0);
    sqrtAlphasCumprod_ = torch::sqrt(alphasCumprod_);
    sqrtOneMinusAlphasCumprod_ = torch::sqrt(1.0f - alphasCumprod_);
}

NoiseScheduler::~NoiseScheduler() = default;

NoiseScheduler::NoisedSample NoiseScheduler::addNoise(
    const torch::Tensor& x0, const torch::Tensor& t) {
    // x0: [batch, ...], t: [batch] (integer timesteps)

    auto sqrtAlpha = sqrtAlphasCumprod_.index_select(0, t);
    auto sqrtOneMinusAlpha = sqrtOneMinusAlphasCumprod_.index_select(0, t);

    // Reshape for broadcasting
    auto shape = x0.sizes().vec();
    std::vector<int64_t> broadcastShape(shape.size(), 1);
    broadcastShape[0] = t.size(0);

    sqrtAlpha = sqrtAlpha.view(broadcastShape);
    sqrtOneMinusAlpha = sqrtOneMinusAlpha.view(broadcastShape);

    auto noise = torch::randn_like(x0);
    auto noisedData = sqrtAlpha * x0 + sqrtOneMinusAlpha * noise;

    return {noisedData, noise};
}

torch::Tensor NoiseScheduler::ddimStep(
    const torch::Tensor& xt,
    const torch::Tensor& predictedNoise,
    int currentStep, int nextStep) {

    float alphaCurrent = alphasCumprod_[currentStep].item<float>();
    float alphaNext = (nextStep >= 0) ? alphasCumprod_[nextStep].item<float>() : 1.0f;

    // DDIM deterministic sampling (eta=0)
    // x_{t-1} = sqrt(alpha_{t-1}) * predicted_x0 + sqrt(1-alpha_{t-1}) * direction
    float sqrtAlphaCurrent = std::sqrt(alphaCurrent);
    float sqrtOneMinusAlphaCurrent = std::sqrt(1.0f - alphaCurrent);

    // Predicted x0
    auto predictedX0 = (xt - sqrtOneMinusAlphaCurrent * predictedNoise) / sqrtAlphaCurrent;

    // Clamp predicted x0 for stability
    predictedX0 = torch::clamp(predictedX0, -5.0f, 5.0f);

    // Direction pointing to xt
    float sqrtOneMinusAlphaNext = std::sqrt(1.0f - alphaNext);
    auto direction = sqrtOneMinusAlphaNext * predictedNoise;

    // Next sample
    float sqrtAlphaNext = std::sqrt(alphaNext);
    auto xNext = sqrtAlphaNext * predictedX0 + direction;

    return xNext;
}

std::vector<int> NoiseScheduler::getDDIMSchedule(int numInferenceSteps) const {
    std::vector<int> schedule;
    schedule.reserve(numInferenceSteps);

    float stepSize = static_cast<float>(numTimesteps_) / numInferenceSteps;
    for (int i = numInferenceSteps - 1; i >= 0; --i) {
        int t = static_cast<int>(i * stepSize);
        t = std::clamp(t, 0, numTimesteps_ - 1);
        schedule.push_back(t);
    }

    return schedule;
}

} // namespace hm::ml
