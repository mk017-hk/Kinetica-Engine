#include "HyperMotion/ml/NoiseScheduler.h"

#include <algorithm>
#include <numeric>

namespace hm::ml {

NoiseScheduler::NoiseScheduler(int numTimesteps, float betaStart, float betaEnd)
    : numTimesteps_(numTimesteps), alphasCumprod_(numTimesteps) {

    // Linear beta schedule -> cumulative product of alphas
    float step = (betaEnd - betaStart) / static_cast<float>(numTimesteps - 1);
    float cumprod = 1.0f;
    for (int i = 0; i < numTimesteps; ++i) {
        float beta = betaStart + step * static_cast<float>(i);
        cumprod *= (1.0f - beta);
        alphasCumprod_[i] = cumprod;
    }
}

void NoiseScheduler::ddimStep(const float* x_t, const float* predictedNoise,
                               int numElements, int currentStep, int nextStep,
                               float* output) const {

    float alphaCur  = alphasCumprod_[currentStep];
    float alphaNext = (nextStep >= 0) ? alphasCumprod_[nextStep] : 1.0f;

    float sqrtAlphaCur     = std::sqrt(alphaCur);
    float sqrt1mAlphaCur   = std::sqrt(1.0f - alphaCur);
    float sqrtAlphaNext    = std::sqrt(alphaNext);
    float sqrt1mAlphaNext  = std::sqrt(1.0f - alphaNext);

    for (int i = 0; i < numElements; ++i) {
        // Predict x0
        float x0 = (x_t[i] - sqrt1mAlphaCur * predictedNoise[i]) / sqrtAlphaCur;
        // Clamp for stability
        x0 = std::clamp(x0, -5.0f, 5.0f);
        // DDIM deterministic step
        output[i] = sqrtAlphaNext * x0 + sqrt1mAlphaNext * predictedNoise[i];
    }
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
