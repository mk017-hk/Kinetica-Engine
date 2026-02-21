#pragma once

#include <torch/torch.h>
#include <vector>

namespace hm::ml {

class NoiseScheduler {
public:
    explicit NoiseScheduler(int numTimesteps = 1000,
                             float betaStart = 0.0001f,
                             float betaEnd = 0.02f);
    ~NoiseScheduler();

    // Add noise to clean data at timestep t
    // Returns noised data and the noise that was added
    struct NoisedSample {
        torch::Tensor noisedData;
        torch::Tensor noise;
    };
    NoisedSample addNoise(const torch::Tensor& x0, const torch::Tensor& t);

    // DDIM sampling step
    torch::Tensor ddimStep(const torch::Tensor& xt,
                           const torch::Tensor& predictedNoise,
                           int currentStep, int nextStep);

    // Get DDIM inference timestep schedule (e.g., 50 steps)
    std::vector<int> getDDIMSchedule(int numInferenceSteps = 50) const;

    int numTimesteps() const { return numTimesteps_; }

    // Access precomputed schedule values
    torch::Tensor alphasCumprod() const { return alphasCumprod_; }

private:
    int numTimesteps_;
    torch::Tensor betas_;
    torch::Tensor alphas_;
    torch::Tensor alphasCumprod_;
    torch::Tensor sqrtAlphasCumprod_;
    torch::Tensor sqrtOneMinusAlphasCumprod_;
};

} // namespace hm::ml
