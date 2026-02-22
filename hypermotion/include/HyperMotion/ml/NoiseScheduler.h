#pragma once

#include <vector>
#include <cmath>

namespace hm::ml {

/// Pure-math DDIM noise scheduler for inference.
/// No neural network weights — just the beta/alpha schedule and DDIM stepping.
/// This replaces the old LibTorch-based NoiseScheduler.
class NoiseScheduler {
public:
    explicit NoiseScheduler(int numTimesteps = 1000,
                             float betaStart = 0.0001f,
                             float betaEnd = 0.02f);

    /// Precomputed alpha cumulative product at timestep t.
    float alphasCumprod(int t) const { return alphasCumprod_[t]; }

    /// DDIM deterministic sampling step (operates on raw float buffers).
    /// x_t and predictedNoise are flat arrays of the same length.
    /// Result is written into output (same length).
    void ddimStep(const float* x_t, const float* predictedNoise,
                  int numElements, int currentStep, int nextStep,
                  float* output) const;

    /// Get DDIM inference sub-schedule (e.g. 50 evenly-spaced timesteps).
    std::vector<int> getDDIMSchedule(int numInferenceSteps = 50) const;

    int numTimesteps() const { return numTimesteps_; }

private:
    int numTimesteps_;
    std::vector<float> alphasCumprod_;
};

} // namespace hm::ml
