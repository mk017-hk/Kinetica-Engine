#pragma once

#include <vector>
#include <cmath>
#include <string>

namespace hm::ml {

/// Beta schedule type for the diffusion process.
enum class BetaSchedule {
    Linear,         // Linear interpolation from betaStart to betaEnd
    Cosine,         // Cosine schedule (Nichol & Dhariwal, 2021)
    Quadratic,      // Quadratic schedule for gentler noise
    Sigmoid         // Sigmoid schedule for concentrated noise at center
};

/// Full-featured DDPM/DDIM noise scheduler.
///
/// Precomputes all required diffusion coefficients at construction time:
///   - betas, alphas, alphas_cumprod, alphas_cumprod_prev
///   - sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod
///   - posterior mean/variance coefficients for DDPM
///   - signal-to-noise ratios (SNR)
///
/// Supports both DDPM stochastic and DDIM deterministic sampling.
class NoiseScheduler {
public:
    /// Construct with configurable schedule.
    explicit NoiseScheduler(int numTimesteps = 1000,
                             float betaStart = 0.0001f,
                             float betaEnd = 0.02f,
                             BetaSchedule schedule = BetaSchedule::Linear);

    // -----------------------------------------------------------------
    // Schedule access
    // -----------------------------------------------------------------

    /// Beta value at timestep t.
    float beta(int t) const { return betas_[t]; }

    /// Alpha = 1 - beta at timestep t.
    float alpha(int t) const { return alphas_[t]; }

    /// Cumulative product of alphas up to timestep t: prod_{i=0}^{t} alpha_i
    float alphasCumprod(int t) const { return alphasCumprod_[t]; }

    /// Cumulative product shifted by one (alphasCumprod[t-1], with alphasCumprod[-1] = 1.0).
    float alphasCumprodPrev(int t) const { return alphasCumprodPrev_[t]; }

    /// sqrt(alphas_cumprod[t])
    float sqrtAlphasCumprod(int t) const { return sqrtAlphasCumprod_[t]; }

    /// sqrt(1 - alphas_cumprod[t])
    float sqrtOneMinusAlphasCumprod(int t) const { return sqrtOneMinusAlphasCumprod_[t]; }

    /// 1 / sqrt(alphas_cumprod[t])  (used in x0 prediction)
    float recipSqrtAlphasCumprod(int t) const { return recipSqrtAlphasCumprod_[t]; }

    /// sqrt(1/alphas_cumprod[t] - 1)
    float sqrtRecipm1AlphasCumprod(int t) const { return sqrtRecipm1AlphasCumprod_[t]; }

    // -----------------------------------------------------------------
    // Signal-to-noise ratio
    // -----------------------------------------------------------------

    /// SNR(t) = alpha_bar_t / (1 - alpha_bar_t)
    float snr(int t) const { return snr_[t]; }

    /// Log-SNR in decibels: 10 * log10(SNR(t))
    float logSnrDb(int t) const;

    /// Interpolate noise level for a continuous (non-integer) timestep.
    /// Linearly interpolates alphasCumprod between floor(t) and ceil(t).
    float interpolateAlphasCumprod(float t) const;

    // -----------------------------------------------------------------
    // DDPM posterior
    // -----------------------------------------------------------------

    /// Posterior mean coefficient 1: beta_t * sqrt(alpha_bar_{t-1}) / (1 - alpha_bar_t)
    float posteriorMeanCoeff1(int t) const { return posteriorMeanCoeff1_[t]; }

    /// Posterior mean coefficient 2: (1 - alpha_bar_{t-1}) * sqrt(alpha_t) / (1 - alpha_bar_t)
    float posteriorMeanCoeff2(int t) const { return posteriorMeanCoeff2_[t]; }

    /// Posterior variance: beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
    float posteriorVariance(int t) const { return posteriorVariance_[t]; }

    /// Clipped log posterior variance (for numerical stability).
    float posteriorLogVarianceClipped(int t) const { return posteriorLogVarianceClipped_[t]; }

    /// Compute DDPM posterior mean from x_start and x_t.
    /// posteriorMean = coeff1 * x_start + coeff2 * x_t
    void ddpmPosteriorMean(const float* x_start, const float* x_t,
                           int numElements, int t, float* output) const;

    /// Full DDPM sampling step: given x_t and predicted noise, produce x_{t-1}.
    /// Adds stochastic noise scaled by posterior variance.
    /// noiseToAdd should be Gaussian noise (or nullptr for deterministic DDPM at t=0).
    void ddpmStep(const float* x_t, const float* predictedNoise,
                  const float* noiseToAdd, int numElements, int t,
                  float* output) const;

    // -----------------------------------------------------------------
    // DDIM sampling
    // -----------------------------------------------------------------

    /// DDIM deterministic sampling step (operates on raw float buffers).
    /// x_t and predictedNoise are flat arrays of the same length.
    /// Result is written into output (same length).
    /// eta controls stochasticity: 0 = deterministic DDIM, 1 = DDPM-equivalent.
    void ddimStep(const float* x_t, const float* predictedNoise,
                  int numElements, int currentStep, int nextStep,
                  float* output, float eta = 0.0f,
                  const float* noiseToAdd = nullptr) const;

    /// DDIM step that predicts x0 first and returns both x_{t-1} and predicted x0.
    void ddimStepWithX0(const float* x_t, const float* predictedNoise,
                        int numElements, int currentStep, int nextStep,
                        float* x_prev_out, float* x0_pred_out,
                        float eta = 0.0f,
                        const float* noiseToAdd = nullptr) const;

    /// Get DDIM inference sub-schedule (e.g. 50 evenly-spaced timesteps).
    /// Returns timesteps in descending order (largest first).
    std::vector<int> getDDIMSchedule(int numInferenceSteps = 50) const;

    /// Get DDIM schedule with uniform spacing variant.
    std::vector<int> getDDIMScheduleQuadratic(int numInferenceSteps = 50) const;

    // -----------------------------------------------------------------
    // Forward diffusion helpers
    // -----------------------------------------------------------------

    /// Add noise to clean data: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise
    void addNoise(const float* x0, const float* noise,
                  int numElements, int t, float* x_t) const;

    /// Predict x0 from x_t and predicted noise:
    /// x0 = (x_t - sqrt(1-alpha_bar_t) * noise) / sqrt(alpha_bar_t)
    void predictX0FromNoise(const float* x_t, const float* predictedNoise,
                            int numElements, int t, float* x0) const;

    /// Predict noise from x_t and x0:
    /// noise = (x_t - sqrt(alpha_bar_t) * x0) / sqrt(1-alpha_bar_t)
    void predictNoiseFromX0(const float* x_t, const float* x0,
                            int numElements, int t, float* noise) const;

    // -----------------------------------------------------------------
    // Misc
    // -----------------------------------------------------------------

    int numTimesteps() const { return numTimesteps_; }
    BetaSchedule scheduleType() const { return schedule_; }

    /// Return a readable summary of the schedule for logging.
    std::string summary() const;

private:
    void computeLinearSchedule(float betaStart, float betaEnd);
    void computeCosineSchedule();
    void computeQuadraticSchedule(float betaStart, float betaEnd);
    void computeSigmoidSchedule(float betaStart, float betaEnd);
    void computeDerivedQuantities();

    int numTimesteps_;
    BetaSchedule schedule_;

    // Primary schedule
    std::vector<float> betas_;
    std::vector<float> alphas_;
    std::vector<float> alphasCumprod_;
    std::vector<float> alphasCumprodPrev_;

    // Precomputed sqrt/reciprocal quantities
    std::vector<float> sqrtAlphasCumprod_;
    std::vector<float> sqrtOneMinusAlphasCumprod_;
    std::vector<float> recipSqrtAlphasCumprod_;
    std::vector<float> sqrtRecipm1AlphasCumprod_;

    // Posterior (DDPM)
    std::vector<float> posteriorMeanCoeff1_;
    std::vector<float> posteriorMeanCoeff2_;
    std::vector<float> posteriorVariance_;
    std::vector<float> posteriorLogVarianceClipped_;

    // SNR
    std::vector<float> snr_;
};

} // namespace hm::ml
