#include "HyperMotion/ml/NoiseScheduler.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <sstream>

namespace hm::ml {

// =====================================================================
// Construction
// =====================================================================

NoiseScheduler::NoiseScheduler(int numTimesteps, float betaStart, float betaEnd,
                               BetaSchedule schedule)
    : numTimesteps_(numTimesteps), schedule_(schedule) {

    // Allocate all vectors
    betas_.resize(numTimesteps);
    alphas_.resize(numTimesteps);
    alphasCumprod_.resize(numTimesteps);
    alphasCumprodPrev_.resize(numTimesteps);
    sqrtAlphasCumprod_.resize(numTimesteps);
    sqrtOneMinusAlphasCumprod_.resize(numTimesteps);
    recipSqrtAlphasCumprod_.resize(numTimesteps);
    sqrtRecipm1AlphasCumprod_.resize(numTimesteps);
    posteriorMeanCoeff1_.resize(numTimesteps);
    posteriorMeanCoeff2_.resize(numTimesteps);
    posteriorVariance_.resize(numTimesteps);
    posteriorLogVarianceClipped_.resize(numTimesteps);
    snr_.resize(numTimesteps);

    // Compute the beta schedule
    switch (schedule) {
        case BetaSchedule::Cosine:
            computeCosineSchedule();
            break;
        case BetaSchedule::Quadratic:
            computeQuadraticSchedule(betaStart, betaEnd);
            break;
        case BetaSchedule::Sigmoid:
            computeSigmoidSchedule(betaStart, betaEnd);
            break;
        case BetaSchedule::Linear:
        default:
            computeLinearSchedule(betaStart, betaEnd);
            break;
    }

    // Compute all derived quantities from betas
    computeDerivedQuantities();
}

// =====================================================================
// Schedule computation
// =====================================================================

void NoiseScheduler::computeLinearSchedule(float betaStart, float betaEnd) {
    float step = (betaEnd - betaStart) / static_cast<float>(numTimesteps_ - 1);
    for (int i = 0; i < numTimesteps_; ++i) {
        betas_[i] = betaStart + step * static_cast<float>(i);
    }
}

void NoiseScheduler::computeCosineSchedule() {
    // Cosine schedule from "Improved Denoising Diffusion Probabilistic Models"
    // (Nichol & Dhariwal, 2021)
    // alpha_bar(t) = f(t) / f(0),  where f(t) = cos((t/T + s) / (1+s) * pi/2)^2
    constexpr float s = 0.008f;
    constexpr float maxBeta = 0.999f;

    auto f = [&](float t) -> float {
        float val = (t / static_cast<float>(numTimesteps_) + s) / (1.0f + s);
        float cosVal = std::cos(val * static_cast<float>(M_PI) * 0.5f);
        return cosVal * cosVal;
    };

    float f0 = f(0.0f);
    std::vector<float> alphaBars(numTimesteps_);
    for (int i = 0; i < numTimesteps_; ++i) {
        alphaBars[i] = f(static_cast<float>(i + 1)) / f0;
    }

    // Compute betas from alpha_bars: beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}
    for (int i = 0; i < numTimesteps_; ++i) {
        float alphaBarPrev = (i == 0) ? 1.0f : alphaBars[i - 1];
        betas_[i] = std::clamp(1.0f - alphaBars[i] / alphaBarPrev, 0.0f, maxBeta);
    }
}

void NoiseScheduler::computeQuadraticSchedule(float betaStart, float betaEnd) {
    // Quadratic interpolation: beta(t) = (sqrt(betaStart) + t/(T-1) * (sqrt(betaEnd) - sqrt(betaStart)))^2
    float sqrtStart = std::sqrt(betaStart);
    float sqrtEnd = std::sqrt(betaEnd);
    for (int i = 0; i < numTimesteps_; ++i) {
        float frac = static_cast<float>(i) / static_cast<float>(numTimesteps_ - 1);
        float sqrtBeta = sqrtStart + frac * (sqrtEnd - sqrtStart);
        betas_[i] = sqrtBeta * sqrtBeta;
    }
}

void NoiseScheduler::computeSigmoidSchedule(float betaStart, float betaEnd) {
    // Sigmoid schedule: concentrate beta changes around the midpoint
    // beta(t) = sigmoid(-6 + 12 * t/(T-1)) * (betaEnd - betaStart) + betaStart
    for (int i = 0; i < numTimesteps_; ++i) {
        float frac = static_cast<float>(i) / static_cast<float>(numTimesteps_ - 1);
        float sig = 1.0f / (1.0f + std::exp(-(-6.0f + 12.0f * frac)));
        betas_[i] = sig * (betaEnd - betaStart) + betaStart;
    }
}

void NoiseScheduler::computeDerivedQuantities() {
    constexpr float kEps = 1e-20f;

    // Alphas and cumulative product
    float cumprod = 1.0f;
    for (int i = 0; i < numTimesteps_; ++i) {
        alphas_[i] = 1.0f - betas_[i];
        cumprod *= alphas_[i];
        alphasCumprod_[i] = cumprod;
        alphasCumprodPrev_[i] = (i == 0) ? 1.0f : alphasCumprod_[i - 1];
    }

    // Precomputed square roots and reciprocals
    for (int i = 0; i < numTimesteps_; ++i) {
        float ac = alphasCumprod_[i];
        float oneMinusAc = 1.0f - ac;

        sqrtAlphasCumprod_[i] = std::sqrt(ac);
        sqrtOneMinusAlphasCumprod_[i] = std::sqrt(oneMinusAc);

        float recipSqrt = 1.0f / std::sqrt(std::max(ac, kEps));
        recipSqrtAlphasCumprod_[i] = recipSqrt;
        sqrtRecipm1AlphasCumprod_[i] = std::sqrt(std::max(1.0f / std::max(ac, kEps) - 1.0f, 0.0f));

        // SNR = alpha_bar / (1 - alpha_bar)
        snr_[i] = ac / std::max(oneMinusAc, kEps);
    }

    // DDPM posterior coefficients
    // q(x_{t-1} | x_t, x_0) = N(x_{t-1}; posteriorMean, posteriorVariance)
    // posteriorMean = coeff1 * x_0 + coeff2 * x_t
    for (int i = 0; i < numTimesteps_; ++i) {
        float beta = betas_[i];
        float acPrev = alphasCumprodPrev_[i];
        float ac = alphasCumprod_[i];
        float oneMinusAc = 1.0f - ac;

        // coeff1 = beta_t * sqrt(alpha_bar_{t-1}) / (1 - alpha_bar_t)
        posteriorMeanCoeff1_[i] = beta * std::sqrt(acPrev) / std::max(oneMinusAc, kEps);

        // coeff2 = (1 - alpha_bar_{t-1}) * sqrt(alpha_t) / (1 - alpha_bar_t)
        posteriorMeanCoeff2_[i] = (1.0f - acPrev) * std::sqrt(alphas_[i]) / std::max(oneMinusAc, kEps);

        // posterior variance = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        float pv = beta * (1.0f - acPrev) / std::max(oneMinusAc, kEps);
        posteriorVariance_[i] = pv;

        // Clipped log for numerical stability (clamp variance to at least 1e-20)
        posteriorLogVarianceClipped_[i] = std::log(std::max(pv, kEps));
    }
}

// =====================================================================
// SNR
// =====================================================================

float NoiseScheduler::logSnrDb(int t) const {
    constexpr float kEps = 1e-20f;
    return 10.0f * std::log10(std::max(snr_[t], kEps));
}

float NoiseScheduler::interpolateAlphasCumprod(float t) const {
    if (t <= 0.0f) return alphasCumprod_[0];
    if (t >= static_cast<float>(numTimesteps_ - 1)) return alphasCumprod_[numTimesteps_ - 1];

    int lo = static_cast<int>(std::floor(t));
    int hi = lo + 1;
    float frac = t - static_cast<float>(lo);

    return alphasCumprod_[lo] * (1.0f - frac) + alphasCumprod_[hi] * frac;
}

// =====================================================================
// Forward diffusion helpers
// =====================================================================

void NoiseScheduler::addNoise(const float* x0, const float* noise,
                               int numElements, int t, float* x_t) const {
    float sqrtAC = sqrtAlphasCumprod_[t];
    float sqrt1mAC = sqrtOneMinusAlphasCumprod_[t];

    for (int i = 0; i < numElements; ++i) {
        x_t[i] = sqrtAC * x0[i] + sqrt1mAC * noise[i];
    }
}

void NoiseScheduler::predictX0FromNoise(const float* x_t, const float* predictedNoise,
                                         int numElements, int t, float* x0) const {
    float recipSqrt = recipSqrtAlphasCumprod_[t];
    float sqrtRm1 = sqrtRecipm1AlphasCumprod_[t];

    for (int i = 0; i < numElements; ++i) {
        x0[i] = recipSqrt * x_t[i] - sqrtRm1 * predictedNoise[i];
    }
}

void NoiseScheduler::predictNoiseFromX0(const float* x_t, const float* x0,
                                         int numElements, int t, float* noise) const {
    float sqrtAC = sqrtAlphasCumprod_[t];
    float sqrt1mAC = sqrtOneMinusAlphasCumprod_[t];
    constexpr float kEps = 1e-12f;
    float invSqrt1mAC = 1.0f / std::max(sqrt1mAC, kEps);

    for (int i = 0; i < numElements; ++i) {
        noise[i] = (x_t[i] - sqrtAC * x0[i]) * invSqrt1mAC;
    }
}

// =====================================================================
// DDPM posterior
// =====================================================================

void NoiseScheduler::ddpmPosteriorMean(const float* x_start, const float* x_t,
                                        int numElements, int t, float* output) const {
    float c1 = posteriorMeanCoeff1_[t];
    float c2 = posteriorMeanCoeff2_[t];

    for (int i = 0; i < numElements; ++i) {
        output[i] = c1 * x_start[i] + c2 * x_t[i];
    }
}

void NoiseScheduler::ddpmStep(const float* x_t, const float* predictedNoise,
                               const float* noiseToAdd, int numElements, int t,
                               float* output) const {
    // First predict x0 from x_t and predicted noise
    std::vector<float> x0_pred(numElements);
    predictX0FromNoise(x_t, predictedNoise, numElements, t, x0_pred.data());

    // Clamp x0 for stability
    for (int i = 0; i < numElements; ++i) {
        x0_pred[i] = std::clamp(x0_pred[i], -5.0f, 5.0f);
    }

    // Compute posterior mean
    ddpmPosteriorMean(x0_pred.data(), x_t, numElements, t, output);

    // Add noise for t > 0
    if (t > 0 && noiseToAdd != nullptr) {
        float sigma = std::sqrt(posteriorVariance_[t]);
        for (int i = 0; i < numElements; ++i) {
            output[i] += sigma * noiseToAdd[i];
        }
    }
}

// =====================================================================
// DDIM sampling
// =====================================================================

void NoiseScheduler::ddimStep(const float* x_t, const float* predictedNoise,
                               int numElements, int currentStep, int nextStep,
                               float* output, float eta,
                               const float* noiseToAdd) const {

    float alphaCur  = alphasCumprod_[currentStep];
    float alphaNext = (nextStep >= 0) ? alphasCumprod_[nextStep] : 1.0f;

    float sqrtAlphaCur     = std::sqrt(alphaCur);
    float sqrt1mAlphaCur   = std::sqrt(1.0f - alphaCur);
    float sqrtAlphaNext    = std::sqrt(alphaNext);

    // Compute sigma for stochastic DDIM (eta > 0)
    // sigma_t = eta * sqrt((1 - alpha_{t-1}) / (1 - alpha_t)) * sqrt(1 - alpha_t / alpha_{t-1})
    float sigma = 0.0f;
    if (eta > 0.0f && nextStep >= 0) {
        float oneMinusAlphaNext = 1.0f - alphaNext;
        float oneMinusAlphaCur = 1.0f - alphaCur;
        float ratio = oneMinusAlphaNext / std::max(oneMinusAlphaCur, 1e-20f);
        float innerTerm = 1.0f - alphaCur / std::max(alphaNext, 1e-20f);
        sigma = eta * std::sqrt(std::max(ratio * innerTerm, 0.0f));
    }

    // Direction pointing to x_t, accounting for sigma
    float dirCoeff = std::sqrt(std::max(1.0f - alphaNext - sigma * sigma, 0.0f));

    for (int i = 0; i < numElements; ++i) {
        // Predict x0: x0 = (x_t - sqrt(1-alpha_t) * eps) / sqrt(alpha_t)
        float x0 = (x_t[i] - sqrt1mAlphaCur * predictedNoise[i]) / std::max(sqrtAlphaCur, 1e-12f);

        // Clamp for stability
        x0 = std::clamp(x0, -5.0f, 5.0f);

        // DDIM update:
        // x_{t-1} = sqrt(alpha_{t-1}) * x0 + dirCoeff * eps + sigma * noise
        output[i] = sqrtAlphaNext * x0 + dirCoeff * predictedNoise[i];

        // Add stochastic noise if eta > 0
        if (sigma > 0.0f && noiseToAdd != nullptr) {
            output[i] += sigma * noiseToAdd[i];
        }
    }
}

void NoiseScheduler::ddimStepWithX0(const float* x_t, const float* predictedNoise,
                                     int numElements, int currentStep, int nextStep,
                                     float* x_prev_out, float* x0_pred_out,
                                     float eta, const float* noiseToAdd) const {
    float alphaCur  = alphasCumprod_[currentStep];
    float alphaNext = (nextStep >= 0) ? alphasCumprod_[nextStep] : 1.0f;

    float sqrtAlphaCur     = std::sqrt(alphaCur);
    float sqrt1mAlphaCur   = std::sqrt(1.0f - alphaCur);
    float sqrtAlphaNext    = std::sqrt(alphaNext);

    // Compute sigma
    float sigma = 0.0f;
    if (eta > 0.0f && nextStep >= 0) {
        float oneMinusAlphaNext = 1.0f - alphaNext;
        float oneMinusAlphaCur = 1.0f - alphaCur;
        float ratio = oneMinusAlphaNext / std::max(oneMinusAlphaCur, 1e-20f);
        float innerTerm = 1.0f - alphaCur / std::max(alphaNext, 1e-20f);
        sigma = eta * std::sqrt(std::max(ratio * innerTerm, 0.0f));
    }

    float dirCoeff = std::sqrt(std::max(1.0f - alphaNext - sigma * sigma, 0.0f));

    for (int i = 0; i < numElements; ++i) {
        // Predict x0
        float x0 = (x_t[i] - sqrt1mAlphaCur * predictedNoise[i]) / std::max(sqrtAlphaCur, 1e-12f);
        x0 = std::clamp(x0, -5.0f, 5.0f);

        // Store x0 prediction
        x0_pred_out[i] = x0;

        // DDIM update
        x_prev_out[i] = sqrtAlphaNext * x0 + dirCoeff * predictedNoise[i];
        if (sigma > 0.0f && noiseToAdd != nullptr) {
            x_prev_out[i] += sigma * noiseToAdd[i];
        }
    }
}

// =====================================================================
// Schedule generation
// =====================================================================

std::vector<int> NoiseScheduler::getDDIMSchedule(int numInferenceSteps) const {
    // Uniform spacing: evenly spaced timesteps in descending order
    std::vector<int> schedule;
    schedule.reserve(numInferenceSteps);

    float stepSize = static_cast<float>(numTimesteps_) / static_cast<float>(numInferenceSteps);
    for (int i = numInferenceSteps - 1; i >= 0; --i) {
        int t = static_cast<int>(static_cast<float>(i) * stepSize);
        t = std::clamp(t, 0, numTimesteps_ - 1);
        schedule.push_back(t);
    }
    return schedule;
}

std::vector<int> NoiseScheduler::getDDIMScheduleQuadratic(int numInferenceSteps) const {
    // Quadratic spacing: more steps at lower noise levels (near t=0)
    // t_i = floor((i/N)^2 * T)
    std::vector<int> schedule;
    schedule.reserve(numInferenceSteps);

    for (int i = numInferenceSteps - 1; i >= 0; --i) {
        float frac = static_cast<float>(i) / static_cast<float>(numInferenceSteps);
        int t = static_cast<int>(frac * frac * static_cast<float>(numTimesteps_));
        t = std::clamp(t, 0, numTimesteps_ - 1);
        schedule.push_back(t);
    }

    // Remove duplicates while preserving order
    std::vector<int> unique;
    unique.reserve(schedule.size());
    for (int t : schedule) {
        if (unique.empty() || unique.back() != t) {
            unique.push_back(t);
        }
    }
    return unique;
}

// =====================================================================
// Summary
// =====================================================================

std::string NoiseScheduler::summary() const {
    std::ostringstream oss;
    oss << "NoiseScheduler(T=" << numTimesteps_
        << ", schedule=";

    switch (schedule_) {
        case BetaSchedule::Linear:    oss << "linear"; break;
        case BetaSchedule::Cosine:    oss << "cosine"; break;
        case BetaSchedule::Quadratic: oss << "quadratic"; break;
        case BetaSchedule::Sigmoid:   oss << "sigmoid"; break;
    }

    oss << ", beta=[" << betas_.front() << ", " << betas_.back() << "]"
        << ", alpha_bar=[" << alphasCumprod_.back() << ", " << alphasCumprod_.front() << "]"
        << ", SNR_dB=[" << logSnrDb(numTimesteps_ - 1) << ", " << logSnrDb(0) << "]"
        << ")";

    return oss.str();
}

} // namespace hm::ml
