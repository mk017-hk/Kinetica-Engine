#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/ml/OnnxInference.h"
#include <string>
#include <memory>
#include <vector>
#include <functional>

#ifdef HM_HAS_TORCH
#include <torch/torch.h>
#include "HyperMotion/ml/MotionTransformer.h"
#include "HyperMotion/ml/ConditionEncoder.h"
#endif

namespace hm::ml {

struct MotionDiffusionConfig {
    int motionDim = FRAME_DIM;          // 132
    int condDim = MotionCondition::DIM; // 78
    int seqLen = 64;                    // Generated motion length
    int numTimesteps = 1000;
    int numInferenceSteps = 50;         // DDIM steps
    float betaStart = 0.0001f;
    float betaEnd = 0.02f;

    /// Path to the ONNX denoiser (exported from Python/C++ training).
    /// Leave empty when training — the LibTorch modules are used instead.
    std::string onnxModelPath;
    bool useGPU = true;

    // Classifier-free guidance
    bool enableCFG = false;              // Enable classifier-free guidance
    float cfgScale = 7.5f;              // Guidance scale (1.0 = no guidance)
    float cfgDropoutRate = 0.1f;        // Probability of dropping condition during training

    // Training parameters (only used when HM_HAS_TORCH)
    float learningRate = 1e-4f;
    int batchSize = 64;
    float gradClipNorm = 1.0f;          // Max gradient norm for clipping
    float emaDecay = 0.9999f;           // EMA decay rate for model weights
    bool useEMA = true;                 // Enable exponential moving average
    float weightDecay = 0.0f;           // AdamW weight decay

    // DDIM stochasticity
    float ddimEta = 0.0f;              // 0 = deterministic, 1 = DDPM equivalent

    // Clamp predicted x0 for numerical stability
    float x0ClampMin = -5.0f;
    float x0ClampMax = 5.0f;
};

/// Progress callback for DDIM sampling: (currentStep, totalSteps)
using SamplingProgressCallback = std::function<void(int, int)>;

/// Diffusion model with dual-mode support:
///   - **Training** (HM_HAS_TORCH): LibTorch MotionTransformer + ConditionEncoder
///     with EMA, gradient clipping, and classifier-free guidance
///   - **Inference**: ONNX Runtime denoiser + pure-math DDIM schedule
///     with classifier-free guidance support
class MotionDiffusionModel {
public:
    explicit MotionDiffusionModel(const MotionDiffusionConfig& config = {});
    ~MotionDiffusionModel();

    MotionDiffusionModel(const MotionDiffusionModel&) = delete;
    MotionDiffusionModel& operator=(const MotionDiffusionModel&) = delete;
    MotionDiffusionModel(MotionDiffusionModel&&) noexcept;
    MotionDiffusionModel& operator=(MotionDiffusionModel&&) noexcept;

    /// Load ONNX model (inference) or create LibTorch modules (training).
    /// If onnxModelPath is empty AND HM_HAS_TORCH, initialises in training mode.
    bool initialize();
    bool isInitialized() const;

    /// Return the active config.
    const MotionDiffusionConfig& config() const;

    // ---- ONNX Inference API (always available) ----

    /// Generate motion via DDIM sampling (requires ONNX model).
    std::vector<SkeletonFrame> generate(const std::vector<float>& condition);
    std::vector<float> generateRaw(const std::vector<float>& condition);

    /// Generate with classifier-free guidance.
    /// Runs the denoiser twice per step (conditional + unconditional) and
    /// blends: eps = eps_uncond + cfgScale * (eps_cond - eps_uncond)
    std::vector<float> generateRawCFG(const std::vector<float>& condition,
                                       float cfgScale = -1.0f);

    /// Generate with a progress callback.
    std::vector<float> generateRawWithProgress(
        const std::vector<float>& condition,
        SamplingProgressCallback progressCb);

    /// Generate unconditional motion (zero condition).
    std::vector<float> generateUnconditional();

#ifdef HM_HAS_TORCH
    // ---- LibTorch Training API ----

    /// Access the transformer module for optimizer parameter registration.
    MotionTransformer transformer();

    /// Access the condition encoder module for optimizer parameter registration.
    ConditionEncoder condEncoder();

    /// One training step: sample noise, add to x0, predict, return MSE loss.
    /// Supports classifier-free guidance training (randomly drops condition).
    /// @param x0   Clean motion data [B, seqLen, motionDim]
    /// @param cond Raw condition [B, condDim] (encoded internally)
    torch::Tensor trainStep(torch::Tensor x0, torch::Tensor cond);

    /// Training step with gradient clipping.
    /// Returns {loss, grad_norm_before_clip}.
    std::pair<torch::Tensor, float> trainStepWithClipping(
        torch::Tensor x0, torch::Tensor cond,
        torch::optim::Optimizer& optimizer);

    /// Update EMA weights from current model weights.
    /// Should be called after each optimizer step.
    void updateEMA();

    /// Copy EMA weights to the model (for evaluation/export).
    void applyEMA();

    /// Restore original (non-EMA) weights.
    void restoreFromEMA();

    /// Save both transformer and condition encoder weights.
    void save(const std::string& path);

    /// Load pretrained weights from a saved checkpoint.
    /// @param path Path to the .pt file saved by save()
    /// @param strict If true, all keys must match; if false, missing keys are ignored
    bool load(const std::string& path, bool strict = true);

    /// Export the model to ONNX format for inference.
    /// @param onnxPath Output .onnx file path
    /// @param useEmaWeights If true, export EMA weights
    bool exportONNX(const std::string& onnxPath, bool useEmaWeights = true);

    /// Get current EMA decay rate.
    float emaDecay() const;

    /// Check if EMA shadow weights are available.
    bool hasEMAWeights() const;
#endif

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::ml
