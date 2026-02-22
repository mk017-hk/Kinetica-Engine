#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/ml/OnnxInference.h"
#include <string>
#include <memory>
#include <vector>

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

    // Training parameters (only used when HM_HAS_TORCH)
    float learningRate = 1e-4f;
    int batchSize = 64;
};

/// Diffusion model with dual-mode support:
///   - **Training** (HM_HAS_TORCH): LibTorch MotionTransformer + ConditionEncoder
///   - **Inference**: ONNX Runtime denoiser + pure-math DDIM schedule
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

    // ---- ONNX Inference API (always available) ----

    /// Generate motion via DDIM sampling (requires ONNX model).
    std::vector<SkeletonFrame> generate(const std::vector<float>& condition);
    std::vector<float> generateRaw(const std::vector<float>& condition);

#ifdef HM_HAS_TORCH
    // ---- LibTorch Training API ----

    /// Access the transformer module for optimizer parameter registration.
    MotionTransformer transformer();

    /// Access the condition encoder module for optimizer parameter registration.
    ConditionEncoder condEncoder();

    /// One training step: sample noise, add to x0, predict, return MSE loss.
    /// @param x0   Clean motion data [B, seqLen, motionDim]
    /// @param cond Raw condition [B, condDim] (encoded internally)
    torch::Tensor trainStep(torch::Tensor x0, torch::Tensor cond);

    /// Save both transformer and condition encoder weights.
    void save(const std::string& path);
#endif

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::ml
