#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/ml/OnnxInference.h"
#include <string>
#include <memory>
#include <vector>

namespace hm::ml {

struct MotionDiffusionConfig {
    int motionDim = FRAME_DIM;          // 132
    int condDim = MotionCondition::DIM; // 78
    int seqLen = 64;                    // Generated motion length
    int numTimesteps = 1000;
    int numInferenceSteps = 50;         // DDIM steps
    float betaStart = 0.0001f;
    float betaEnd = 0.02f;

    /// Path to the ONNX denoiser (exported from Python).
    /// The DDIM schedule is pure math and runs in C++.
    std::string onnxModelPath;
    bool useGPU = true;
};

/// Inference-only diffusion model.
///
/// The neural network (MotionTransformer + ConditionEncoder) is loaded from
/// an ONNX file produced by the Python training pipeline.  The DDIM sampling
/// loop and noise schedule are reimplemented in C++ (pure math, no weights).
class MotionDiffusionModel {
public:
    explicit MotionDiffusionModel(const MotionDiffusionConfig& config = {});
    ~MotionDiffusionModel();

    MotionDiffusionModel(const MotionDiffusionModel&) = delete;
    MotionDiffusionModel& operator=(const MotionDiffusionModel&) = delete;
    MotionDiffusionModel(MotionDiffusionModel&&) noexcept;
    MotionDiffusionModel& operator=(MotionDiffusionModel&&) noexcept;

    /// Load the ONNX denoiser and precompute the noise schedule.
    bool initialize();
    bool isInitialized() const;

    /// Generate motion via DDIM sampling.
    /// @param condition  Flat vector of MotionCondition::DIM floats.
    /// @return 64 frames of generated motion as SkeletonFrame vector.
    std::vector<SkeletonFrame> generate(const std::vector<float>& condition);

    /// Raw generation returning flat float buffer [seqLen * FRAME_DIM].
    std::vector<float> generateRaw(const std::vector<float>& condition);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::ml
