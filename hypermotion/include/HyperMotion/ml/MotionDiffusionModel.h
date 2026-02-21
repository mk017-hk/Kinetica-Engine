#pragma once

#include "HyperMotion/core/Types.h"
#include "HyperMotion/ml/ConditionEncoder.h"
#include "HyperMotion/ml/NoiseScheduler.h"
#include "HyperMotion/ml/MotionTransformer.h"
#include <torch/torch.h>
#include <string>
#include <memory>

namespace hm::ml {

struct MotionDiffusionConfig {
    int motionDim = FRAME_DIM;      // 132
    int condDim = MotionCondition::DIM; // 78
    int seqLen = 64;                // Generated motion length
    int numTimesteps = 1000;
    int numInferenceSteps = 50;     // DDIM steps
    float betaStart = 0.0001f;
    float betaEnd = 0.02f;
    float learningRate = 1e-4f;
    int batchSize = 64;
    std::string modelSavePath;
};

class MotionDiffusionModel {
public:
    explicit MotionDiffusionModel(const MotionDiffusionConfig& config = {});
    ~MotionDiffusionModel();

    MotionDiffusionModel(const MotionDiffusionModel&) = delete;
    MotionDiffusionModel& operator=(const MotionDiffusionModel&) = delete;
    MotionDiffusionModel(MotionDiffusionModel&&) noexcept;
    MotionDiffusionModel& operator=(MotionDiffusionModel&&) noexcept;

    bool initialize();
    bool isInitialized() const;

    // Training: single step
    // x0: [batch, seqLen, 132], condition: [batch, 78]
    // Returns MSE loss
    torch::Tensor trainStep(const torch::Tensor& x0, const torch::Tensor& condition);

    // Inference: generate motion from noise + condition
    // condition: [batch, 78]
    // Returns: [batch, seqLen, 132]
    torch::Tensor generate(const torch::Tensor& condition);

    // Save/load model
    void save(const std::string& path);
    void load(const std::string& path);

    // Access components
    MotionTransformer& transformer();
    ConditionEncoder& condEncoder();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hm::ml
