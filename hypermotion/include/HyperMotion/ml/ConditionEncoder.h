#pragma once

#include "HyperMotion/core/Types.h"

#ifdef HM_HAS_TORCH
#include <torch/torch.h>

namespace hm::ml {

/// Condition encoder: 78D MotionCondition -> 256D latent.
/// Linear(78,512) -> ReLU -> Linear(512,512) -> ReLU -> Linear(512,256)
struct ConditionEncoderImpl : torch::nn::Module {
    ConditionEncoderImpl(int inputDim = MotionCondition::DIM,
                          int hiddenDim = 512,
                          int outputDim = 256);

    torch::Tensor forward(torch::Tensor x);

private:
    torch::nn::Linear fc1_{nullptr}, fc2_{nullptr}, fc3_{nullptr};
};

TORCH_MODULE(ConditionEncoder);

} // namespace hm::ml

#else

namespace hm::ml {
// ConditionEncoder requires LibTorch for training.
// At inference time it is fused into the ONNX denoiser graph.
} // namespace hm::ml

#endif
