#pragma once

#include "HyperMotion/core/Types.h"
#include <torch/torch.h>

namespace hm::ml {

// Condition Encoder: Linear(78, 512) -> ReLU -> Linear(512, 512) -> ReLU -> Linear(512, 256)
struct ConditionEncoderImpl : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    ConditionEncoderImpl();
    torch::Tensor forward(torch::Tensor x);  // [batch, 78] -> [batch, 256]
};
TORCH_MODULE(ConditionEncoder);

} // namespace hm::ml
