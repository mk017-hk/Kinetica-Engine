#include "HyperMotion/ml/ConditionEncoder.h"

namespace hm::ml {

ConditionEncoderImpl::ConditionEncoderImpl() {
    fc1 = register_module("fc1", torch::nn::Linear(MotionCondition::DIM, 512));
    fc2 = register_module("fc2", torch::nn::Linear(512, 512));
    fc3 = register_module("fc3", torch::nn::Linear(512, 256));
}

torch::Tensor ConditionEncoderImpl::forward(torch::Tensor x) {
    x = torch::relu(fc1(x));
    x = torch::relu(fc2(x));
    x = torch::relu(fc3(x));
    return x;  // [batch, 256]
}

} // namespace hm::ml
