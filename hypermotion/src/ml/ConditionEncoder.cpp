#include "HyperMotion/ml/ConditionEncoder.h"

#ifdef HM_HAS_TORCH

namespace hm::ml {

ConditionEncoderImpl::ConditionEncoderImpl(int inputDim, int hiddenDim, int outputDim) {
    fc1_ = register_module("fc1", torch::nn::Linear(inputDim, hiddenDim));
    fc2_ = register_module("fc2", torch::nn::Linear(hiddenDim, hiddenDim));
    fc3_ = register_module("fc3", torch::nn::Linear(hiddenDim, outputDim));
}

torch::Tensor ConditionEncoderImpl::forward(torch::Tensor x) {
    x = torch::relu(fc1_->forward(x));
    x = torch::relu(fc2_->forward(x));
    x = fc3_->forward(x);
    return x;
}

} // namespace hm::ml

#endif
